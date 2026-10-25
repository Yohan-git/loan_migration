"""Synthetic rating models and their annual migration matrices.

Stands in for the SQL source: `to_matrix_table(models)` returns the matrices in the same flat format
(model, scenario, year, rank, 1..14) that `matrices.MigrationMatrices.from_frame` reads.

A rating model owns a through-the-cycle (TTC) matrix and, per scenario, three point-in-time
matrices (year 1, year 2, year 3+) derived with the one-factor Vasicek (Z-shift) approach.
Buckets are 1-based and ordered best -> worst; the last bucket is Default (absorbing).
"""
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from matrices import DEFAULT_SCENARIO

SEED = 42
SCENARIOS = ["base", "adverse", "severely_adverse"]
N_MATRIX_YEARS = 3  # year 1, year 2, year 3+
N_CLASSES = 14      # width of the matrix table: every model uses columns 1..14

# Credit-cycle index path per scenario: [year 1, year 2, year 3+]
Z_PATHS = {
    "base":             [ 0.25,  0.10,  0.00],
    "adverse":          [-1.00, -0.60, -0.20],
    "severely_adverse": [-2.00, -1.30, -0.50],
}

MODEL_SPECS = {
    "CORP_14": dict(n_buckets=14, rho=0.06, pd_min=0.0003, pd_max=0.20,
                    stay_best=0.92, stay_worst=0.70, decay=1.1, downgrade_bias=1.6),
    "SME_10":  dict(n_buckets=10, rho=0.09, pd_min=0.0015, pd_max=0.25,
                    stay_best=0.85, stay_worst=0.62, decay=0.9, downgrade_bias=1.4),
}

_N = NormalDist()
norm_cdf = np.vectorize(_N.cdf)
norm_ppf = np.vectorize(lambda p: _N.inv_cdf(min(max(p, 1e-12), 1 - 1e-12)))


def build_ttc_matrix(n_buckets, pd_min, pd_max, stay_best, stay_worst,
                     decay, downgrade_bias, noise=0.15, rng=None):
    """Through-the-cycle annual matrix. Buckets ordered best -> worst, last bucket = Default.

    pd_min / pd_max     : PD of the best / worst performing bucket (log-linear in between)
    stay_best / worst   : probability of staying in the bucket, best / worst bucket (linear in between)
    decay               : migration weight = exp(-decay * notches)
    downgrade_bias      : multiplier on downgrade weights vs upgrades
    noise               : lognormal noise on migration weights
    """
    rng = rng if rng is not None else np.random.default_rng(SEED)
    n_perf = n_buckets - 1
    pds = np.geomspace(pd_min, pd_max, n_perf)        # monotone PD curve
    stay = np.linspace(stay_best, stay_worst, n_perf)  # diagonal weakens for worse grades
    grades = np.arange(n_perf)

    M = np.zeros((n_buckets, n_buckets))
    for i in range(n_perf):
        dist = np.abs(grades - i)
        w = np.exp(-decay * dist) * np.where(grades > i, downgrade_bias, 1.0)
        w *= rng.lognormal(0.0, noise, n_perf)         # a bit of randomness in migrations
        w[i] = 0.0
        w /= w.sum()
        M[i, :n_perf] = (1.0 - stay[i] - pds[i]) * w
        M[i, i] = stay[i]
        M[i, -1] = pds[i]
    M[-1, -1] = 1.0                                    # default is absorbing
    return M


def condition_on_cycle(M, z, rho):
    """Vasicek one-factor Z-shift of a migration matrix.
    z < 0 -> downturn (more downgrades / defaults), z > 0 -> benign."""
    # C[i, j] = P(end in bucket j or worse | start i)
    cum_worse = np.cumsum(M[:, ::-1], axis=1)[:, ::-1]
    thresholds = norm_ppf(cum_worse)
    cond = norm_cdf((thresholds - np.sqrt(rho) * z) / np.sqrt(1.0 - rho))
    cond[:, 0] = 1.0
    cond[cum_worse == 0] = 0.0                         # keep impossible transitions impossible
    P = cond - np.append(cond[:, 1:], np.zeros((len(M), 1)), axis=1)
    P[-1] = 0.0
    P[-1, -1] = 1.0
    return P / P.sum(axis=1, keepdims=True)


@dataclass
class RatingModel:
    """`matrix(scenario, year)` returns the annual matrix for projection year `year`
    (1-based; any year >= 3 uses the year-3+ matrix). Bucket `n_buckets` is Default."""
    name: str
    n_buckets: int
    rho: float
    ttc: np.ndarray = field(repr=False)
    matrices: dict = field(default_factory=dict, repr=False)  # scenario -> [M_y1, M_y2, M_y3+]

    @property
    def performing_buckets(self):
        return list(range(1, self.n_buckets))

    def matrix(self, scenario=DEFAULT_SCENARIO, year=1):
        return self.matrices[scenario][min(max(year, 1), N_MATRIX_YEARS) - 1]

    def check(self):
        """Raise AssertionError if any matrix is inconsistent."""
        for y in range(1, N_MATRIX_YEARS + 1):
            for s in SCENARIOS:
                M = self.matrix(s, y)
                assert M.shape == (self.n_buckets, self.n_buckets)
                assert np.all(M >= 0) and np.allclose(M.sum(axis=1), 1.0), (self.name, s, y)
                assert M[-1, -1] == 1.0                    # absorbing default
                assert np.all(np.diff(M[:-1, -1]) > 0)     # PD increases with bucket
                assert np.all(M.argmax(axis=1) == np.arange(self.n_buckets))  # staying is the most likely outcome
            # scenario ordering: base < adverse < severely adverse
            pds = [self.matrix(s, y)[:-1, -1] for s in SCENARIOS]
            assert np.all(pds[0] < pds[1]) and np.all(pds[1] < pds[2])

    @classmethod
    def generate(cls, name, n_buckets, rho, z_paths=Z_PATHS, rng=None, **ttc_params):
        ttc = build_ttc_matrix(n_buckets, rng=rng, **ttc_params)
        matrices = {s: [condition_on_cycle(ttc, z, rho) for z in z_paths[s]] for s in SCENARIOS}
        return cls(name, n_buckets, rho, ttc, matrices)


def generate_models(specs=MODEL_SPECS, rng=None, check=True):
    """{name: RatingModel} built from `specs`; runs the consistency checks by default."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    models = {name: RatingModel.generate(name, rng=rng, **spec) for name, spec in specs.items()}
    if check:
        for m in models.values():
            m.check()
    return models


def to_matrix_table(models, n_classes=N_CLASSES):
    """Matrices in the SQL extract format: model, scenario, year, rank, 1..n_classes.
    One row per performing starting rank; unused class columns are 0."""
    rows = [[m.name, s, y, r + 1, *m.matrix(s, y)[r], *np.zeros(n_classes - m.n_buckets)]
            for m in models.values() for s in SCENARIOS for y in range(1, N_MATRIX_YEARS + 1)
            for r in range(m.n_buckets - 1)]
    return pd.DataFrame(rows, columns=["model", "scenario", "year", "rank", *map(str, range(1, n_classes + 1))])
