"""Migration matrices used by the projection, loaded from a flat table (e.g. a SQL extract).

Expected table, one row per (model, scenario, year, starting rank):

    model | scenario | year | rank | 1 | 2 | ... | 14

* year     : year of application (1, 2, 3). The last year available applies to every later year (3 = "3+").
* rank     : starting class the row's probabilities apply to (1-based).
* 1 .. 14  : probability of ending in each class. A model with fewer classes leaves the extra columns at 0 / NULL;
             its last class holding a probability is its Default class.
* Default rows are optional (default is made absorbing). Probabilities can be in [0, 1] or in % (rows summing to 100).

    matrices = MigrationMatrices.from_frame(df)
    matrices.annual("CORP_14", "adverse", year=1)       # annual matrix (numpy)
    matrices.quarterly("CORP_14", "adverse", quarter=5) # quarterly matrix applied in projection quarter 5
"""
import numpy as np
import pandas as pd

DEFAULT_SCENARIO = "adverse"
QUARTERS_PER_YEAR = 4
ROW_SUM_TOL = 1e-4  # rows summing to 1 +/- tol are renormalised, otherwise rejected


def year_label(year, last_year):
    return f"Y{year}+" if year >= last_year else f"Y{year}"


def matrix_root(M, k=QUARTERS_PER_YEAR):
    """k-th root of an annual matrix (transition matrix over 1/k year).

    Principal root via eigendecomposition. If the root is not a valid transition matrix,
    negative entries are set to 0 and taken off the diagonal (diagonal adjustment)."""
    w, V = np.linalg.eig(M)
    R = (V @ np.diag(w.astype(complex) ** (1.0 / k)) @ np.linalg.inv(V)).real
    neg = np.minimum(R, 0.0)
    R = np.maximum(R, 0.0)
    R[np.diag_indices_from(R)] += neg.sum(axis=1)
    R[-1] = 0.0
    R[-1, -1] = 1.0                                    # default stays absorbing
    return R / R.sum(axis=1, keepdims=True)


class MigrationMatrices:
    """Annual matrices per (model, scenario, year) and the quarterly matrices derived from them."""

    def __init__(self, annual):
        self._annual = annual                          # {(model, scenario, year): n x n array}
        self._quarterly = {}
        self.n_buckets, self.n_years = {}, {}
        for (m, s, y), M in annual.items():
            if self.n_buckets.setdefault(m, len(M)) != len(M):
                raise ValueError(f"{m}: number of classes differs across scenarios / years")
            self.n_years[m, s] = max(self.n_years.get((m, s), 0), y)
        self.check()

    @classmethod
    def from_frame(cls, df, model="model", scenario="scenario", year="year", rank="rank", class_cols=None):
        """Build from the flat table. If your column names differ, pass them, e.g.
        `from_frame(df, model="MODEL_NAME", year="APPLICATION_YEAR")`.
        `class_cols` defaults to every other column, in table order."""
        df = df.reset_index(drop=True)
        keys = [model, scenario, year, rank]
        class_cols = list(class_cols) if class_cols is not None else [c for c in df.columns if c not in keys]
        P = df[class_cols].apply(pd.to_numeric).fillna(0.0).to_numpy(float)
        if P.sum(axis=1).mean() > 1.5:                 # probabilities given in %
            P = P / 100.0

        annual = {}
        for (m, s, y), grp in df.groupby([model, scenario, year], sort=False):   # keeps table order
            idx = grp.index
            ranks = grp[rank].astype(int).to_numpy()
            rows = P[idx]
            where = f"{m} / {s} / year {y}"
            if len(set(ranks)) != len(ranks):
                raise ValueError(f"{where}: duplicate ranks")
            n = int(np.nonzero(rows.any(axis=0))[0].max()) + 1   # last class with probability = Default
            if ranks.max() > n:
                raise ValueError(f"{where}: rank {ranks.max()} beyond the default class {n}")
            missing = sorted(set(range(1, n)) - set(ranks))
            if missing:
                raise ValueError(f"{where}: missing ranks {missing}")
            M = np.zeros((n, n))
            M[-1, -1] = 1.0
            for r, row in zip(ranks, rows):
                M[r - 1] = row[:n]
            M[-1] = 0.0
            M[-1, -1] = 1.0                            # default absorbing whether or not a default row is given
            sums = M.sum(axis=1, keepdims=True)
            bad = (np.flatnonzero(np.abs(sums[:, 0] - 1) > ROW_SUM_TOL) + 1).tolist()
            if bad:
                raise ValueError(f"{where}: ranks {bad} do not sum to 1 (sums: {sums[np.array(bad) - 1, 0].round(4).tolist()})")
            annual[str(m), str(s), int(y)] = M / sums
        return cls(annual)

    # ------------------------------------------------------------ access

    @property
    def models(self):
        return list(self.n_buckets)

    def scenarios(self, model=None):
        return list(dict.fromkeys(s for (m, s, _) in self._annual if model is None or m == model))

    def performing_buckets(self, model):
        return list(range(1, self.n_buckets[model]))

    def labels(self, model):
        return [str(b) for b in self.performing_buckets(model)] + ["D"]

    def year_labels(self, model, scenario):
        last = self.n_years[model, scenario]
        return {y: year_label(y, last) for y in range(1, last + 1)}

    def annual(self, model, scenario=DEFAULT_SCENARIO, year=1):
        """Annual matrix for projection year `year` (years beyond the last available use the last one)."""
        y = min(max(year, 1), self.n_years[model, scenario])
        return self._annual[model, scenario, y]

    def quarterly(self, model, scenario=DEFAULT_SCENARIO, quarter=1):
        """Quarterly matrix for projection quarter `quarter` (1-based): 4th root of that year's annual matrix."""
        y = min((max(quarter, 1) - 1) // QUARTERS_PER_YEAR + 1, self.n_years[model, scenario])
        if (model, scenario, y) not in self._quarterly:
            self._quarterly[model, scenario, y] = matrix_root(self._annual[model, scenario, y])
        return self._quarterly[model, scenario, y]

    def annual_df(self, model, scenario=DEFAULT_SCENARIO, year=1):
        labels = self.labels(model)
        return pd.DataFrame(self.annual(model, scenario, year), index=labels, columns=labels)

    def quarterly_df(self, model, scenario=DEFAULT_SCENARIO, year=1):
        """Quarterly matrix used during the quarters of projection year `year`."""
        labels = self.labels(model)
        q = (year - 1) * QUARTERS_PER_YEAR + 1
        return pd.DataFrame(self.quarterly(model, scenario, q), index=labels, columns=labels)

    # ------------------------------------------------------------ summaries / checks

    def summary(self):
        return pd.DataFrame([{"model": m, "n_classes": self.n_buckets[m], "scenarios": ", ".join(self.scenarios(m)),
                              "years": ", ".join(self.year_labels(m, self.scenarios(m)[0]).values())}
                             for m in self.models]).set_index("model")

    def __repr__(self):
        return f"MigrationMatrices({len(self._annual)} annual matrices)\n{self.summary().to_string()}"

    def pd_table(self, model):
        """1-year PD by performing class, per scenario and year of application."""
        return pd.concat({(s, lbl): pd.Series(self.annual(model, s, y)[:-1, -1], index=self.performing_buckets(model))
                          for s in self.scenarios(model)
                          for y, lbl in self.year_labels(model, s).items()}, axis=1)

    def root_errors(self):
        """max |Q^4 - annual matrix| (pp) per model, scenario and year."""
        out = {}
        for (m, s, y), M in self._annual.items():
            Q = self.quarterly(m, s, (y - 1) * QUARTERS_PER_YEAR + 1)
            out[(m, s), year_label(y, self.n_years[m, s])] = np.abs(np.linalg.matrix_power(Q, QUARTERS_PER_YEAR) - M).max() * 100
        return pd.Series(out).unstack(0)

    def check(self):
        """Every matrix (annual and quarterly) is a valid transition matrix with an absorbing default."""
        for (m, s, y), M in self._annual.items():
            Q = self.quarterly(m, s, (y - 1) * QUARTERS_PER_YEAR + 1)
            for name, A in (("annual", M), ("quarterly", Q)):
                assert np.all(A >= 0) and np.allclose(A.sum(axis=1), 1.0), (m, s, y, name)
                assert A[-1, -1] == 1.0, (m, s, y, name, "default not absorbing")

    def to_frame(self, n_classes=14):
        """Back to the flat table format (default rows included)."""
        rows = [[m, s, y, r + 1, *M[r], *np.zeros(n_classes - len(M))]
                for (m, s, y), M in self._annual.items() for r in range(len(M))]
        return pd.DataFrame(rows, columns=["model", "scenario", "year", "rank", *map(str, range(1, n_classes + 1))])
