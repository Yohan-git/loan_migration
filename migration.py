"""Deterministic quarterly migration of the portfolio, one amount vector per rating model.

Inputs: a loan-level portfolio (REQUIRED_COLUMNS) and a `matrices.MigrationMatrices`.

Each rating model's portfolio is an amount vector over its classes (last class = Default).
Every quarter:
  1. the vector migrates with the quarterly matrix of that projection year (Q^4 = annual matrix),
  2. performing amounts amortise according to the loans' maturities. Default is absorbing and not amortised.

To amortise exactly, each model vector is split into maturity cohorts (loans maturing in the same quarter);
cohorts migrate independently and repay on their own schedule. The model vector is the sum of its cohorts.

    projections = project_portfolio(portfolio, matrices, "adverse")
    projections["CORP_14"].vectors   # quarter x class amounts
    projections["CORP_14"].flows     # quarter x flows (defaults, repayments, ...)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from matrices import DEFAULT_SCENARIO, QUARTERS_PER_YEAR

AMORTIZATION_TYPES = ("bullet", "linear")
REQUIRED_COLUMNS = ["rating_model", "rating", "amount", "maturity_years"]


def validate_portfolio(portfolio, matrices):
    """Raise ValueError if the portfolio cannot be migrated with these matrices."""
    missing = [c for c in REQUIRED_COLUMNS if c not in portfolio.columns]
    if missing:
        raise ValueError(f"portfolio is missing columns {missing}")
    unknown = sorted(set(portfolio.rating_model) - set(matrices.models))
    if unknown:
        raise ValueError(f"no matrices for rating models {unknown}")
    max_rating = portfolio.rating_model.map(lambda m: matrices.n_buckets[m] - 1)
    bad = portfolio[(portfolio.rating < 1) | (portfolio.rating > max_rating)]
    if len(bad):
        raise ValueError(f"{len(bad)} loans have a rating outside their model's performing classes, e.g.\n{bad.head()}")


def portfolio_models(portfolio, matrices):
    """Rating models present in the portfolio, in the matrices' order."""
    present = set(portfolio.rating_model)
    return [m for m in matrices.models if m in present]


def maturity_quarter(maturity_years):
    """Projection quarter (1-based) in which a loan matures; repayment happens at the end of that quarter."""
    return np.maximum(np.ceil(np.asarray(maturity_years) * QUARTERS_PER_YEAR).astype(int), 1)


def initial_vectors(portfolio, matrices):
    """{model: amount per class (1..n-1, D)}; each total equals the amount rated by that model."""
    out = {}
    for m in portfolio_models(portfolio, matrices):
        v = portfolio[portfolio.rating_model == m].groupby("rating")["amount"].sum()
        out[m] = pd.Series(v.reindex(range(1, matrices.n_buckets[m] + 1), fill_value=0.0).to_numpy(),
                           index=matrices.labels(m), name=m)
    return out


def _cohorts(portfolio, matrices, model):
    """Maturity quarter of each cohort and its amount vector (cohorts x classes)."""
    sub = portfolio[portfolio.rating_model == model]
    q = maturity_quarter(sub.maturity_years.to_numpy())
    cohort_q = np.unique(q)
    V = np.zeros((len(cohort_q), matrices.n_buckets[model]))
    np.add.at(V, (np.searchsorted(cohort_q, q), sub.rating.to_numpy() - 1), sub.amount.to_numpy())
    return cohort_q, V


def amortization_factor(cohort_q, t, amortization="bullet"):
    """Share of each cohort's performing amount still outstanding after quarter t (vs after quarter t-1).

    bullet : full repayment at the end of the maturity quarter
    linear : straight-line repayment from the reference date down to 0 at the maturity quarter
    """
    if amortization == "bullet":
        return (t < cohort_q).astype(float)
    if amortization == "linear":
        before = np.clip(1 - (t - 1) / cohort_q, 0, None)
        after = np.clip(1 - t / cohort_q, 0, None)
        return np.divide(after, before, out=np.zeros_like(after), where=before > 0)
    raise ValueError(f"amortization must be one of {AMORTIZATION_TYPES}")


@dataclass
class Projection:
    model: str
    scenario: str
    amortization: str
    vectors: pd.DataFrame   # quarter x class amounts (end of quarter)
    flows: pd.DataFrame     # quarter x [opening, new default, repayment, closing, cumulative ...]


def project(portfolio, matrices, model, scenario=DEFAULT_SCENARIO, amortization="bullet",
            n_quarters=None, reference_date=None):
    """Migrate one rating model's vector quarter by quarter."""
    cohort_q, V = _cohorts(portfolio, matrices, model)
    n_quarters = n_quarters or int(cohort_q.max())

    vectors, flows = [V.sum(axis=0)], []
    cum_default, cum_repaid = V[:, -1].sum(), 0.0
    for t in range(1, n_quarters + 1):
        opening = V.sum(axis=0)
        V = V @ matrices.quarterly(model, scenario, t)
        new_default = V[:, -1].sum() - opening[-1]
        f = amortization_factor(cohort_q, t, amortization)
        repaid = (V[:, :-1].sum(axis=1) * (1 - f)).sum()
        V[:, :-1] *= f[:, None]
        cum_default += new_default
        cum_repaid += repaid
        vectors.append(V.sum(axis=0))
        flows.append({
            "quarter": t,
            "opening_performing": opening[:-1].sum(),
            "new_default": new_default,
            "repayment": repaid,
            "closing_performing": V[:, :-1].sum(),
            "cumulative_default": cum_default,
            "cumulative_repaid": cum_repaid,
        })

    vectors = pd.DataFrame(vectors, index=pd.Index(range(n_quarters + 1), name="quarter"), columns=matrices.labels(model))
    vectors.columns.name = "class"
    flows = pd.DataFrame(flows).set_index("quarter")
    if reference_date is not None:
        flows.insert(0, "date", [pd.Timestamp(reference_date) + pd.DateOffset(months=3 * t) for t in flows.index])
    initial = vectors.iloc[0].sum()
    flows["quarterly_default_rate"] = flows["new_default"] / flows["opening_performing"].where(flows["opening_performing"] > 0)
    flows["cumulative_default_rate"] = flows["cumulative_default"] / initial
    flows["check"] = flows["closing_performing"] + flows["cumulative_default"] + flows["cumulative_repaid"] - initial
    return Projection(model, scenario, amortization, vectors, flows)


def project_portfolio(portfolio, matrices, scenario=DEFAULT_SCENARIO, amortization="bullet", reference_date=None):
    """{model: Projection} for every rating model in the portfolio, all on the same horizon (longest maturity)."""
    validate_portfolio(portfolio, matrices)
    n_quarters = int(maturity_quarter(portfolio.maturity_years).max())
    return {m: project(portfolio, matrices, m, scenario, amortization, n_quarters, reference_date)
            for m in portfolio_models(portfolio, matrices)}


def scenario_comparison(portfolio, matrices, amortization="bullet", column="cumulative_default_rate"):
    """quarter x (model, scenario) table of one flows column."""
    return pd.concat({(m, s): p.flows[column]
                      for s in matrices.scenarios()
                      for m, p in project_portfolio(portfolio, matrices, s, amortization).items()},
                     axis=1).sort_index(axis=1, level=0, sort_remaining=False)
