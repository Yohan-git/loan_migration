"""Synthetic loan portfolio linked to rating models.

Each loan: id, rating model, current rating bucket (performing buckets only), amount, maturity.
The migration needs the columns rating_model, rating, amount, maturity_years (see `migration.REQUIRED_COLUMNS`).
"""
import numpy as np
import pandas as pd

from matrices import DEFAULT_SCENARIO
from rating_models import SEED

REFERENCE_DATE = pd.Timestamp("2026-10-23")
N_LOANS = 1000
MAX_MATURITY_Y = 7
MODEL_MIX = {"CORP_14": 0.6, "SME_10": 0.4}
AMOUNT_PARAMS = {"CORP_14": (5e6, 1.0), "SME_10": (5e5, 0.8)}  # lognormal median, sigma


def draw_ratings(model, size, rng):
    """Bell-shaped rating distribution centred slightly below mid-scale."""
    grades = np.array(model.performing_buckets)
    centre, spread = 0.55 * len(grades), 0.22 * len(grades)
    w = np.exp(-0.5 * ((grades - centre) / spread) ** 2)
    return rng.choice(grades, size=size, p=w / w.sum())


def generate_portfolio(models, n_loans=N_LOANS, model_mix=MODEL_MIX, amount_params=AMOUNT_PARAMS,
                       reference_date=REFERENCE_DATE, max_maturity=MAX_MATURITY_Y, rng=None):
    """Loan-level DataFrame indexed by loan_id."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    model_col = rng.choice(list(model_mix), size=n_loans, p=list(model_mix.values()))
    ratings = np.empty(n_loans, dtype=int)
    amounts = np.empty(n_loans)
    for name, m in models.items():
        mask = model_col == name
        ratings[mask] = draw_ratings(m, mask.sum(), rng)
        median, sigma = amount_params[name]
        amounts[mask] = np.round(rng.lognormal(np.log(median), sigma, mask.sum()), -3)

    maturity_years = rng.uniform(0, max_maturity, n_loans)

    portfolio = pd.DataFrame({
        "loan_id": [f"L{i:05d}" for i in range(1, n_loans + 1)],
        "rating_model": model_col,
        "rating": ratings,
        "amount": amounts,
        "maturity_years": maturity_years.round(4),
        "maturity_date": (reference_date + pd.to_timedelta(maturity_years * 365.25, unit="D")).normalize(),
    })
    portfolio["n_buckets"] = portfolio["rating_model"].map({k: m.n_buckets for k, m in models.items()})
    portfolio["pd_ttc"] = [models[mo].ttc[r - 1, -1] for mo, r in zip(portfolio["rating_model"], portfolio["rating"])]
    return portfolio.set_index("loan_id")


def loan_matrix(portfolio, matrices, loan_id, scenario=DEFAULT_SCENARIO, year=1):
    """Annual migration matrix associated with a loan for a given scenario / projection year."""
    return matrices.annual(portfolio.at[loan_id, "rating_model"], scenario, year)


def model_summary(portfolio):
    """Loans, exposure, average amount / maturity / TTC PD per rating model."""
    return portfolio.groupby("rating_model").agg(
        n_loans=("amount", "size"),
        exposure=("amount", "sum"),
        avg_amount=("amount", "mean"),
        avg_maturity_y=("maturity_years", "mean"),
        avg_pd_ttc=("pd_ttc", "mean"),
    )
