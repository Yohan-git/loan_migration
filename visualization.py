"""Styled pandas views of the migration matrices, the loan portfolio and the migration.

Matrix views take a `matrices.MigrationMatrices` and a model name.
Shading is plain CSS (no matplotlib needed). Explorers use ipywidgets and only work inside Jupyter.
"""
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

from matrices import DEFAULT_SCENARIO
from migration import AMORTIZATION_TYPES, project_portfolio

# RGB per cell type, opacity scales with the value
COLORS = {
    "stay":      (66, 133, 244),   # blue   - diagonal
    "upgrade":   (52, 168, 83),    # green  - below diagonal
    "downgrade": (234, 134, 0),    # orange - above diagonal
    "default":   (217, 48, 37),    # red    - default column
}
HEADER_CSS = [{"selector": "caption", "props": "caption-side: top; font-weight: bold; text-align: left; padding-bottom: 6px"},
              {"selector": "th", "props": "text-align: center"}]
MATURITY_LABELS = [f"{a}-{a + 1}y" for a in range(7)]


def _bg(rgb, intensity):
    """CSS background with opacity in [0.08, 0.8] driven by intensity in [0, 1]."""
    if pd.isna(intensity) or intensity <= 0:
        return ""
    alpha = 0.08 + 0.72 * min(float(intensity), 1.0)
    return f"background-color: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha:.2f})"


def heat(df, rgb=COLORS["stay"], exclude=("Total",)):
    """Shade a numeric table (Total row / column left plain)."""
    rows = [r for r in df.index if r not in exclude]
    cols = [c for c in df.columns if c not in exclude]
    vmax = np.nanmax(df.loc[rows, cols].to_numpy()) or 1
    return df.style.map(lambda v: _bg(rgb, v / vmax), subset=pd.IndexSlice[rows, cols]).set_table_styles(HEADER_CSS)


# ---------------------------------------------------------------- matrices

def style_matrix_table(table, rows=None):
    """The flat matrix table (SQL format), probabilities in %."""
    df = table if rows is None else table.head(rows)
    keys = [c for c in ("model", "scenario", "year", "rank") if c in df.columns]
    probs = [c for c in df.columns if c not in keys]
    return (df.style.format({c: lambda v: f"{v * 100:.2f}" if v else "" for c in probs})
            .set_caption("Matrix table - one row per model / scenario / year / starting rank (probabilities in %)")
            .set_table_styles(HEADER_CSS))


def style_pd_table(matrices, model, decimals=3):
    """1-year PD (%) by class for every scenario / year of application."""
    return (heat(matrices.pd_table(model) * 100, COLORS["default"]).format(f"{{:.{decimals}f}}")
            .set_caption(f"{model} - 1-year PD by class (%)"))


def style_matrix(matrices, model, scenario=DEFAULT_SCENARIO, year=1, decimals=2):
    """Annual migration matrix (%) as a heat map: diagonal blue, upgrades green, downgrades orange, default red."""
    label = matrices.year_labels(model, scenario)[min(year, matrices.n_years[model, scenario])]
    return _style_transition(matrices.annual_df(model, scenario, year), decimals,
                             f"{model} | {scenario} | {label} - annual migration probabilities (%)")


def style_quarterly_matrix(matrices, model, scenario=DEFAULT_SCENARIO, year=1, decimals=3):
    """Quarterly matrix (%) used during the quarters of projection year `year`."""
    label = matrices.year_labels(model, scenario)[min(year, matrices.n_years[model, scenario])]
    return _style_transition(matrices.quarterly_df(model, scenario, year), decimals,
                             f"{model} | {scenario} | quarters of {label} - quarterly migration probabilities (%)")


def _style_transition(matrix_df, decimals, caption):
    df = (matrix_df * 100).rename_axis(index="from", columns="to")
    M = df.to_numpy()
    n = len(df)
    off = M.copy()
    np.fill_diagonal(off, 0)
    off[:, -1] = 0
    off_max, pd_max = off.max() or 1, M[:-1, -1].max() or 1

    css = np.full(M.shape, "", dtype=object)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if v == 0:
                css[i, j] = "color: #9aa0a6"
            elif i == j:
                css[i, j] = _bg(COLORS["stay"], v / 100)
            elif j == n - 1:
                css[i, j] = _bg(COLORS["default"], np.sqrt(v / pd_max))
            else:   # sqrt so that small migrations stay visible
                css[i, j] = _bg(COLORS["upgrade" if j < i else "downgrade"], np.sqrt(v / off_max))

    return (df.style.format(f"{{:.{decimals}f}}")
            .apply(lambda _: pd.DataFrame(css, index=df.index, columns=df.columns), axis=None)
            .set_caption(caption)
            .set_table_styles(HEADER_CSS))


def style_matrix_diff(matrices, model, scenario=DEFAULT_SCENARIO, year=1, vs="base", decimals=2):
    """Difference vs another scenario in percentage points: red = more probability, blue = less."""
    df = ((matrices.annual_df(model, scenario, year) - matrices.annual_df(model, vs, year)) * 100
          ).rename_axis(index="from", columns="to")
    vmax = np.abs(df.to_numpy()).max() or 1
    css = lambda v: _bg(COLORS["default"] if v > 0 else COLORS["stay"], np.sqrt(abs(v) / vmax))
    return (df.style.format(f"{{:+.{decimals}f}}").map(css)
            .set_caption(f"{model} | year {year} - {scenario} minus {vs} (pp)")
            .set_table_styles(HEADER_CSS))


def matrix_explorer(matrices):
    """Dropdowns to browse every annual / quarterly matrix, or its difference vs the base scenario."""
    scenarios = matrices.scenarios()

    def show(model, scenario, year, view):
        if view == "Annual":
            display(style_matrix(matrices, model, scenario, year))
        elif view == "Quarterly":
            display(style_quarterly_matrix(matrices, model, scenario, year))
        else:
            display(style_matrix_diff(matrices, model, scenario, year, vs=scenarios[0]))

    widgets.interact(
        show,
        model=widgets.Dropdown(options=matrices.models, description="Model"),
        scenario=widgets.Dropdown(options=scenarios, value=DEFAULT_SCENARIO if DEFAULT_SCENARIO in scenarios else scenarios[0],
                                  description="Scenario"),
        year=widgets.Dropdown(options=list(range(1, max(matrices.n_years.values()) + 1)), description="Year"),
        view=widgets.ToggleButtons(options=["Annual", "Quarterly", f"Diff vs {scenarios[0]}"], description="View"),
    )


# ---------------------------------------------------------------- portfolio

def style_portfolio(df):
    """Loan table with formatted numbers and in-cell bars for amount and PD."""
    fmt = {"amount": "{:,.0f}", "maturity_years": "{:.2f}", "pd_ttc": "{:.3%}",
           "maturity_date": lambda d: d.strftime("%Y-%m-%d")}
    return (df.style.format({k: v for k, v in fmt.items() if k in df.columns})
            .bar(subset=[c for c in ("amount", "pd_ttc") if c in df.columns], color="#9ec5fe", vmin=0)
            .set_table_styles(HEADER_CSS))


def style_model_summary(summary):
    """Formats the output of `portfolio.model_summary`."""
    return (summary.style.format({"exposure": "{:,.0f}", "avg_amount": "{:,.0f}",
                                  "avg_maturity_y": "{:.2f}", "avg_pd_ttc": "{:.3%}"})
            .set_table_styles(HEADER_CSS))


def portfolio_summary(portfolio):
    """Loans, exposure, share (and TTC PD if available) per model and rating."""
    aggs = dict(n_loans=("amount", "size"), exposure_m=("amount", "sum"), avg_maturity_y=("maturity_years", "mean"))
    if "pd_ttc" in portfolio.columns:
        aggs["pd_ttc"] = ("pd_ttc", "first")
    g = portfolio.groupby(["rating_model", "rating"]).agg(**aggs)
    g["exposure_m"] /= 1e6
    g["share"] = g["exposure_m"] / g.groupby(level=0)["exposure_m"].transform("sum")
    fmt = {"exposure_m": "{:,.1f}", "avg_maturity_y": "{:.2f}", "share": "{:.1%}", "pd_ttc": "{:.3%}"}
    return (g.style.format({k: v for k, v in fmt.items() if k in g.columns})
            .bar(subset=["exposure_m", "share"], color="#9ec5fe", vmin=0)
            .bar(subset=[c for c in ["pd_ttc"] if c in g.columns], color="#f4a7a1", vmin=0)
            .set_table_styles(HEADER_CSS))


def portfolio_pivot(portfolio, matrices, model, values="amount", aggfunc="sum", scale=1e6):
    """Rating x residual maturity bucket table for one model, with totals."""
    sub = portfolio[portfolio.rating_model == model]
    mat = pd.cut(sub.maturity_years, bins=range(0, 8), right=False, labels=MATURITY_LABELS)
    t = sub.pivot_table(index="rating", columns=mat, values=values, aggfunc=aggfunc, fill_value=0, observed=False)
    t = t.reindex(matrices.performing_buckets(model), fill_value=0) / scale
    t.columns = t.columns.astype(str)
    t.columns.name = "residual_maturity"
    t["Total"] = t.sum(axis=1)
    t.loc["Total"] = t.sum()
    return t


def style_exposure_grid(portfolio, matrices, model):
    """Exposure (millions) by rating x residual maturity."""
    return (heat(portfolio_pivot(portfolio, matrices, model)).format("{:,.1f}")
            .set_caption(f"{model} - exposure (millions) by rating x residual maturity"))


def style_count_grid(portfolio, matrices, model):
    """Number of loans by rating x residual maturity."""
    return (heat(portfolio_pivot(portfolio, matrices, model, aggfunc="count", scale=1), COLORS["upgrade"]).format("{:,.0f}")
            .set_caption(f"{model} - number of loans by rating x residual maturity"))


def portfolio_explorer(portfolio, matrices):
    """Filter by model / rating / maturity, sort, and show the matching loans."""
    max_rating = max(matrices.n_buckets.values()) - 1

    def show(model, rating, maturity, sort_by, descending, rows):
        df = portfolio if model == "All" else portfolio[portfolio.rating_model == model]
        df = df[df.rating.between(*rating) & df.maturity_years.between(*maturity)]
        msg = f"{len(df)} loans | exposure {df.amount.sum() / 1e6:,.1f}m"
        if "pd_ttc" in df.columns and len(df):
            msg += f" | exposure-weighted PD TTC {np.average(df.pd_ttc, weights=df.amount):.3%}"
        print(msg)
        display(style_portfolio(df.sort_values(sort_by, ascending=not descending).head(rows)))

    widgets.interact(
        show,
        model=widgets.Dropdown(options=["All"] + matrices.models, description="Model"),
        rating=widgets.IntRangeSlider(value=(1, max_rating), min=1, max=max_rating, description="Rating"),
        maturity=widgets.FloatRangeSlider(value=(0, 7), min=0, max=7, step=0.5, description="Maturity (y)"),
        sort_by=widgets.Dropdown(options=[c for c in ("amount", "maturity_years", "rating", "pd_ttc") if c in portfolio.columns],
                                 description="Sort by"),
        descending=widgets.Checkbox(value=True, description="Descending"),
        rows=widgets.IntSlider(value=20, min=5, max=200, step=5, description="Rows"),
    )


# ---------------------------------------------------------------- migration

def style_initial_vectors(vectors, scale=1e6):
    """Starting amount vectors (millions), one column per rating model, with totals."""
    labels = max((v.index for v in vectors.values()), key=len)   # longest scale; "D" aligns across models
    df = pd.concat(vectors, axis=1).reindex(labels) / scale
    df.loc["Total"] = df.sum()
    return (heat(df).format("{:,.1f}", na_rep="")
            .set_caption("Starting vectors - amount per class (millions)"))


def style_vectors(proj, scale=1e6, quarters=None):
    """Vectors migrating quarter by quarter (millions): classes as rows, quarters as columns.
    Performing classes in blue, Default in red; repaid amounts and the reconciliation below."""
    v = proj.vectors if quarters is None else proj.vectors.loc[list(quarters)]
    df = v.T / scale
    cum_repaid = pd.concat([pd.Series([0.0], index=[0]), proj.flows["cumulative_repaid"]])
    df.loc["Repaid (cum.)"] = cum_repaid.reindex(v.index).to_numpy() / scale
    df.loc["Total"] = df.sum()
    df.columns = [f"Q{q}" for q in v.index]

    perf = list(proj.vectors.columns[:-1])
    pmax = df.loc[perf].to_numpy().max() or 1
    dmax = df.loc["D"].max() or 1
    return (df.style.format("{:,.1f}")
            .map(lambda x: _bg(COLORS["stay"], x / pmax), subset=pd.IndexSlice[perf, :])
            .map(lambda x: _bg(COLORS["default"], np.sqrt(x / dmax)), subset=pd.IndexSlice[["D"], :])
            .map(lambda x: _bg(COLORS["upgrade"], 0.3) if x else "", subset=pd.IndexSlice[["Repaid (cum.)"], :])
            .set_properties(subset=pd.IndexSlice[["Total"], :], **{"font-weight": "bold"})
            .set_caption(f"{proj.model} | {proj.scenario} | {proj.amortization} - "
                         f"amount per class at quarter end (millions)")
            .set_table_styles(HEADER_CSS))


def style_flows(proj, scale=1e6):
    """Quarterly flows: opening, new defaults, repayments, closing, default rates, reconciliation."""
    df = proj.flows.copy()
    amounts = ["opening_performing", "new_default", "repayment", "closing_performing",
               "cumulative_default", "cumulative_repaid", "check"]
    df[amounts] = df[amounts] / scale
    if "date" in df.columns:
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    fmt = {c: "{:,.2f}" for c in amounts}
    fmt.update({"quarterly_default_rate": "{:.3%}", "cumulative_default_rate": "{:.2%}", "check": "{:,.6f}"})
    return (df.style.format(fmt, na_rep="")
            .bar(subset=["new_default"], color="#f4a7a1", vmin=0)
            .bar(subset=["repayment"], color="#a8dab5", vmin=0)
            .bar(subset=["cumulative_default_rate"], color="#f4a7a1", vmin=0)
            .set_caption(f"{proj.model} | {proj.scenario} | {proj.amortization} - quarterly flows "
                         f"(amounts in millions, check = closing + cum. default + cum. repaid - initial)")
            .set_table_styles(HEADER_CSS))


def style_scenario_comparison(table, caption="Cumulative default rate (% of initial amount)"):
    """quarter x (model, scenario) table from `migration.scenario_comparison`."""
    df = table * 100
    df.index = [f"Q{q}" for q in df.index]
    return heat(df, COLORS["default"]).format("{:.2f}").set_caption(caption)


def migration_explorer(portfolio, matrices, reference_date=None):
    """Pick model / scenario / amortisation and view the migrating vectors or the flows."""
    scenarios = matrices.scenarios()

    def show(model, scenario, amortization, view):
        proj = project_portfolio(portfolio, matrices, scenario, amortization, reference_date)[model]
        display(style_vectors(proj) if view == "Vectors" else style_flows(proj))

    widgets.interact(
        show,
        model=widgets.Dropdown(options=[m for m in matrices.models if m in set(portfolio.rating_model)], description="Model"),
        scenario=widgets.Dropdown(options=scenarios, value=DEFAULT_SCENARIO if DEFAULT_SCENARIO in scenarios else scenarios[0],
                                  description="Scenario"),
        amortization=widgets.Dropdown(options=AMORTIZATION_TYPES, description="Amortisation"),
        view=widgets.ToggleButtons(options=["Vectors", "Flows"], description="View"),
    )
