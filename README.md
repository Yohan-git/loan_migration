# Loan rating migration framework

Deterministic projection of a loan portfolio through rating migration matrices, on a quarterly basis, with amortisation.
All data in this repository is **synthetic**.

## What it does

1. **Matrices**: loads annual migration matrices from a flat table (one row per model / scenario / year / starting rank,
   then the probabilities of the 14 classes, as a SQL extract would provide them). Models can have fewer than 14 classes;
   the last class holding a probability is Default (absorbing).
2. **Quarterly matrices**: each annual matrix is turned into a quarterly one (4th root, `Q^4 = annual`).
   Quarters 1-4 use year 1, 5-8 year 2, 9+ year 3+.
3. **Migration**: the portfolio is aggregated into one amount vector per rating model and migrated quarter by quarter.
   Loans are repaid at maturity (bullet) or on a straight line (linear); defaulted amounts stay in Default.
   Vectors are split by maturity cohort so repayments remove exactly the loans that mature.
4. **Report**: an Excel workbook with the matrices used and the portfolio vector per quarter.

## Files

| File | Content |
|---|---|
| `migration_framework.ipynb` | Walk-through: matrices, portfolio, migration, Excel report |
| `matrices.py` | `MigrationMatrices`: reads the matrix table, gives annual and quarterly matrices, validates them |
| `migration.py` | Quarterly migration of the portfolio vectors with amortisation |
| `visualization.py` | Styled pandas tables and Jupyter explorers |
| `excel_export.py` | Excel report (matrices of the scenario used + portfolio vector per quarter) |
| `rating_models.py` | Synthetic matrices (TTC matrix + one-factor Vasicek stress per scenario / year) |
| `portfolio.py` | Synthetic loan portfolio |
| `output/migration_report.xlsx` | Example report (adverse scenario) |

## Usage

```python
from matrices import MigrationMatrices
from migration import project_portfolio
from excel_export import export_workbook

matrices = MigrationMatrices.from_frame(matrix_table)   # columns: model, scenario, year, rank, 1..14
projections = project_portfolio(portfolio, matrices, "adverse", "bullet")
projections["CORP_14"].vectors                           # amount per class at the end of each quarter

export_workbook("migration_report.xlsx", matrices, portfolio, "adverse")
```

The portfolio needs the columns `rating_model`, `rating`, `amount` and `maturity_years`.

Requirements: Python 3.10+, `numpy`, `pandas`, `openpyxl`, `ipywidgets` (notebook explorers only).
