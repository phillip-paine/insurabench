# Sample data

Two small, fixed CSVs (seeded, so they never change) used by
`01_getting_started.ipynb`. Small enough to open directly in Excel or a
text editor if you just want to look at the shape of the data before
touching any code.

Generated with `insurabench.data.datasets.make_synthetic_two_table(
n_policies=200, renewals=True, seed=42)` -- see that function's docstring
if you want to generate a fresh (larger, or differently-seeded) book
yourself instead of using the fixed CSVs here.

## `sample_policies.csv` -- 414 rows, one row per policy-period

| Column          | Meaning                                                         |
|-----------------|------------------------------------------------------------------|
| `policy_id`     | Contract identifier. **Recurs** across renewal terms -- 200 distinct contracts, 414 rows, because a contract can renew 1-3 times. This is why `term_start` exists: `policy_id` alone is not a unique key here. |
| `term_start`    | Start date of this policy-period.                                |
| `term_end`      | End date of this policy-period.                                  |
| `exposure`      | Fraction of a year this period covers (between 0 and 1).          |
| `vehicle_power` | Rating factor (continuous).                                      |
| `driver_age`    | Rating factor (continuous).                                      |
| `vehicle_brand` | Rating factor (categorical): Alpha / Beta / Gamma / Delta.        |
| `region`        | Rating factor (categorical): North / South / East / West.         |
| `premium`       | Premium charged for this period.                                  |

## `sample_claims.csv` -- 89 rows, one row per claim

| Column         | Meaning                                                          |
|----------------|-------------------------------------------------------------------|
| `claim_id`     | Unique claim identifier.                                          |
| `policy_id`    | Which contract this claim belongs to -- **not** enough on its own to know which policy-period; insurabench matches a claim to the one period whose date window contains `claim_date`. |
| `claim_date`   | Date of loss.                                                      |
| `claim_amount` | Amount, treated as final (no reserving/development -- see the main package docs). |
| `peril`        | collision / theft / weather / liability.                          |
| `coverage`     | Collision / Comprehensive / Liability.                            |

## Why two files, not one?

Because that's how real insurers store this data -- policies and claims
are separate systems. `01_getting_started.ipynb` shows how to load both
and get them joined correctly.
