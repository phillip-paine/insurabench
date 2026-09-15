# insurabench

The go-to Python library for end-to-end non-life insurance statistical
modeling and risk estimation, aimed at data scientists rather than
actuaries.

This is a work in progress, built in phases (see `insurabench_build_order.md`
in the project). **This phase implements the data layer only:**
`PolicyFrame`, the policies/claims loaders, join validation, and the
frequency/severity reshape. Modeling, curves, evaluation, and geo modules
are not implemented yet.

## Quickstart

```python
from insurabench.data.datasets import make_synthetic_two_table
from insurabench.data.policy_frame import PolicyFrame

book = make_synthetic_two_table(n_policies=500)

pf = PolicyFrame.from_tables(
    book.policies, book.claims, book.policy_schema, book.claims_schema
)

freq_df = pf.frequency_view()   # one row per policy-period: claim_count, exposure, features
sev_df = pf.severity_view()     # one row per claim: claim_amount + features
print(pf.describe())
```

## Design principles this layer enforces

See `insurabench_design_brief.md` §3 for the full rationale. In short:

- **Two separate tables, explicitly linked, never pre-joined.** Real
  insurers store policies and claims separately; every tutorial that starts
  from a flat file skips the hard part.
- **Policy-period identity is explicit.** `policy_id` may recur across
  renewal terms. `insurabench.data.schema.PolicySchema` requires you to say
  so (`term_start_col`/`term_end_col`) rather than silently assuming
  uniqueness.
- **Fail loudly, never silently drop rows.** Orphan claims, claims dated
  outside their policy's term, ambiguous policy identity, non-positive
  exposure -- all raise a specific, informative exception
  (`insurabench.exceptions`) rather than being cleaned away.
- **Frequency and severity are different grains, exposed explicitly.**
  `PolicyFrame.frequency_view()` / `.severity_view()` are named, callable,
  inspectable steps -- never an implicit reshape inside a model's `.fit()`.

## Development

```bash
pip install -e ".[dev]"
pytest
```
