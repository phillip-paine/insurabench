# insurabench

The go-to Python library for end-to-end non-life insurance statistical
modeling and risk estimation, aimed at data scientists rather than
actuaries.

This is a work in progress, built in phases (see `insurabench_build_order.md`
in the project). **Implemented so far:** the data layer (`PolicyFrame` and
friends) and the GLM wrapper. Curves, evaluation, the GBM wrapper, geo, and
reporting modules are not implemented yet.

## Quickstart: data layer

```python
from insurabench.data.datasets import make_synthetic_two_table
from insurabench.data.policy_frame import PolicyFrame

book = make_synthetic_two_table(n_policies=500)

pf = PolicyFrame.from_tables(
    book.policies, book.claims, book.policy_schema, book.claims_schema
)

freq_df = pf.frequency_view()       # one row per policy-period: claim_count, exposure, features
sev_df = pf.severity_view()         # one row per claim: claim_amount + features
pp_df = pf.pure_premium_view()      # one row per policy-period: total claim amount, exposure, features
print(pf.describe())

# peril-/coverage-specific views (requires peril_col/coverage_col on the schema)
theft_freq = pf.frequency_view(peril="theft")
```

## Quickstart: GLM (Tweedie pure-premium example)

```python
from insurabench.data.datasets import make_synthetic_two_table
from insurabench.data.policy_frame import PolicyFrame
from insurabench.model_selection import train_test_split_policy_frame
from insurabench.models.glm import GLMPricingModel

book = make_synthetic_two_table(n_policies=5000, renewals=True, seed=0)
pf = PolicyFrame.from_tables(book.policies, book.claims, book.policy_schema, book.claims_schema)

train, test = train_test_split_policy_frame(pf, test_size=0.25, seed=0)

model = GLMPricingModel(family="tweedie", power=1.5, alpha=0.01)
model.fit_policy_frame(train, target="pure_premium")

print(model.relativities())               # fitted coefficients + multiplicative relativities
print(model.score_policy_frame(test))     # deviance-based D^2 on held-out data
preds = model.predict_policy_frame(test)  # predicted pure premium per policy-period
```

The same `GLMPricingModel` also fits `target="frequency"` (Poisson/NB,
`claim_count` with `offset=log(exposure)`) and `target="severity"`
(Gamma, per-claim `claim_amount`) if you'd rather compose two separate
models instead of one direct Tweedie fit -- see design brief §4.

## Design principles this layer enforces

See `insurabench_design_brief.md` §3-4 for the full rationale. In short:

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
- **Frequency, severity, and pure premium are different grains, exposed
  explicitly.** `PolicyFrame.frequency_view()` / `.severity_view()` /
  `.pure_premium_view()` are named, callable, inspectable steps -- never an
  implicit reshape inside a model's `.fit()`.
- **Splitting for train/test never leaks a claim across the split.**
  `insurabench.model_selection` splits at the policy-period grain, not by
  shuffling policies and claims independently.
- **GLM wrapping doesn't reimplement glum, but doesn't trust its defaults
  blindly either.** `family="tweedie"` requires an explicit `power` --
  glum's own default (`power=0`, i.e. Normal) is silently wrong for
  insurance data. Rating-factor columns are cast to pandas `category`
  dtype before fitting -- a plain string column is silently dropped from
  the design matrix by glum's underlying `tabmat` otherwise.

## Development

```bash
pip install -e ".[dev]"
pytest
```
