# insurabench — Progress Notes & Bulletin Board

Companion to `insurabench_design_brief.md` and `insurabench_build_order.md`.
Those two describe what to build and in what order; this one is a running
log of what's actually been built, plus hard-won gotchas that cost real
debugging time the first time round. **Read the "Bulletin board" section
before touching `aggregate.py`, `models/`, or anything that joins
policies to claims.** Update this file at the end of each build-order step.

---

## Progress so far

### Build order step 1 — Data layer: DONE
### Build order step 2 — GLM wrapper: DONE

(unchanged from before — see prior notes / git history.)

### Build order step 3 — Curves + evaluation layer: DONE

Everything in design brief §5/§6 built and proven against `GLMPricingModel`.
Full function list: `curves.one_way_curve`, `curves.two_way_curve`,
`curves.relativity_table`, `curves.partial_dependence` (the one function
that takes a fitted model directly rather than a `y_pred` array — a PDP
inherently re-predicts after perturbing a feature);
`evaluation.lift_chart`, `evaluation.double_lift_chart`,
`evaluation.gini_curve`/`gini_index`,
`evaluation.calibration_table`/`calibration_index`,
`evaluation.bootstrap_relativities`/`split_relativities`/
`stability_summary`, `evaluation.rate_change_by_level`/
`rate_change_distribution` (repricing/disruption impact between two
models' predictions — deliberately a different question from
`double_lift_chart`'s model-accuracy comparison; needs no actual outcome
at all).

Shared plumbing: `curves._common.build_target_frame` (normalizes
frequency/severity/pure_premium onto a common numerator/weight/
predicted_numerator triple — every curve/evaluation function is built on
this and never calls a model) and `exposure_deciles` (the one fixed
decile-by-cumulative-exposure-share bucketing convention, shared by
`lift_chart`/`double_lift_chart`/`calibration_table`).

### Build order §8 — viz layer: DONE (built ahead of geo/reporting)

`insurabench/viz/` — one plotting function per curves/evaluation output
(`plot_one_way_curve`, `plot_two_way_curve`, `plot_relativity_table`,
`plot_partial_dependence`, `plot_lift_chart`, `plot_double_lift_chart`,
`plot_gini_curve`, `plot_calibration_table`, `plot_stability_summary`,
`plot_rate_change_by_level`, `plot_rate_change_distribution`), all pure
presentation (take the DataFrame/float a curves/evaluation call already
produced, never a model). Shared `viz.theme` house style (semantic
`PALETTE`, matplotlib `rc_context`, an exposure-bars-behind-rate-lines
combo-chart helper). Every function takes `ax=`/`save_path=`.

### Integration testing: DONE

`tests/test_end_to_end.py` — automated pipeline test on a realistic
2000-policy synthetic book (not a 4-row fixture): data layer → split →
GLM fit → every curves/evaluation/viz function, with loose sanity
assertions (the toy-fixture tests own correctness; this owns "do the
pieces actually compose"). `docs/examples/quickstart.py` — a standalone
runnable script (not a test) producing real PNGs + a
`relativity_table.csv`.

Running this for real (not just writing it) surfaced a genuine finding:
a default frequency Poisson GLM on the synthetic book scores a small
**negative** deviance D² on both train and test, despite small-positive
Gini (~0.07–0.08) — i.e. it doesn't beat an intercept-only null model by
deviance despite some real ranking ability. Likely a functional-form
issue (linear terms for `driver_age`/`vehicle_power` probably don't
match the synthetic generator's threshold-style injected signal). Not
fixed — flagged in both the test (as a comment explaining why `D² > 0`
is deliberately NOT asserted) and the quickstart script's own printed
output.

### Build order step 4 — GBM wrapper: DONE

`insurabench/models/gbm.py` — `GBMPricingModel`, wraps
`catboost.CatBoostRegressor`, same `PricingModel` contract as
`GLMPricingModel` (`fit`/`predict`/`relativities`,
`fit_policy_frame`/`predict_policy_frame`/`score_policy_frame`).

**CatBoost specifically, not LightGBM/XGBoost, on cited evidence**: So,
B. (2024), "Enhanced Gradient Boosting for Zero-Inflated Insurance
Claims and Comparative Analysis of CatBoost, XGBoost, and LightGBM",
*Scandinavian Actuarial Journal*, 2024(10) (arXiv:2307.07771) —
benchmarks all three on French MTPL + a synthetic telematics dataset,
finds CatBoost consistently best, largely attributed to its "Ordered
Target Statistic" categorical encoding suiting high-cardinality rating
factors. Four families:

- `family="poisson"` — CatBoost's native Poisson loss. The commodity
  baseline (design brief §1/§4: the model layer isn't the differentiator).
- `family="zip"` — **the paper's actual contribution**: the ZIPB1
  zero-inflated Poisson boosted tree (eqs. 13–19). Reparameterizes the
  inflation probability `p` as a function of the fitted mean `mu` via one
  fixed hyperparameter `gamma` (`p = 1/(1+mu**gamma)`), so it's a single
  tree ensemble rather than two — the paper's other variant, ZIPB2 (`p`
  and `mu` modeled independently via alternating coordinate descent
  across two ensembles), is **not implemented** — genuinely different
  training loop, bigger engineering lift, and the paper itself recommends
  ZIPB1 specifically when the goal is feature interpretation, which is
  this library's whole positioning. Implemented as a CatBoost custom
  objective (`_ZIPB1Objective`).
- `family="gamma"` — a custom Gamma GLM-equivalent loss (log link), for
  `target="severity"`. **Not** implemented via CatBoost's native Tweedie
  — see bulletin item 13.
- `family="tweedie"` — CatBoost's native Tweedie loss, `1 < power < 2`,
  for `target="pure_premium"` only.

`.relativities()` returns CatBoost's normalized `PredictionValuesChange`
feature importances (NOT a multiplicative relativity — a GBM has no
per-level coefficient; documented explicitly as a different kind of
per-model sanity check than GLM's, per the `PricingModel.relativities`
ABC docstring). `.score()`/`score_policy_frame()` computes a proper
deviance-based D² matching each family (Poisson deviance; the ZIP
deviance from So (2024) §4.1.1, verbatim; standard Tweedie deviance) —
deliberately not CatBoost's own generic `.score()` (plain R², wrong for
any of these families) — so it's semantically comparable to
`GLMPricingModel.score`.

**Refactor**: `models/base.py` gained a shared `build_design(pf, target,
...)` — the `(X, y, sample_weight, offset)` construction previously
private to `glm.py`, now used by both wrappers so they can't drift apart
on how a target is built (same rationale as `curves._common`). Verified
the refactor was safe by rerunning the full suite immediately after,
before writing any GBM code on top of it.

**Correctness verification** (not just "it ran and looked plausible"):
both custom objectives' analytic gradient/hessian were checked against a
central finite-difference derivative of their own loss function
directly, across a wide `(y, raw score, gamma)` sweep — baked into
`tests/test_gbm.py` as `test_zip_objective_matches_finite_difference_...`
/ `test_gamma_objective_matches_finite_difference_...`, not just run
ad hoc. Separately, CatBoost's custom-objective sign convention itself
(`der1/der2 = -gradient/-hessian` of the loss) was verified by hand-
building a trivial custom Poisson objective and confirming it reproduces
CatBoost's *native* Poisson predictions to within floating-point error
before either paper formula was trusted.

**Two real bugs found and fixed while building this**, both by actually
running the thing against real data rather than trusting the code once
it imported cleanly:

1. CatBoost's native Tweedie loss **rejects `variance_power=2`
   outright** (raises `CatBoostError`) — despite power=2 being
   mathematically the Gamma distribution, a valid Tweedie special case.
   Confirmed directly, not assumed. This is why severity gets its own
   `family="gamma"` custom objective rather than reusing `"tweedie"` the
   way `GLMPricingModel` reuses one Tweedie family for both severity and
   pure premium.
2. **Severity cold-start collapse**: with no natural offset (unlike
   frequency's `log(exposure)`), a custom objective starts every row's
   raw score at exactly 0 (`mu=1`). For a severity target whose true
   scale is in the thousands, the trees have to climb the *entire*
   log-scale distance from scratch — at a low-but-plausible iteration
   count this produced predictions collapsed near 1 (off by ~100x) while
   still "training successfully" with no error or warning. A GLM gets an
   equivalent for free from its exact intercept; boosting has none.
   Fixed by auto-computing `baseline = log(weighted mean of y)` at fit
   time for any custom objective given no explicit offset, and reusing
   it automatically at predict time — the boosting equivalent of a GLM's
   intercept. Regression test:
   `test_gamma_severity_lands_on_the_right_scale_even_at_few_iterations`.

**Model-agnosticism proof** (build order step 4's actual point):
`tests/test_gbm.py`'s `test_curves_and_evaluation_run_unchanged_against_a_fitted_gbm`
and `test_glm_and_gbm_gini_are_directly_comparable` rerun
`one_way_curve`/`relativity_table`/`lift_chart`/`gini_index` against a
fitted `GBMPricingModel`'s predictions with zero GBM-specific branches
anywhere in `curves`/`evaluation` — confirming the abstraction built in
step 3 wasn't accidentally GLM-shaped. On the synthetic book, the ZIP
model's D² (~0.022) beat plain Poisson boosted tree's (~0.010), the same
direction So (2024) reports.

**Top-level `insurabench/__init__.py` and `README.md` were both stale**
by this point — the `__init__.py` docstring/`__all__` still described
step-2-era state (no curves/evaluation/viz/GBM exports at all); the
README claimed curves/evaluation/GBM "not implemented yet" and
instructed `pip install -e ".[dev]"` against a `[dev]` extras group that
doesn't exist (confirmed: pip just warns and silently skips it, no dev
tools actually installed). `__init__.py` was brought fully current this
step. **README.md is still not fixed** — flagged, not yet done.

`pyproject.toml`: added `matplotlib` (viz step) and `catboost` (this
step) as dependencies; separately fixed the long-flagged invalid
`requires-python = "^3.12"` (poetry-style, not PEP 621) to `">=3.10"`,
and aligned the version string (`pyproject.toml` said `0.1.0`,
`__init__.py` said `0.1.0.dev0` — now both `0.2.0.dev0`) — both
flagged as drive-by fixes to Phillip, not made silently.

Full suite: **125 tests passing**, `ruff check --isolated insurabench
tests` clean.

### `models/frequency_severity.py` — DONE

`FrequencySeverityModel` composes two independently-configured
`PricingModel`s (either GLM or GBM, and the two legs need not match —
tested GLM+GLM, GLM+GBM, GBM+GLM, GBM+GBM) into the second pure-premium
route design brief §4 describes: `pure_premium_rate = frequency_pred *
severity_pred / exposure`, the alternative to a single model's direct
`target="pure_premium"` Tweedie fit.

**Deliberately does not subclass `PricingModel`.** The ABC's `fit`/
`predict` are single-`X` methods; composition inherently needs two
different views (frequency: policy-period grain; severity: claim grain),
so forcing one low-level `.fit(X, y)` would mean picking one leg's data
and hiding the other — exactly the "invisible reshape" design brief
§3.3/§9.2 warns against. Confirmed by rereading `curves/_common.py`'s own
docstring that model-agnosticism in this codebase is enforced by
`build_target_frame` consuming a plain `y_pred` array, never by every
model type sharing one low-level interface — so `FrequencySeverityModel`
instead mirrors `GLMPricingModel`/`GBMPricingModel`'s PolicyFrame-level
trio (`fit_policy_frame`/`predict_policy_frame`/`score_policy_frame`),
which is the level that actually matters for curves/evaluation
compatibility.

**Real design snag found while building this, not just plumbing**:
`curves.partial_dependence` is NOT supported for a composed model, and
this was worth working through rather than papering over. That module
drives a fitted model's low-level `.predict(X, offset=...)` directly on
perturbed feature columns, with `offset` only ever set for
`target="frequency"` (exposure enters `target="pure_premium"` purely via
`sample_weight` at *fit* time for a direct Tweedie model, never needed
again at predict time). But a composed model's frequency leg specifically
needs `offset=log(exposure)` at predict time to produce anything — and
`partial_dependence` never passes exposure through for the
`pure_premium` case, because a direct Tweedie model never needed it to.
Rather than smuggle exposure through some side channel (e.g. requiring
exposure to be a declared feature, which would double-count it elsewhere),
`FrequencySeverityModel` doesn't implement a low-level `.predict` at all
and this is documented as an explicit, known gap — consistent with this
codebase's fail-loudly-don't-guess habit. If a composed model's PDP is
ever wanted, `curves.partial_dependence` itself needs to grow a way to
pass exposure through for this one case; flagged, not done.

**Refactor**: `_poisson_unit_deviance`/`_tweedie_unit_deviance` moved
from `models/gbm.py` (where they were private to `GBMPricingModel.score`)
into `models/base.py`, since `FrequencySeverityModel.score_policy_frame`
needed the same Tweedie deviance formula to put a composed model's D² on
the same scale as a direct Tweedie fit's — same rationale as
`build_design`'s own earlier move from `glm.py` to `base.py`.
`models/gbm.py` re-imports both under their original names so existing
call sites, including `tests/test_gbm.py`'s direct imports of them from
`insurabench.models.gbm`, are unaffected — verified by rerunning the full
suite immediately after the move, before writing any
`frequency_severity.py` code on top of it (same safety-check habit as the
GBM step's own `build_design` refactor).

**Footgun guarded explicitly**: passing the *same* `PricingModel`
instance for both `frequency_model` and `severity_model` would silently
clobber whichever leg was fit first (fitting a `GLMPricingModel`/
`GBMPricingModel` re-fits its estimator and overwrites its own `target_`
bookkeeping in place) — raises `ValueError` at construction rather than
letting that happen unnoticed.

`score_policy_frame(pf, *, power)` requires `power` explicitly, no
default — matching this codebase's now-three-times-established
convention (`models.glm._resolve_family`, `models.gbm._resolve_loss`,
now this) of never letting Tweedie power default silently.

`relativities()` returns each leg's own relativities separately
(`{"frequency": ..., "severity": ...}`) rather than attempting a combined
multiplicative table — a GLM leg's relativities are real per-level
numbers, a GBM leg's are `PredictionValuesChange` importances, and
multiplying those together (or even multiplying two GLM legs' without
re-normalizing base levels against the composed model's own predictions)
would produce a number with no coherent interpretation. Documented
`curves.relativity_table` run against `predict_policy_frame`'s output as
the right way to get a properly base-level-normalized combined view.

Tests (`tests/test_frequency_severity.py`, 13 tests): basic GLM+GLM
plumbing, all four leg-type combinations (GLM+GLM, GLM+GBM, GBM+GLM,
GBM+GBM), the same-instance-for-both-legs guard, the family/target
mismatch guard (confirmed it still fires via delegation to each leg's own
`fit_policy_frame`, not reimplemented), model-agnosticism at the
curves/evaluation layer (mirroring `test_gbm.py`'s section 3 —
`one_way_curve`/`relativity_table`/`lift_chart`/`gini_index` all run
unchanged against a composed model's predictions), a composed-vs-direct-
Tweedie Gini/D² comparability check, and a peril-filter check confirming
the same peril is applied to both legs. Full suite now **138 tests
passing**, `ruff check --isolated insurabench tests` clean.

### Build order step 6 — reporting layer: DONE

`reporting/model_card.py` (`generate_model_card`/`ModelCard`) and
`reporting/export.py` (`export_rating_table`), both against design brief
§8. Both are thin consumers of curves/evaluation/models, as expected of
this step — except one part wasn't, and is the actual finding from this
step.

**Real bug found, not just plumbing: glum's own `.aic()`/`.bic()` are
silently wrong for any offset-fit model.** Their signature
(`aic(X, y, sample_weight=None, *, context=None)`) has no `offset`
parameter at all, and internally recompute mu via a bare `self.predict(X)`
with no offset — so for `target="frequency"` (always offset-fit), glum's
native AIC/BIC evaluate log-likelihood at the wrong mu. Confirmed
directly, not assumed: on a synthetic offset-Poisson fit, the correctly
offset-adjusted AIC was 1352.4 vs. glum's own (broken) 1419.1 — a ~5%
error, not a rounding-level difference. Fixed by adding
`GLMPricingModel.information_criteria`/`.information_criteria_policy_frame`
(`models/glm.py`), which recomputes AIC/BIC from the fitted family's own
`log_likelihood`, evaluated at `self.predict(X, offset=offset)` — the
same offset-aware call site `fit`/`predict`/`score` already use. Same
degrees-of-freedom convention as glum's own (non-zero coefficients +
intercept), same ridge-regularization "not well defined" warning
preserved. `reporting.model_card` calls this corrected method, never
glum's own. This is now the fourth glum-specific footgun this codebase
has had to catch and guard against explicitly (alongside categorical-
column-dropping, the Tweedie `power=0`/Normal default, and the
`fit_intercept`/`drop_first` rank-deficiency issue) — see bulletin item
17.

`generate_model_card(model, target, test_pf, *, train_pf=None, peril=,
coverage=, n_deciles=, top_n=, power=)` → a `ModelCard` dataclass: model
type/family, train/test D², Gini, calibration index, a lift-ratio scalar
(top decile actual / bottom decile actual, alongside the full lift
table), AIC/BIC, and the top-N rating-factor relativities by deviation
from 1.0. Handles all three model types uniformly:

- AIC/BIC: `None` (with an explicit "not applicable" in `.to_markdown()`)
  for `GBMPricingModel` (no closed-form likelihood/parameter count in the
  classical sense) and `FrequencySeverityModel` (two legs, no single
  joint likelihood) — reporting `None` and saying so beats fabricating a
  number that looks precise but isn't well-defined.
- `family`: a plain string for GLM/GBM, a `{"frequency": ..., "severity":
  ...}` dict for a composed model.
- Top relativities: deliberately built from `curves.relativity_table`
  (base-level-normalized, model-agnostic) rather than each model's own
  `.relativities()` — a GLM's are real per-level numbers, a GBM's are
  `PredictionValuesChange` importances, and a top-N-by-deviation ranking
  across those wouldn't mean the same thing. Each model's own raw
  `.relativities()` is still exposed separately as `raw_relativities`,
  for anyone who wants the model-specific diagnostic too.
- `power` is required (no default) to score a `FrequencySeverityModel` on
  `target="pure_premium"`, matching that class's own
  `score_policy_frame` — threaded through rather than silently picked
  here.

Unlike curves/evaluation (which never touch a model — see
`curves/_common.py`'s own docstring), `model_card` does touch the model
directly, the same way `curves.partial_dependence` does and for the same
reason: AIC/BIC and raw relativities aren't derivable from a plain
`y_pred` array. Everywhere a number IS derivable that way (Gini,
calibration, lift, top relativities), `model_card` still routes through
the existing curves/evaluation functions rather than recomputing
anything.

`export_rating_table(pf, target, y_pred, output_path, ...)` — builds on
`curves.relativity_table`, adds an explicit `is_base` column (the row
per feature whose relativity is closest to 1.0 — not an exact
`relativity == 1.0` floating-point equality check, which the division
producing `relativity` isn't guaranteed to hit exactly even for the base
row itself). Format chosen by `output_path`'s suffix: `.csv` (one long
tidy table) or `.xlsx` (one sheet per rating factor, sheet names
truncated to Excel's 31-char limit — the layout an actuarial rating-table
workbook actually has, not a tidy table they'd have to pivot themselves).
`.xlsx` requires `openpyxl` — added as a new optional
`[project.optional-dependencies] excel` extra in `pyproject.toml` (not a
hard dependency), raising a clear `ImportError` with the install
instruction if missing rather than a cryptic one from inside pandas.

Wired into `docs/examples/quickstart.py`: now also writes `model_card.md`
and `rating_table.xlsx` alongside the existing PNGs/CSV. Ran end to end
to confirm, not just unit-tested in isolation.

Tests: `tests/test_glm.py` (new — closes part of the previously-flagged
"no dedicated GLM test file" gap, focused on `information_criteria`):
matches a hand-computed offset-aware AIC/BIC exactly, a regression test
confirming it *disagrees* with glum's own broken `.aic()` (so if this
ever starts passing because the two numbers match, that means glum fixed
the upstream bug — worth investigating, not deleting), policy-frame vs.
low-level-call consistency, pre-fit `RuntimeError`, finite AIC/BIC for
severity/pure_premium (no offset bug there either way, just confirming
it still works), and the ridge-regularization warning.
`tests/test_reporting.py` (new): card assembly correctness for all three
model types, the composed-model `power` requirement, top-relativities
sorting, markdown rendering, and both export formats — including
confirming exactly one `is_base` row per feature and that flagged row's
relativity is actually 1.0. Full suite now **153 tests passing**, `ruff
check --isolated insurabench tests` clean.

### Workbook/ -- demo notebooks and sample data: DONE

Four Jupyter notebooks plus a small fixed sample dataset, aimed at
advertising/onboarding rather than CI (that's still `docs/examples/
quickstart.py`'s job -- plain Python, runs in CI, checks the pipeline
still works end to end; `Workbook/` is narrative, rendered notebooks,
each pre-executed with real output committed so it reads correctly on
GitHub without anyone needing to run it first):

- `01_getting_started.ipynb` -- load the two sample CSVs (`Workbook/
  data/`, 200 policies/414 periods/89 claims, fixed seed), fit a
  frequency GLM, check D²/Gini/calibration, a lift chart, a one-way
  curve. Framed entirely as "you want to do X" steps, no linkage/schema
  theory -- that's deliberately NOT what this notebook is for.
- `02_model_agnostic_pricing.ipynb` -- the headline pitch: a GLM and a
  GBM, fit on a larger (4,000-policy) synthetic book, with *identical*
  curves/evaluation code run against both, plus a direct double-lift
  comparison.
- `03_pricing_the_whole_book.ipynb` -- direct Tweedie GLM vs. a composed
  `FrequencySeverityModel` (GBM frequency leg + GLM severity leg),
  framed as "which route do you take to get a full price," compared on
  Gini/D²/lift, landing on an honest "depends what you need, check your
  own numbers" takeaway rather than declaring a universal winner.
- `04_from_model_to_deliverables.ipynb` -- `generate_model_card` +
  `export_rating_table`, closing the loop from fitted model to the two
  things reporting produces.

Sample data (`Workbook/data/sample_policies.csv`/`sample_claims.csv`) is
`make_synthetic_two_table(n_policies=200, renewals=True, seed=42)`,
fixed and checked in (not regenerated at notebook-run time) specifically
so it's small enough to open directly in Excel/a text editor -- with its
own `README.md` data dictionary. Notebooks 02-04 generate their own
larger, in-memory book instead (the sample data is too small for GBM/GLM
to meaningfully differ).

**Second gotcha, caught by Phillip reviewing rather than by me**: the
root `.gitignore`'s blanket `*.csv` rule (there specifically so no real
insurance data ever gets committed) has no directory scoping, so it
silently also swallowed `Workbook/data/sample_policies.csv`/
`sample_claims.csv` -- confirmed directly with a throwaway `git init` +
`git check-ignore -v`, not just reasoned about: both files reported as
ignored by that exact root rule. The two sample CSVs are deliberately
fixed, checked-in data (the entire point of having them), not generated
output, so this would have meant the sample data silently never made it
into version control at all. Fixed with a scoped negation in
`Workbook/.gitignore` (`!data/*.csv`) rather than loosening the root
rule -- keeps "no real insurance data committed" intact everywhere else
in the repo. Re-verified with the same throwaway-git-repo method that
the two CSVs are now tracked, and separately scanned every other `.py`/
`.md`/`.ipynb`/`.csv`/`.toml` file in the repo against `git ls-files`
to confirm nothing else has this same blind spot -- nothing else did.

**Real bug found and fixed while building these, not just notebook
content**: every `insurabench.viz.plot_*` function silently stopped
producing any visible output after the *first* plot in a Jupyter/inline-
backend session -- no error, nothing. Confirmed directly, isolated down
to a minimal repro: `matplotlib_inline`'s auto-display post-execute hook
stops firing for any figure drawn inside a second (or later)
`with plt.rc_context(...):` block in the same kernel session -- and
every `insurabench.viz` function wraps its drawing in `theme()`, which
is exactly such an rc_context block (by design, so the shared style
never leaks into the caller's own matplotlib state). This would have hit
literally the first person who tried to make more than one insurabench
chart in a real notebook -- which is to say, every actual user. Fixed by
adding an explicit `plt.show()` at the end of `viz/theme.py`'s
`finalize()` (harmless under a non-interactive backend like Agg, a
script/CI run's default, where `plt.show()` is simply a no-op). Pinned
with a regression test (`tests/test_viz.py::test_finalize_calls_plt_show`,
monkeypatching `plt.show` and asserting `finalize` calls it) -- the
existing `test_viz.py` suite forces the Agg backend at import time for
every other test in that file, so it structurally could never have
caught this; worth remembering `test_viz.py`'s own structural checks
don't cover anything about actual interactive/notebook display behavior.
`viz/theme.py`'s module docstring, which had claimed `theme()` was "safe
... repeatedly inside a dashboard process" without this having actually
been verified for repeated *display*, corrected to point at this.

Full suite now **154 tests passing**, `ruff check --isolated insurabench
tests` clean.

### Not started yet

Geo (`geo/credibility_smoothing.py`, `geo/spatial_glm_term.py`,
`geo/plotting.py`) — the one remaining item from design brief §2's v1
must-have list. Fairness stub. ZIPB2 (see above). `README.md` still
stale. `test_model_selection.py` still doesn't exist (the GLM-specific
gap is now partly closed by `tests/test_glm.py`, but `model_selection.py`
itself still has no dedicated test file — only indirect coverage via
`test_end_to_end.py`). `curves.partial_dependence` support for a composed
`FrequencySeverityModel` (see above — needs a curves-layer change, not a
model-layer one).

---

## Bulletin board — things that will cost you time if you relearn them

(Items 1–12 unchanged from before — see prior notes.)

### 13. CatBoost's native Tweedie loss rejects `variance_power=2` (the Gamma case)

Mathematically, Tweedie power=2 *is* the Gamma distribution — but
CatBoost's own implementation raises `CatBoostError` at that exact
boundary (confirmed directly against the installed version, not assumed
from docs). Don't reach for `family="tweedie", power=2.0` for a severity
model expecting it to work like `GLMPricingModel`'s Tweedie does — use
`GBMPricingModel(family="gamma")` instead, a separate custom objective
built specifically because of this. If you're ever tempted to "simplify"
by merging `"gamma"` back into `"tweedie"` with `power=2`, don't — it
will fail at `.fit()`, not at construction, so the failure is easy to
attribute to the wrong thing.

### 14. A custom CatBoost objective with no offset needs an auto-baseline, or it silently under-converges

Any future custom objective added to `models/gbm.py` that has no natural
per-row offset (the way frequency has `log(exposure)`) will start every
row at `raw=0` (`mu=1`) and can look like it "trained successfully" —
no error, no warning — while actually being badly under-converged if the
target's true scale is far from 1, especially at a plausible-but-modest
iteration count. This isn't a hypothetical: it's exactly what happened
building the `"gamma"` family (see step 4 above), caught only by
actually checking the fitted predictions' scale against the data's own
mean, not by the code running without error. `GBMPricingModel.fit`
already handles this generically for `self._uses_custom_objective`
families with no supplied offset — reuse that mechanism (the
`_auto_baseline_` attribute, set in `fit`, applied automatically in
`predict`) for any new offset-less custom objective rather than
re-deriving it.

### 15. CatBoost custom objectives need an explicit `eval_metric`

`CatBoostRegressor(loss_function=<custom object>)` raises
(`"If loss function is a user defined object, then the eval metric must
be specified"`) unless `eval_metric=` is also passed — a built-in metric
name like `"Poisson"` or `"RMSE"` used purely for monitoring/early
stopping, unrelated to the actual training gradient your custom object
supplies. `_resolve_loss` in `gbm.py` already threads this through for
both `"zip"` and `"gamma"`; a new custom-objective family needs the same.

### 16. `FrequencySeverityModel` has no low-level `.predict` — don't try to feed it to `curves.partial_dependence`

Every other curves/evaluation function only ever sees a plain array
(`predict_policy_frame`'s output), so `FrequencySeverityModel` supports
all of them. `curves.partial_dependence` is the one exception in the
whole package: it drives a fitted model's low-level `.predict(X,
offset=...)` directly, and for `target="pure_premium"` it calls that with
`offset=None` (a direct Tweedie model's Fit uses exposure only as a
sample weight, never needing it again at predict time). A composed
model's frequency leg genuinely needs `offset=log(exposure)` to produce
anything — and there's no exposure column inside the `X` `partial_
dependence` builds to fall back on. `FrequencySeverityModel` intentionally
has no `.predict()` at all, so calling it that way fails immediately and
obviously (`AttributeError`) rather than quietly returning a wrong
number. If a composed-model PDP is ever wanted, this needs a
`curves/partial_dependence.py` change to pass exposure through for this
one case — not a workaround on the model side.

### 17. glum's own `.aic()`/`.bic()` are wrong for any offset-fit model — don't call them directly

Their signature (`aic(X, y, sample_weight=None, *, context=None)`) has no
`offset` parameter, and internally recompute mu via a bare
`self.predict(X)` with no offset applied. This is silent — no warning,
no error — and wrong specifically for `target="frequency"` fits (always
offset-fit in this codebase) or any other offset-fit model someone builds
by calling glum directly. Confirmed directly (not assumed from reading
the source): on a synthetic offset-Poisson fit, the correctly
offset-adjusted AIC was 1352.4 vs. glum's own 1419.1 — several percent
off, not a rounding artifact. Use
`GLMPricingModel.information_criteria`/`.information_criteria_policy_frame`
(`models/glm.py`) instead — never `model._estimator.aic()`/`.bic()`
directly. This is the fourth glum-specific footgun caught in this
codebase (alongside categorical-column-dropping — item 2, the Tweedie
`power=0` default — item 3, and the `fit_intercept`/`drop_first`
rank-deficiency issue — item 4) — worth remembering glum's convenience
methods generally deserve the same "check it against a hand-computation
before trusting it" treatment those three already got, not just `.aic()`/
`.bic()` specifically.

---

## Environment / tooling notes

- Python ≥3.10, `pandas>=2.0`, `numpy>=1.24`, `glum>=3.0` (tested against
  3.4.1), `scipy`, `matplotlib>=3.9`, `catboost>=1.2` (tested against
  1.2.10). Optional `[excel]` extra: `openpyxl>=3.1` (only needed for
  `export_rating_table(..., '.xlsx')`). No `pygam`/`geopandas` yet.
  `Workbook/`'s notebooks need `jupyter` to run (not a package
  dependency -- nothing in `insurabench/` itself needs it).
- `ruff check --isolated insurabench tests` is clean; keep it that way.
  CI-equivalent for this repo is
  `ruff check --isolated insurabench tests && pytest -q` — the
  `requires-python` string that used to break a bare `ruff check .` is
  now fixed (see step 4 above), so a plain `ruff check .` from the repo
  root should work too now; `--isolated` is kept above only out of habit
  and could probably be dropped, worth re-testing.
- Test suite: `tests/` at the repo root, 154 tests. Fixtures in
  `tests/conftest.py` (`simple_book`, `renewal_book`, `toy_frequency_pf`,
  `toy_severity_pf`, `toy_two_feature_pf`) plus a module-scoped realistic
  synthetic book fixture local to `test_end_to_end.py`, `test_gbm.py`,
  `test_frequency_severity.py`, `test_glm.py`, and `test_reporting.py`
  each (not shared — each file builds its own at a size suited to what
  it's testing).
