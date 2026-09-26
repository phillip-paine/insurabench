"""Frequency-severity composition -- the second pure-premium route (design
brief §4, build order step 4's stated follow-on: "build ``frequency_severity.py``
composition ... only after both wrappers pass independently", which is now
true for ``GLMPricingModel`` and ``GBMPricingModel``).

``insurabench`` offers two ways to get a pure-premium model:

1. Direct -- a single ``GLMPricingModel``/``GBMPricingModel`` fit with
   ``target="pure_premium"`` (Tweedie compound Poisson-Gamma).
2. Composed -- ``FrequencySeverityModel`` here, wiring together two
   *independently chosen* ``PricingModel``s: one fit on
   ``target="frequency"``, one on ``target="severity"``, multiplied
   together. Either leg can be a GLM or a GBM, and the two legs need not
   match (a GLM frequency leg composed with a GBM severity leg is fine).

Both routes must be directly comparable through the curves/evaluation
layer (design brief §4's model-agnosticism requirement extends to this
composition, not just to GLM-vs-GBM) -- ``predict_policy_frame`` returns
the same pure-premium *rate* scale (``claim_amount_total / exposure``)
that ``GLMPricingModel``/``GBMPricingModel`` return for
``target="pure_premium"``, and ``score_policy_frame`` reports D² on the
same Tweedie-deviance scale (see ``insurabench.models.base``'s
``_tweedie_unit_deviance``, shared with the direct route for exactly this
reason).

Deliberately does NOT subclass ``PricingModel``. That ABC's ``fit``/
``predict`` are single-``X`` methods (see its docstring) -- and the whole
point of frequency-severity composition is that the two legs fit on two
different views (frequency: one row per policy-period; severity: one row
per claim). Forcing a single ``.fit(X, y)`` here would mean silently
picking one leg's ``X``/``y`` and hiding what happens to the other, which
is exactly the kind of "invisible reshape" design brief §3.3/§9.2 warns
against. Model-agnosticism in this codebase is enforced by
``curves._common.build_target_frame`` consuming a plain ``y_pred`` array
-- never by every model type sharing one abstract low-level interface
(see that module's own docstring) -- so mirroring
``GLMPricingModel``/``GBMPricingModel``'s ``fit_policy_frame`` /
``predict_policy_frame`` / ``score_policy_frame`` trio is what actually
matters here, and is what this class does instead.

Known limitation -- ``curves.partial_dependence`` is NOT supported for a
composed model yet, and ``FrequencySeverityModel`` does not implement a
``.predict(X, offset=...)`` that could paper over it. ``partial_dependence``
drives a fitted model's own low-level ``.predict`` directly on perturbed
feature columns alone, with no exposure column available in ``X`` for
``target="pure_premium"`` (exposure enters only as an offset for the
frequency-only case -- see its ``_offset_for``). But this composition's
frequency leg specifically needs ``offset=log(exposure)`` at predict time
to produce a pure-premium prediction at all, and there is nowhere for
that exposure to come from inside ``partial_dependence``'s call. Rather
than smuggle exposure through some side channel, this is left
unsupported and documented -- consistent with this codebase's habit of
failing loudly over guessing (design brief §9, ``exceptions.py``). If a
composed model's PDP is ever wanted, ``curves.partial_dependence`` itself
needs to grow a way to pass exposure through for this one case; that is a
curves-layer change, not something to hack around here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurabench.data.policy_frame import PolicyFrame
from insurabench.models.base import PricingModel, _tweedie_unit_deviance, build_design


class FrequencySeverityModel:
    """Composes two independently-fit ``PricingModel``s into a pure-premium
    model: ``pure_premium_rate = frequency_prediction * severity_prediction
    / exposure``.

    Parameters
    ----------
    frequency_model, severity_model:
        Unfitted (or at least not-yet-fit-for-this-purpose) ``PricingModel``
        instances, each configured for its own target -- e.g.
        ``GLMPricingModel(family="poisson")`` for ``frequency_model`` and
        ``GBMPricingModel(family="gamma")`` for ``severity_model``.
        ``fit_policy_frame`` fits each internally (via its own
        ``fit_policy_frame(pf, "frequency" | "severity", ...)``), so pass
        configured-but-unfitted instances, not ones you've already fit
        yourself against some other target -- family/target-compatibility
        checks (e.g. ``GBMPricingModel`` requiring ``family="gamma"`` for
        severity) run exactly as they do for direct use of either wrapper.

        The same object must not be passed for both legs: fitting a
        ``GLMPricingModel``/``GBMPricingModel`` re-fits its internal
        estimator and overwrites its own ``target_``/``peril_``/
        ``coverage_`` bookkeeping in place, so fitting one instance twice
        (once per leg) would silently clobber whichever leg was fit
        first, rather than raising -- guarded against explicitly here.
    """

    def __init__(self, frequency_model: PricingModel, severity_model: PricingModel) -> None:
        if frequency_model is severity_model:
            raise ValueError(
                "frequency_model and severity_model must be two separate "
                "instances -- fitting the same model instance twice (once "
                "per target) overwrites its internal estimator in place, "
                "silently discarding whichever leg was fit first."
            )
        self.frequency_model = frequency_model
        self.severity_model = severity_model

        self.peril_: str | list[str] | None = None
        self.coverage_: str | list[str] | None = None
        self._fitted = False

    def fit_policy_frame(
        self,
        pf: PolicyFrame,
        *,
        peril: str | list[str] | None = None,
        coverage: str | list[str] | None = None,
    ) -> FrequencySeverityModel:
        """Fit both legs on ``pf``, each on its own view
        (``frequency_view``/``severity_view``) via its own
        ``fit_policy_frame`` -- see ``GLMPricingModel.fit_policy_frame``/
        ``GBMPricingModel.fit_policy_frame`` for what ``peril``/
        ``coverage`` restrict (the same peril/coverage filter is applied
        to both legs, so e.g. a theft-only composed pure-premium model
        has a theft-only frequency leg and a theft-only severity leg)."""
        self.frequency_model.fit_policy_frame(pf, "frequency", peril=peril, coverage=coverage)
        self.severity_model.fit_policy_frame(pf, "severity", peril=peril, coverage=coverage)
        self.peril_ = peril
        self.coverage_ = coverage
        self._fitted = True
        return self

    def _predict_rate(self, pf: PolicyFrame, *, for_score: bool = False):
        """Shared plumbing for ``predict_policy_frame``/``score_policy_frame``:
        build X/exposure once from ``pf``'s ``pure_premium_view`` (the
        grain both a composed and a direct pure-premium model predict on),
        run both legs' low-level ``.predict`` against it, and combine.

        Reuses ``models.base.build_design``'s ``"pure_premium"`` branch
        purely for its ``(X, exposure)`` construction (and, when
        ``for_score``, its observed rate ``y``) -- not because this is a
        Tweedie fit, but because it is exactly the same feature/exposure
        construction a direct Tweedie pure-premium model uses, and reusing
        it is what keeps the two pure-premium routes on an identical
        feature grain rather than risking two hand-rolled, quietly
        different views of the same thing.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_policy_frame before predict_policy_frame/score_policy_frame.")
        X, y, exposure, _ = build_design(
            pf, "pure_premium", peril=self.peril_, coverage=self.coverage_, for_predict=not for_score
        )
        freq_pred = self.frequency_model.predict(X, offset=np.log(exposure))
        sev_pred = self.severity_model.predict(X)
        mu = freq_pred * sev_pred / exposure
        return (mu, y, exposure) if for_score else mu

    def predict_policy_frame(self, pf: PolicyFrame) -> np.ndarray:
        """Predict the pure-premium *rate* (``claim_amount_total /
        exposure``) on a (typically held-out) PolicyFrame, using the
        ``peril``/``coverage`` set by ``fit_policy_frame``.

        Same scale as ``GLMPricingModel``/``GBMPricingModel``'s own
        ``predict_policy_frame`` under ``target="pure_premium"`` -- so
        it plugs directly into every ``curves``/``evaluation`` function
        that takes ``target="pure_premium"`` (e.g. ``one_way_curve``,
        ``lift_chart``, ``gini_index``), the same way either wrapper's
        direct-Tweedie route does. ``curves.partial_dependence`` is the
        one exception -- see the module docstring.

        Same caveat as ``GLMPricingModel.predict_policy_frame`` regarding
        genuinely prospective new-business scoring (no claims history at
        all): this method's ``pure_premium_view`` assumes a claims table
        is present.
        """
        return self._predict_rate(pf)

    def score_policy_frame(self, pf: PolicyFrame, *, power: float) -> float:
        """Deviance-based D² (1 for a perfect fit, 0 for a model no
        better than the weighted mean), on the same Tweedie-deviance
        scale ``GLMPricingModel``/``GBMPricingModel`` report for
        ``target="pure_premium"`` -- so a composed model's D² is directly
        comparable to a direct Tweedie fit's.

        ``power`` is required, not defaulted, matching this codebase's
        established convention for Tweedie power (``models.glm.
        _resolve_family``, ``models.gbm._resolve_loss``): there is no
        universally-reasonable default, and silently picking one (e.g.
        glum's own ``power=0``/Normal default) has already been a real
        footgun elsewhere in this codebase. Pass whatever power is
        appropriate for the book's frequency/severity mix (typically
        ``1 < power < 2``).
        """
        mu, y, exposure = self._predict_rate(pf, for_score=True)
        unit_dev = _tweedie_unit_deviance(y, mu, power)
        null_mu = np.full_like(y, np.average(y, weights=exposure))
        null_dev = _tweedie_unit_deviance(y, null_mu, power)
        mean_dev = np.average(unit_dev, weights=exposure)
        mean_null_dev = np.average(null_dev, weights=exposure)
        if mean_null_dev == 0:
            return float("nan")
        return float(1.0 - mean_dev / mean_null_dev)

    def relativities(self) -> dict[str, pd.DataFrame]:
        """Each leg's own ``.relativities()``, kept separate rather than
        combined into one table.

        A GLM leg's relativities are real multiplicative per-level
        numbers; a GBM leg's are ``PredictionValuesChange`` importances
        (see ``GBMPricingModel.relativities``) -- multiplying those
        together (or even multiplying two GLM legs' relativities without
        re-normalizing base levels against this composed model's own
        predictions) would produce a number with no coherent
        interpretation. For a properly base-level-normalized combined
        rating-factor view, run ``curves.relativity_table`` against
        ``predict_policy_frame``'s output instead -- that's exactly the
        design brief §5 tool built for this, and it's already
        model-agnostic by construction.
        """
        return {
            "frequency": self.frequency_model.relativities(),
            "severity": self.severity_model.relativities(),
        }
