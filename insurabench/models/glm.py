"""GLM pricing model -- a thin wrapper over glum (design brief §4).

insurabench does not reimplement GLM fitting; glum already handles
Poisson/Gamma/Tweedie/Negative-Binomial families, native exposure
offsets/weights, elastic-net regularization, and standard errors, and does
it well. This wrapper's job is to plug that engine into insurabench's
PolicyFrame / PricingModel contract, and to close off one specific glum
footgun (see ``_resolve_family``).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution

from insurabench.data.policy_frame import PolicyFrame
from insurabench.models.base import PricingModel, Target, build_design

Family = Literal["poisson", "gamma", "tweedie", "negative_binomial", "normal"]

_FAMILY_STRINGS = {
    "poisson": "poisson",
    "gamma": "gamma",
    "negative_binomial": "negative.binomial",
    "normal": "normal",
}


def _resolve_family(family: Family, power: float | None):
    """Build the glum ``family`` argument.

    ``glum.GeneralizedLinearRegressor(family="tweedie")`` -- a bare string,
    no power -- silently fits with ``power=0``, i.e. a *Normal*
    distribution, not a compound Poisson-Gamma Tweedie, because that is
    ``TweedieDistribution``'s own default. For insurance pure-premium
    modeling that default is almost never what's wanted, so insurabench
    requires ``power`` explicitly whenever ``family="tweedie"`` instead of
    letting glum's default apply unnoticed.
    """
    if family == "tweedie":
        if power is None:
            raise ValueError(
                "family='tweedie' requires an explicit power (typically "
                "1 < power < 2 for compound Poisson-Gamma pure-premium "
                "modeling: 1 is Poisson, 2 is Gamma). glum's own default "
                "(power=0, i.e. Normal) is almost never what you want for "
                "insurance data, so insurabench does not pass that default "
                "through silently -- pass e.g. power=1.5 explicitly."
            )
        return TweedieDistribution(power=power)
    if power is not None:
        raise ValueError(
            f"power is only meaningful for family='tweedie', got family={family!r}."
        )
    if family not in _FAMILY_STRINGS:
        raise ValueError(f"Unknown family {family!r}. Choose one of {list(_FAMILY_STRINGS) + ['tweedie']}.")
    return _FAMILY_STRINGS[family]


class GLMPricingModel(PricingModel):
    """Wraps ``glum.GeneralizedLinearRegressor`` behind the
    ``insurabench.models.base.PricingModel`` contract.

    Parameters
    ----------
    family:
        One of "poisson", "gamma", "tweedie", "negative_binomial", "normal".
    power:
        Required (and only meaningful) when ``family="tweedie"`` -- see
        ``_resolve_family``.
    link:
        Passed through to glum; "log" is standard for insurance rating
        (it makes fitted relativities multiplicative).
    alpha, l1_ratio:
        Elastic-net regularization strength / L1-L2 mix, passed through.
    fit_intercept:
        Whether to fit an intercept. When True, categorical columns are
        encoded with their first level dropped (glum's ``drop_first``) to
        keep the design matrix full-rank -- an intercept plus every
        category's dummy is a classic silent-collinearity trap, and glum
        will otherwise fit anyway with an ill-conditioned, hard-to-interpret
        solution rather than raising.
    **glum_kwargs:
        Passed through to ``glum.GeneralizedLinearRegressor`` verbatim
        (e.g. ``max_iter``, ``P1``/``P2`` for custom penalty weighting).
    """

    def __init__(
        self,
        family: Family = "tweedie",
        *,
        power: float | None = None,
        link: str = "log",
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        **glum_kwargs,
    ) -> None:
        self.family = family
        self.power = power
        self.link = link
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.glum_kwargs = glum_kwargs

        self._estimator = GeneralizedLinearRegressor(
            family=_resolve_family(family, power),
            link=link,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            drop_first=fit_intercept,
            **glum_kwargs,
        )

        # Populated by fit_policy_frame; lets predict_policy_frame /
        # score_policy_frame rebuild the identical X/y/weight/offset shape
        # for a different (e.g. held-out) PolicyFrame without the caller
        # having to remember how this particular target was set up.
        self.target_: Target | None = None
        self.peril_: str | list[str] | None = None
        self.coverage_: str | list[str] | None = None

    # -- PricingModel contract ------------------------------------------------

    def fit(self, X, y, *, sample_weight=None, offset=None) -> GLMPricingModel:
        self._estimator.fit(X, y, sample_weight=sample_weight, offset=offset)
        return self

    def predict(self, X, *, offset=None) -> np.ndarray:
        return self._estimator.predict(X, offset=offset)

    def relativities(self) -> pd.DataFrame:
        """Fitted coefficients, exponentiated into multiplicative
        relativities when ``link="log"``. See the module-level note on
        ``PricingModel.relativities`` for how this differs from the
        curves-layer relativity table."""
        names = list(self._estimator.feature_names_)
        coefs = np.asarray(self._estimator.coef_)
        rows = []
        if self.fit_intercept:
            rows.append({"feature": "(intercept)", "coefficient": float(self._estimator.intercept_)})
        rows.extend(
            {"feature": name, "coefficient": float(coef)} for name, coef in zip(names, coefs)
        )
        out = pd.DataFrame(rows)
        if self.link == "log":
            out["relativity"] = np.exp(out["coefficient"])
        return out

    def score(self, X, y, *, sample_weight=None, offset=None) -> float:
        """Deviance-based D² (glum's built-in pseudo-R²): 1 for a perfect
        fit, 0 for a model no better than the weighted mean. A quick sanity
        check that the fit generalizes to ``X``/``y`` -- not the design
        brief's evaluation layer (lift/Gini/calibration are build order
        step 3, not built yet)."""
        return self._estimator.score(X, y, sample_weight=sample_weight, offset=offset)

    # -- PolicyFrame convenience ------------------------------------------------

    def fit_policy_frame(
        self,
        pf: PolicyFrame,
        target: Target,
        *,
        peril: str | list[str] | None = None,
        coverage: str | list[str] | None = None,
    ) -> GLMPricingModel:
        """Build X/y/weight-or-offset from a PolicyFrame's views and fit.

        target:
          - ``"frequency"`` -- ``claim_count`` ~ features, with
            ``offset=log(exposure)``. Use ``family="poisson"`` (or
            ``"negative_binomial"``).
          - ``"severity"`` -- ``claim_amount`` ~ features, one row per
            claim. Use ``family="gamma"``.
          - ``"pure_premium"`` -- total claim amount per unit exposure ~
            features, with ``sample_weight=exposure``. The direct Tweedie
            compound Poisson-Gamma route (design brief §4's alternative to
            composing separate frequency/severity models) -- use
            ``family="tweedie"`` with ``1 < power < 2``.

        ``peril``/``coverage`` are forwarded to the underlying view (see
        ``PolicyFrame.frequency_view``/``severity_view``/``pure_premium_view``)
        to fit a peril- or coverage-specific model.
        """
        X, y, sample_weight, offset = build_design(pf, target, peril=peril, coverage=coverage)
        self.target_ = target
        self.peril_ = peril
        self.coverage_ = coverage
        return self.fit(X, y, sample_weight=sample_weight, offset=offset)

    def predict_policy_frame(self, pf: PolicyFrame) -> np.ndarray:
        """Predict on a (typically held-out/unseen) PolicyFrame, using the
        same target/peril/coverage setup established by
        ``fit_policy_frame``. For genuinely prospective scoring of new
        business with no claims history at all, build X directly (e.g. via
        ``insurabench.models.base.prepare_features`` on a policies-only
        frame) and call ``.predict()`` instead -- this method's per-target
        views assume a claims table is present, which is right for
        held-out evaluation but not for quoting brand-new risks.
        """
        if self.target_ is None:
            raise RuntimeError("Call fit_policy_frame before predict_policy_frame.")
        X, _, _, offset = build_design(
            pf, self.target_, peril=self.peril_, coverage=self.coverage_, for_predict=True
        )
        return self.predict(X, offset=offset)

    def score_policy_frame(self, pf: PolicyFrame) -> float:
        """Deviance-based D² (see ``score``) on a (typically held-out)
        PolicyFrame, using the target/peril/coverage set by
        ``fit_policy_frame``."""
        if self.target_ is None:
            raise RuntimeError("Call fit_policy_frame before score_policy_frame.")
        X, y, sample_weight, offset = build_design(
            pf, self.target_, peril=self.peril_, coverage=self.coverage_
        )
        return self.score(X, y, sample_weight=sample_weight, offset=offset)

