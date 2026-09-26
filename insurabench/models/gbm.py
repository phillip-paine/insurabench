"""GBM pricing model -- a CatBoost wrapper, including the zero-inflated
Poisson boosted tree from So (2024) for frequency (design brief §4).

CatBoost is the specific engine chosen here, not LightGBM/XGBoost, on
direct evidence rather than by default: So, B. (2024), "Enhanced Gradient
Boosting for Zero-Inflated Insurance Claims and Comparative Analysis of
CatBoost, XGBoost, and LightGBM", Scandinavian Actuarial Journal, 2024(10)
(arXiv:2307.07771) benchmarks all three on two auto claim frequency
datasets (French MTPL, a synthetic telematics set) and finds CatBoost
consistently best -- its "Ordered Target Statistic" categorical encoding
in particular suits the high-cardinality categorical rating factors
(region, vehicle brand) this kind of data typically has.

``family="poisson"`` is the standard native-objective baseline (the
"commodity" GBM option the design brief describes -- see §1/§4: the
model layer isn't this library's differentiator). ``family="zip"`` is
this module's actual point: the paper's ZIPB1 model, which reparameterizes
the zero-inflated Poisson's inflation probability p as a function of the
Poisson mean mu (a single tree ensemble, one fixed hyperparameter gamma)
rather than training two separate models for p and mu independently
(the paper's other variant, ZIPB2 -- not implemented here; see the
"Scope" note below). So (2024) itself recommends ZIPB1 specifically when
the goal is feature interpretation over squeezing out the last bit of
accuracy, which is exactly this library's positioning.

Scope: ZIPB2 (p and mu modeled independently, trained by alternating
coordinate descent across two tree ensembles -- the paper's Algorithm 2)
is NOT implemented. It requires a genuinely different training loop (two
CatBoost models updated in alternating cycles, each holding the other's
current score fixed), not just a different loss function passed to one
`.fit()` call the way ZIPB1 is -- a materially larger engineering task
than this module's other additions, and one the paper's own conclusion
frames as the *less* interpretable of the two options. If ZIPB2 is ever
wanted, it's a new addition alongside this one, not a small extension of
it.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from insurabench.data.policy_frame import PolicyFrame
from insurabench.models.base import (
    PricingModel,
    Target,
    _poisson_unit_deviance,
    _tweedie_unit_deviance,
    build_design,
)

Family = Literal["poisson", "zip", "gamma", "tweedie"]

_FREQUENCY_FAMILIES = {"poisson", "zip"}
_SEVERITY_FAMILIES = {"gamma"}
_PURE_PREMIUM_FAMILIES = {"tweedie"}


class _ZIPB1Objective:
    """CatBoost custom loss object for the ZIPB1 zero-inflated Poisson
    boosted tree (So (2024), eqs. 13-19): ln(mu) = ln(w) + F_T(x) (w =
    exposure, supplied to CatBoost as ``baseline`` at both fit and
    predict time -- CatBoost has no separate "offset" concept, baseline
    is its equivalent), p = 1 / (1 + mu**gamma), gamma a fixed positive
    hyperparameter.

    ``calc_ders_range`` returns CatBoost's expected (der1, der2) --
    the *negative* of the loss's own gradient/hessian w.r.t. the raw
    score, matching CatBoost's own convention (verified directly against
    CatBoost's built-in Poisson loss: a hand-derived plain-Poisson custom
    objective using this same sign convention reproduces CatBoost's
    native Poisson predictions to within floating-point error). The
    gradient and hessian formulas themselves (eqs. 18-19) were verified
    by comparing them to a central finite-difference derivative of the
    loss (eq. 17) directly -- max error ~2e-5 (gradient) / ~6e-4
    (hessian) across a wide (y, raw score, gamma) sweep at the
    finite-difference step size that minimizes truncation-vs-cancellation
    error, consistent with correct analytic formulas rather than a
    transcription error.
    """

    def __init__(self, gamma: float):
        if gamma <= 0:
            raise ValueError(
                f"gamma must be > 0 -- So (2024) restricts it to keep the inflation "
                f"probability p decreasing in mu (eq. 16), which is the only sensible "
                f"direction for claim data (higher expected frequency should not imply "
                f"*more* excess zeros). Got gamma={gamma!r}."
            )
        self.gamma = gamma

    def calc_ders_range(self, approxes, targets, weights):
        gamma = self.gamma
        approx = np.asarray(approxes, dtype=float)
        y = np.asarray(targets, dtype=float)
        w = np.ones_like(approx) if weights is None else np.asarray(weights, dtype=float)

        # mu can only get numerically dangerous (overflow in mu**gamma or
        # underflow in exp(-mu)) transiently, early in boosting before
        # the ensemble has learned a sane scale -- clip defensively; this
        # guard has no counterpart in the paper's idealized math, it's a
        # practical numerical-stability necessity for the optimizer.
        mu = np.clip(np.exp(approx), 1e-10, 1e10)
        mu_g = mu**gamma

        g = np.empty_like(mu)
        h = np.empty_like(mu)

        is_zero = y == 0

        mu0 = mu[is_zero]
        mu_g0 = mu_g[is_zero]
        e_neg_mu0 = np.exp(-mu0)
        denom0 = 1 + mu_g0 * e_neg_mu0
        g[is_zero] = (mu_g0 * e_neg_mu0 * (mu0 - gamma)) / denom0 + (gamma * mu_g0) / (1 + mu_g0)
        h[is_zero] = (
            mu_g0 * e_neg_mu0 * (mu0 - (gamma - mu0) ** 2) * denom0
            + (mu_g0 * e_neg_mu0 * (gamma - mu0)) ** 2
        ) / denom0**2 + (gamma**2 * mu_g0) / (1 + mu_g0) ** 2

        pos = ~is_zero
        mu_p = mu[pos]
        mu_g_p = mu_g[pos]
        g[pos] = (gamma * mu_g_p) / (1 + mu_g_p) + mu_p - gamma - y[pos]
        h[pos] = (gamma**2 * mu_g_p) / (1 + mu_g_p) ** 2 + mu_p

        der1 = -g * w
        der2 = -h * w
        return list(zip(der1.tolist(), der2.tolist(), strict=True))


def _zip_relativity(mu: np.ndarray, gamma: float) -> np.ndarray:
    """p = 1 / (1 + mu**gamma) -- eq. 16, the ZIPB1 model's implied
    excess-zero probability at a given fitted mu. Exposed for diagnostics
    (e.g. plotting p against mu), not part of the ``PricingModel``
    contract.
    """
    return 1.0 / (1.0 + np.clip(mu, 1e-10, 1e10) ** gamma)


class _GammaObjective:
    """CatBoost custom loss for a Gamma GLM-equivalent (log link): the
    standard ``y/mu + ln(mu)`` Gamma deviance-consistent loss (the same
    one XGBoost's built-in ``reg:gamma`` and LightGBM's ``gamma``
    objective use). Needed because CatBoost's *native* Tweedie loss
    strictly requires ``1 < variance_power < 2`` and raises on
    ``variance_power=2`` (confirmed directly against the installed
    CatBoost version -- Tweedie power=2 is mathematically the Gamma
    distribution, but CatBoost's own implementation doesn't accept it at
    that boundary), so severity can't reuse the ``tweedie`` family the
    way ``GLMPricingModel`` reuses one Tweedie family for both severity
    and pure premium.

    Gradient/hessian verified against a central finite-difference
    derivative of the loss directly (max error ~2e-6 / ~3e-5 across a
    wide (y, raw score) sweep), same verification approach as
    ``_ZIPB1Objective``.
    """

    def calc_ders_range(self, approxes, targets, weights):
        approx = np.asarray(approxes, dtype=float)
        y = np.asarray(targets, dtype=float)
        w = np.ones_like(approx) if weights is None else np.asarray(weights, dtype=float)
        mu = np.clip(np.exp(approx), 1e-10, 1e10)
        g = 1.0 - y / mu
        h = y / mu
        der1 = -g * w
        der2 = -h * w
        return list(zip(der1.tolist(), der2.tolist(), strict=True))


def _resolve_loss(family: Family, power: float | None, zip_gamma: float | None):
    """Returns (loss_function, eval_metric, uses_custom_objective) for
    ``CatBoostRegressor``. Mirrors ``models.glm._resolve_family``'s
    footgun-guard philosophy: a family that needs an extra parameter
    (``power`` for tweedie, ``zip_gamma`` for zip) must get it explicitly,
    never silently defaulted.
    """
    if family == "poisson":
        if power is not None or zip_gamma is not None:
            raise ValueError("power/zip_gamma are not meaningful for family='poisson'.")
        return "Poisson", None, False
    if family == "zip":
        if power is not None:
            raise ValueError("power is not meaningful for family='zip' (use zip_gamma instead).")
        if zip_gamma is None:
            raise ValueError(
                "family='zip' requires an explicit zip_gamma (So (2024) selects it by grid "
                "search over the training data; there is no universally-reasonable default). "
                "Try zip_gamma=1.0 as a starting point and tune from there."
            )
        return _ZIPB1Objective(zip_gamma), "Poisson", True
    if family == "gamma":
        if power is not None or zip_gamma is not None:
            raise ValueError("power/zip_gamma are not meaningful for family='gamma'.")
        return _GammaObjective(), "RMSE", True
    if family == "tweedie":
        if zip_gamma is not None:
            raise ValueError("zip_gamma is not meaningful for family='tweedie' (use power instead).")
        if power is None:
            raise ValueError(
                "family='tweedie' requires an explicit power, 1 < power < 2, for compound "
                "Poisson-Gamma pure_premium modeling (matching GLMPricingModel's own Tweedie "
                "footgun-guard). Note power=2 (Gamma) is NOT accepted here -- CatBoost's native "
                "Tweedie loss itself rejects variance_power=2; use family='gamma' for severity "
                "instead, which is a separate custom objective built for exactly that case."
            )
        if not (1 < power < 2):
            raise ValueError(
                f"CatBoost's native Tweedie loss requires 1 < power < 2, got power={power!r}. "
                f"Use family='gamma' (power=2 case) or family='poisson'/'zip' (power=1 case) instead."
            )
        return f"Tweedie:variance_power={power}", None, False
    raise ValueError(f"Unknown family: {family!r}. Choose 'poisson', 'zip', 'gamma', or 'tweedie'.")


def _cat_feature_names(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def _zip_unit_deviance(y: np.ndarray, mu: np.ndarray, p: np.ndarray) -> np.ndarray:
    """So (2024) §4.1.1's zero-inflated Poisson unit deviance. ``p`` may
    be a scalar (the null/base-rate model, p_bar=0.5 per the paper) or an
    array (a fitted model's per-row p).
    """
    mu = np.clip(mu, 1e-10, None)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    is_zero = y == 0
    out = np.empty_like(mu, dtype=float)
    p_zero = p[is_zero] if np.ndim(p) > 0 else p
    p_pos = p[~is_zero] if np.ndim(p) > 0 else p
    out[is_zero] = -2.0 * np.log(p_zero + (1 - p_zero) * np.exp(-mu[is_zero]))
    y_pos = y[~is_zero]
    mu_pos = mu[~is_zero]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[~is_zero] = 2.0 * (
            y_pos * np.log(y_pos) - y_pos - np.log(1 - p_pos) - y_pos * np.log(mu_pos) + mu_pos
        )
    return out


class GBMPricingModel(PricingModel):
    """Wraps ``catboost.CatBoostRegressor`` behind the
    ``insurabench.models.base.PricingModel`` contract -- the GBM
    counterpart to ``GLMPricingModel``, built to the same interface so
    curves/evaluation code written against one works unchanged against
    the other (design brief §4's model-agnosticism requirement).

    Parameters
    ----------
    family:
        - ``"poisson"`` -- CatBoost's native Poisson loss. Standard
          baseline; use for ``target="frequency"``.
        - ``"zip"`` -- the ZIPB1 zero-inflated Poisson boosted tree (see
          module docstring). Use for ``target="frequency"``; requires
          ``zip_gamma``.
        - ``"gamma"`` -- a custom Gamma GLM-equivalent loss (log link).
          Use for ``target="severity"``. CatBoost has no native Gamma
          loss, and its native Tweedie loss rejects ``variance_power=2``
          (the Gamma case) outright -- confirmed directly, not assumed --
          so this is its own custom objective rather than a Tweedie call.
        - ``"tweedie"`` -- CatBoost's native Tweedie loss, requiring
          ``1 < power < 2``. Use for ``target="pure_premium"`` (the
          compound Poisson-Gamma case), matching ``GLMPricingModel``'s
          Tweedie convention. NOT valid for severity -- use ``"gamma"``.
    power:
        Required (and only meaningful) for ``family="tweedie"``, and
        must satisfy ``1 < power < 2`` (CatBoost's own native Tweedie
        loss enforces this range; it is not an insurabench restriction).
    zip_gamma:
        Required (and only meaningful) for ``family="zip"`` -- the fixed
        hyperparameter controlling how the inflation probability p
        relates to the fitted mean mu (eq. 16); So (2024) selects it by
        grid search, there's no universal default.
    monotone_constraints:
        Optional ``{feature_name: 1 | -1 | 0}`` passed through to
        CatBoost -- a GBM-specific capability the GLM wrapper has no
        equivalent of (a GLM's monotonicity is whatever sign its fitted
        coefficient comes out to; a GBM's isn't guaranteed monotonic in
        any feature unless told to be).
    **catboost_kwargs:
        Passed through to ``CatBoostRegressor`` verbatim (e.g.
        ``iterations``, ``learning_rate``, ``depth``, ``l2_leaf_reg``).
        ``verbose`` defaults to ``False`` here (CatBoost's own default is
        noisy per-iteration logging) unless overridden.
    """

    def __init__(
        self,
        family: Family = "poisson",
        *,
        power: float | None = None,
        zip_gamma: float | None = None,
        monotone_constraints: dict[str, int] | None = None,
        **catboost_kwargs,
    ) -> None:
        self.family = family
        self.power = power
        self.zip_gamma = zip_gamma
        self.monotone_constraints = monotone_constraints
        self.catboost_kwargs = catboost_kwargs

        loss_function, eval_metric, uses_custom_objective = _resolve_loss(family, power, zip_gamma)
        self._uses_custom_objective = uses_custom_objective

        catboost_kwargs = {"verbose": False, **catboost_kwargs}
        if monotone_constraints is not None:
            catboost_kwargs["monotone_constraints"] = monotone_constraints
        if eval_metric is not None:
            catboost_kwargs["eval_metric"] = eval_metric

        self._estimator = CatBoostRegressor(loss_function=loss_function, **catboost_kwargs)

        self._cat_features: list[str] | None = None
        self._auto_baseline_: float | None = None
        self.target_: Target | None = None
        self.peril_: str | list[str] | None = None
        self.coverage_: str | list[str] | None = None

    # -- PricingModel contract ------------------------------------------------

    def fit(self, X: pd.DataFrame, y, *, sample_weight=None, offset=None) -> GBMPricingModel:
        cat_features = _cat_feature_names(X)
        if self._uses_custom_objective and offset is None:
            # A custom objective with no natural offset (e.g. family="gamma",
            # which -- unlike frequency -- has no exposure-like term to
            # supply) starts every row's raw score at exactly 0 (mu=1).
            # Confirmed directly (not assumed) that this can leave the
            # model badly under-converged well within a plausible
            # iteration budget whenever the target's true scale is far
            # from 1 (e.g. severity in the thousands): the trees then
            # have to learn the entire base level from scratch through
            # many small learning-rate steps rather than starting near
            # the right place. A GLM gets an equivalent for free from its
            # exact intercept; boosting has none, so this supplies the
            # same role with one constant: log of the (weight-averaged)
            # target's mean, used as every row's starting baseline.
            y_arr = np.asarray(y, dtype=float)
            w_arr = np.ones_like(y_arr) if sample_weight is None else np.asarray(sample_weight, dtype=float)
            mean_y = np.clip(np.average(y_arr, weights=w_arr), 1e-10, None)
            self._auto_baseline_: float | None = float(np.log(mean_y))
            offset = np.full(len(y_arr), self._auto_baseline_)
        else:
            self._auto_baseline_ = None
        pool = Pool(X, y, cat_features=cat_features, weight=sample_weight, baseline=offset)
        self._estimator.fit(pool)
        self._cat_features = cat_features
        return self

    def predict(self, X: pd.DataFrame, *, offset=None) -> np.ndarray:
        if self._cat_features is None:
            raise RuntimeError("Call fit before predict.")
        if offset is None and self._auto_baseline_ is not None:
            offset = np.full(len(X), self._auto_baseline_)
        pool = Pool(X, cat_features=self._cat_features, baseline=offset)
        if self._uses_custom_objective:
            raw = self._estimator.predict(pool, prediction_type="RawFormulaVal")
            return np.exp(raw)
        return self._estimator.predict(pool)

    def relativities(self) -> pd.DataFrame:
        """CatBoost's ``PredictionValuesChange`` feature importances,
        normalized to sum to 100 (CatBoost's own convention, and the same
        formula So (2024) describes in its "Interpretation" section).

        NOT a multiplicative relativity the way ``GLMPricingModel``'s is
        -- a GBM has no per-level coefficient to exponentiate. This is a
        ranking of which features move the prediction the most, nothing
        about direction or magnitude at a specific level. See the
        ``PricingModel.relativities`` docstring for why this differs by
        model type and is not the curves-layer relativity table.
        """
        importances = self._estimator.get_feature_importance()
        names = self._estimator.feature_names_
        out = pd.DataFrame({"feature": names, "importance": importances})
        return out.sort_values("importance", ascending=False, ignore_index=True)

    def score(self, X: pd.DataFrame, y, *, sample_weight=None, offset=None) -> float:
        """Deviance-based D² (1 for a perfect fit, 0 for a model no
        better than the weighted mean) -- same semantics and same
        interpretation as ``GLMPricingModel.score``, computed here from
        the unit deviance matching this model's own family (Poisson
        deviance for ``family="poisson"``, the ZIP deviance from So
        (2024) §4.1.1 for ``family="zip"``, Gamma deviance for
        ``family="gamma"``, standard Tweedie deviance for
        ``family="tweedie"``) rather than CatBoost's own generic
        ``.score()`` (which is a plain R², not appropriate for any of
        these families).
        """
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        mu = self.predict(X, offset=offset)

        if self.family == "poisson":
            unit_dev = _poisson_unit_deviance(y, mu)
            null_mu = np.full_like(y, np.average(y, weights=w))
            null_dev = _poisson_unit_deviance(y, null_mu)
        elif self.family == "zip":
            p = _zip_relativity(mu, self.zip_gamma)
            unit_dev = _zip_unit_deviance(y, mu, p)
            null_mu = np.full_like(y, np.average(y, weights=w))
            null_dev = _zip_unit_deviance(y, null_mu, 0.5)  # p_bar=0.5, per the paper
        elif self.family == "gamma":
            unit_dev = _tweedie_unit_deviance(y, mu, 2.0)
            null_mu = np.full_like(y, np.average(y, weights=w))
            null_dev = _tweedie_unit_deviance(y, null_mu, 2.0)
        else:  # tweedie
            unit_dev = _tweedie_unit_deviance(y, mu, self.power)
            null_mu = np.full_like(y, np.average(y, weights=w))
            null_dev = _tweedie_unit_deviance(y, null_mu, self.power)

        mean_dev = np.average(unit_dev, weights=w)
        mean_null_dev = np.average(null_dev, weights=w)
        if mean_null_dev == 0:
            return float("nan")
        return float(1.0 - mean_dev / mean_null_dev)

    # -- PolicyFrame convenience ------------------------------------------------

    def fit_policy_frame(
        self,
        pf: PolicyFrame,
        target: Target,
        *,
        peril: str | list[str] | None = None,
        coverage: str | list[str] | None = None,
    ) -> GBMPricingModel:
        """Build X/y/weight-or-offset from a PolicyFrame's views and fit.
        See ``GLMPricingModel.fit_policy_frame`` -- identical target
        semantics, same ``build_design`` helper underneath.
        """
        if target == "frequency" and self.family not in _FREQUENCY_FAMILIES:
            raise ValueError(
                f"target='frequency' needs family='poisson' or 'zip' (a count "
                f"distribution), got family={self.family!r}."
            )
        if target == "severity" and self.family not in _SEVERITY_FAMILIES:
            raise ValueError(
                f"target='severity' needs family='gamma', got family={self.family!r}."
            )
        if target == "pure_premium" and self.family not in _PURE_PREMIUM_FAMILIES:
            raise ValueError(
                f"target='pure_premium' needs family='tweedie', got family={self.family!r}."
            )

        X, y, sample_weight, offset = build_design(pf, target, peril=peril, coverage=coverage)
        self.target_ = target
        self.peril_ = peril
        self.coverage_ = coverage
        return self.fit(X, y, sample_weight=sample_weight, offset=offset)

    def predict_policy_frame(self, pf: PolicyFrame) -> np.ndarray:
        """See ``GLMPricingModel.predict_policy_frame`` -- same caveat
        about held-out evaluation vs. genuinely prospective new-business
        scoring applies here identically.
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
