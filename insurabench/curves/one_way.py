"""Exposure-weighted one-way relativity curves (design brief §5).

The primary differentiator this layer exists for: standardizes what every
existing insurance-modeling tutorial/repo currently hand-rolls -- actual
vs fitted by rating-factor level, with confidence bands on the observed
side -- into one function that behaves identically whichever
target/model produced ``y_pred`` (design brief §4's "critical
requirement": curves must be model-agnostic).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from insurabench.curves._common import Target, build_target_frame
from insurabench.data.policy_frame import PolicyFrame
from insurabench.data.schema import FeatureRole


def _z_score(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}.")
    return float(norm.ppf(0.5 + confidence / 2))


def _level_series(values: pd.Series, role: FeatureRole, *, n_bins: int, binning: str) -> pd.Series:
    if role is FeatureRole.CATEGORICAL:
        return values.astype("category")
    if binning == "quantile":
        return pd.qcut(values, q=n_bins, duplicates="drop")
    if binning == "uniform":
        return pd.cut(values, bins=n_bins)
    raise ValueError(f"Unknown binning {binning!r}. Choose 'quantile' or 'uniform'.")


def one_way_curve(
    pf: PolicyFrame,
    feature: str,
    target: Target,
    y_pred,
    *,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
    n_bins: int = 10,
    binning: Literal["quantile", "uniform"] = "quantile",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Exposure-weighted actual-vs-fitted for one rating factor.

    Parameters
    ----------
    feature:
        A column declared in ``pf.policy_schema.feature_roles``.
        Categorical features are grouped by level as-is; continuous
        features are binned into ``n_bins`` groups first
        (``binning="quantile"``, the default, bins by roughly equal *row
        count* -- not equal exposure; see Notes -- ``"uniform"`` gives
        equal-width bins instead).
    target, y_pred:
        See ``insurabench.curves._common.build_target_frame`` -- ``y_pred``
        is whatever ``model.predict_policy_frame(pf)`` (for *any*
        PricingModel: GLM, GBM, ...) returned for this target; this
        function never calls a model itself.

    Returns
    -------
    DataFrame, one row per level/bin, in level order (increasing bin edge
    for a continuous feature; by descending exposure for a categorical
    one):
        ``level``, ``n_obs``, ``exposure`` (sum of the target's weight --
        earned exposure for frequency/pure_premium, claim count for
        severity), ``observed``, ``observed_lower``, ``observed_upper``,
        ``fitted``.

    Notes
    -----
    Confidence bands are a normal approximation around the exposure-
    weighted mean, using the *reliability-weighted* variance of the
    per-row observed ratio (numerator/weight) within each bin, with
    effective sample size ``n_eff = sum(weight)**2 / sum(weight**2)``.
    This is a deliberately distribution-agnostic convention -- it doesn't
    assume Poisson/Gamma/Tweedie -- since the curve must look the same
    whether the underlying model is a GLM or a GBM. It's a reasonable
    approximation for frequency/pure_premium at typical bin sizes; for
    severity, which is heavy-tailed, treat the band as indicative rather
    than exact, especially in low-count bins.

    Quantile binning (the default) forms bins of equal *row* count, not
    equal exposure -- unlike ``insurabench.evaluation.lift.lift_chart``,
    which explicitly deciles by exposure (design brief §6's fixed
    convention for that chart). A one-way curve is read per-level rather
    than compared decile-to-decile, so equal-row bins here are the more
    standard convention and keep every bin's confidence band at a
    comparable, non-tiny sample size even when exposure is very unevenly
    distributed across a continuous factor.
    """
    if feature not in pf.policy_schema.feature_roles:
        raise ValueError(
            f"{feature!r} is not a declared feature on this PolicyFrame's "
            f"PolicySchema (feature_roles: {list(pf.policy_schema.feature_roles)})."
        )

    tf = build_target_frame(pf, target, y_pred, peril=peril, coverage=coverage)
    role = pf.policy_schema.feature_roles[feature]
    levels = _level_series(tf.view[feature], role, n_bins=n_bins, binning=binning)

    frame = pd.DataFrame(
        {
            "level": levels,
            "numerator": tf.numerator,
            "weight": tf.weight,
            "predicted_numerator": tf.predicted_numerator,
        }
    )
    frame["ratio"] = frame["numerator"] / frame["weight"]

    z = _z_score(confidence)
    rows = []
    for level, g in frame.groupby("level", observed=True, sort=True):
        w = g["weight"].to_numpy()
        weight_sum = float(w.sum())
        observed = float(g["numerator"].sum() / weight_sum)
        fitted = float(g["predicted_numerator"].sum() / weight_sum)
        sum_w_sq = float(np.sum(w**2))
        n_eff = (weight_sum**2) / sum_w_sq if sum_w_sq > 0 else len(g)
        weighted_var = float(np.average((g["ratio"].to_numpy() - observed) ** 2, weights=w))
        se = np.sqrt(weighted_var / n_eff) if n_eff > 0 else np.nan
        rows.append(
            {
                "level": str(level),
                "n_obs": len(g),
                "exposure": weight_sum,
                "observed": observed,
                "observed_lower": observed - z * se,
                "observed_upper": observed + z * se,
                "fitted": fitted,
            }
        )
    out = pd.DataFrame(rows)
    if role is FeatureRole.CATEGORICAL:
        out = out.sort_values("exposure", ascending=False).reset_index(drop=True)
    return out
