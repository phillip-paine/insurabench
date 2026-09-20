"""Exposure-weighted partial dependence for offset/exposure models (design
brief §5).

Unlike every other curves/evaluation function in this package, a PDP
cannot be built from a precomputed ``y_pred`` array -- by definition it
re-predicts after perturbing one feature, holding every other row's
features at their observed values. So this module is the one place in
``insurabench.curves``/``insurabench.evaluation`` that takes a fitted
``PricingModel`` directly instead of a plain array, and is the one place
that isn't purely model-agnostic in the "never touches a model" sense the
rest of the package uses -- it *is* model-agnostic in the sense that
matters for design brief §4/§5: it drives any ``PricingModel`` (GLM, GBM,
...) through the same ``fit``/``predict`` contract, so it works
identically whichever model type is passed.

A plain ``sklearn``-style PDP averages raw predictions across the
perturbed dataset. That is wrong here for two reasons this module fixes:

1. It would average unweighted by exposure, letting a handful of
   high-exposure policy-periods count the same as a single
   short-term one.
2. For an offset model (frequency: ``offset=log(exposure)``), the raw
   prediction is already exposure-scaled (a *count*, not a *rate*) --
   averaging counts directly would just reproduce each row's own
   exposure back at you rather than showing how the *rate* responds to
   the feature. This module always averages on the rate scale (dividing
   back out by exposure/weight) so the offset is handled correctly
   regardless of grid value.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from insurabench.curves.one_way import _level_series
from insurabench.data.policy_frame import PolicyFrame
from insurabench.data.schema import FeatureRole
from insurabench.models.base import PricingModel, prepare_features

Target = Literal["frequency", "severity", "pure_premium"]


def _grid_values(values: pd.Series, role: FeatureRole, *, n_bins: int, binning: str) -> list:
    """Distinct grid points to evaluate the PDP at: every observed level
    for a categorical feature, or ``n_bins`` representative values (bin
    midpoints/medians) for a continuous one -- reusing the exact same
    binning ``one_way_curve`` uses, so a PDP and a one-way curve for the
    same feature are drawn from directly comparable groups.
    """
    if role is FeatureRole.CATEGORICAL:
        return sorted(values.astype("category").cat.categories.tolist())
    binned = _level_series(values, role, n_bins=n_bins, binning=binning)
    # One representative value per bin: the bin's own median of the
    # actually-observed values that fell in it (not the bin's edge
    # midpoint), so a perturbed row always gets a value from its data's
    # own support rather than an interpolated one that may never occur.
    return [
        float(g.median())
        for _, g in values.groupby(binned, observed=True)
        if len(g) > 0
    ]


def _offset_for(target: Target, exposure: np.ndarray) -> np.ndarray | None:
    if target == "frequency":
        return np.log(exposure)
    return None


def partial_dependence(
    pf: PolicyFrame,
    feature: str,
    target: Target,
    model: PricingModel,
    *,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
    n_bins: int = 10,
    binning: Literal["quantile", "uniform"] = "quantile",
) -> pd.DataFrame:
    """Exposure-weighted partial dependence of ``target`` on ``feature``.

    For each grid value ``v`` of ``feature`` (every observed level, if
    categorical; ``n_bins`` representative values, if continuous), every
    row's own ``feature`` value is overwritten with ``v`` (all other
    features left at their observed values), the already-fitted ``model``
    re-predicts, and the exposure-weighted average predicted rate across
    all rows is reported at that grid value.

    Parameters
    ----------
    model:
        An already-fitted ``PricingModel`` (e.g. a ``GLMPricingModel``
        after ``fit_policy_frame``). Must accept the same feature columns
        ``model`` was fit on for ``target`` -- use the same
        ``peril``/``coverage`` here that the model was fit with.

    Returns
    -------
    DataFrame, one row per grid value, in ascending order (bin order for
    continuous, sorted-level order for categorical): ``level``,
    ``partial_dependence`` (the exposure-weighted mean predicted rate
    with ``feature`` fixed at that value across every row).

    Notes
    -----
    This is a marginal/perturbation PDP (Friedman's original definition),
    not an accumulated-local-effects (ALE) plot -- like any PDP, it can
    extrapolate into combinations of feature values that don't occur
    together in the data if ``feature`` is correlated with other declared
    features. That's a property of PDPs generally, not specific to this
    implementation.
    """
    if feature not in pf.policy_schema.feature_roles:
        raise ValueError(
            f"{feature!r} is not a declared feature on this PolicyFrame's "
            f"PolicySchema (feature_roles: {list(pf.policy_schema.feature_roles)})."
        )

    if target == "frequency":
        view = pf.frequency_view(peril=peril, coverage=coverage)
        weight = view[pf.policy_schema.exposure_col].to_numpy(dtype=float)
    elif target == "severity":
        view = pf.severity_view(peril=peril, coverage=coverage)
        weight = np.ones(len(view), dtype=float)
    elif target == "pure_premium":
        view = pf.pure_premium_view(peril=peril, coverage=coverage)
        weight = view[pf.policy_schema.exposure_col].to_numpy(dtype=float)
    else:
        raise ValueError(
            f"Unknown target: {target!r}. Choose 'frequency', 'severity', or 'pure_premium'."
        )

    role = pf.policy_schema.feature_roles[feature]
    grid = _grid_values(view[feature], role, n_bins=n_bins, binning=binning)
    offset = _offset_for(target, weight)

    rows = []
    for v in grid:
        perturbed = view.copy()
        perturbed[feature] = v
        X = prepare_features(perturbed, pf.policy_schema.feature_roles)
        y_pred = np.asarray(model.predict(X, offset=offset), dtype=float)
        # frequency's y_pred is already a count (offset handles the
        # exposure scaling); severity/pure_premium's is already a rate.
        # Only frequency needs dividing back out to the rate scale here.
        rate = y_pred / weight if target == "frequency" else y_pred
        rows.append(
            {
                "level": str(v),
                "partial_dependence": float(np.average(rate, weights=weight)),
            }
        )
    return pd.DataFrame(rows)
