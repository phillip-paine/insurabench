"""Exposure-weighted two-way (interaction) relativity curves (design brief
§5).

The two-factor sibling of ``one_way_curve``: same numerator/weight
alignment via ``build_target_frame``, same model-agnostic contract (this
module never calls a model -- ``y_pred`` is whatever
``model.predict_policy_frame(pf)`` produced), but grouped by a pair of
rating factors instead of one. This is what surfaces interactions a
series of one-way curves cannot: two factors that each look well-fitted
individually can still combine into a cell the model badly misprices.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from insurabench.curves._common import Target, build_target_frame
from insurabench.curves.one_way import _level_series


def two_way_curve(
    pf,
    feature_x: str,
    feature_y: str,
    target: Target,
    y_pred,
    *,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
    n_bins: int = 5,
    binning: Literal["quantile", "uniform"] = "quantile",
) -> pd.DataFrame:
    """Exposure-weighted actual-vs-fitted for a pair of rating factors.

    Parameters
    ----------
    feature_x, feature_y:
        Two distinct columns declared in ``pf.policy_schema.feature_roles``.
        Each is binned independently exactly as ``one_way_curve`` bins a
        single feature (categorical -> by level; continuous -> ``n_bins``
        quantile or uniform bins), then the two binnings are crossed.
    target, y_pred, n_bins, binning:
        See ``one_way_curve`` / ``build_target_frame``.

    Returns
    -------
    DataFrame, one row per (level_x, level_y) cell that has at least one
    observation: ``level_x``, ``level_y``, ``n_obs``, ``exposure``,
    ``observed``, ``fitted``. Empty cells (a combination with zero rows)
    are omitted rather than filled with NaN -- pivot the result yourself
    (e.g. ``result.pivot(index="level_x", columns="level_y",
    values="fitted")``) if a dense grid for a heatmap is wanted; sparse
    cells are common with two continuous factors at typical bin counts
    and are more honestly left out than invented.

    Notes
    -----
    No confidence bands here (unlike ``one_way_curve``) -- cross-binning
    two factors thins out cell exposure quickly, and a per-cell normal
    approximation becomes unreliable well before a one-way curve's does.
    Read a two-way curve for *where* actual and fitted diverge, not for
    per-cell statistical significance.
    """
    if feature_x == feature_y:
        raise ValueError(f"feature_x and feature_y must differ, both got {feature_x!r}.")
    for feature in (feature_x, feature_y):
        if feature not in pf.policy_schema.feature_roles:
            raise ValueError(
                f"{feature!r} is not a declared feature on this PolicyFrame's "
                f"PolicySchema (feature_roles: {list(pf.policy_schema.feature_roles)})."
            )

    tf = build_target_frame(pf, target, y_pred, peril=peril, coverage=coverage)
    role_x = pf.policy_schema.feature_roles[feature_x]
    role_y = pf.policy_schema.feature_roles[feature_y]
    levels_x = _level_series(tf.view[feature_x], role_x, n_bins=n_bins, binning=binning)
    levels_y = _level_series(tf.view[feature_y], role_y, n_bins=n_bins, binning=binning)

    frame = pd.DataFrame(
        {
            "level_x": levels_x,
            "level_y": levels_y,
            "numerator": tf.numerator,
            "weight": tf.weight,
            "predicted_numerator": tf.predicted_numerator,
        }
    )

    rows = []
    for (level_x, level_y), g in frame.groupby(["level_x", "level_y"], observed=True, sort=True):
        weight_sum = float(g["weight"].sum())
        rows.append(
            {
                "level_x": str(level_x),
                "level_y": str(level_y),
                "n_obs": len(g),
                "exposure": weight_sum,
                "observed": float(g["numerator"].sum() / weight_sum),
                "fitted": float(g["predicted_numerator"].sum() / weight_sum),
            }
        )
    return pd.DataFrame(rows)
