"""Rate-change / disruption impact of switching from one model's
predictions to another's, broken down by rating factor.

This is deliberately a different question from
``evaluation.double_lift.double_lift_chart``, and doesn't build on it:

- ``double_lift_chart`` asks a **model-selection** question -- on the
  rows where two models disagree most, which one tracks the *actual*
  outcome better? It requires held-out actual claims experience.
- This module asks a **repricing-impact** question -- if the book were
  switched from ``y_pred_old`` to ``y_pred_new`` today, which
  policyholders would see their rate move, and by how much? It needs no
  actual outcome at all, only the two models' predictions for the same
  rows -- the "was the new model right" question is completely separate
  from "who does the new model charge differently."

Both are standard, complementary parts of a real model-replacement
decision: double-lift for whether to switch, this module for what
switching actually does to the book. Neither is a substitute for the
other.
"""
from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import pandas as pd

from insurabench.curves._common import Target, build_target_frame
from insurabench.curves.one_way import _level_series
from insurabench.data.policy_frame import PolicyFrame


def _predictions(pf, target, y_pred_new, y_pred_old, peril, coverage):
    tf_new = build_target_frame(pf, target, y_pred_new, peril=peril, coverage=coverage)
    tf_old = build_target_frame(pf, target, y_pred_old, peril=peril, coverage=coverage)
    if not np.array_equal(tf_new.weight, tf_old.weight):
        raise ValueError(
            "y_pred_new and y_pred_old must be predictions for the exact same "
            "rows (same PolicyFrame/target/peril/coverage) -- their weight "
            "arrays don't match, which usually means one of them was computed "
            "against a different view."
        )
    return tf_new, tf_old


def rate_change_by_level(
    pf: PolicyFrame,
    feature: str,
    target: Target,
    y_pred_new,
    y_pred_old,
    *,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
    n_bins: int = 10,
    binning: Literal["quantile", "uniform"] = "quantile",
) -> pd.DataFrame:
    """Exposure-weighted rate change from ``y_pred_old`` to ``y_pred_new``,
    by level of ``feature`` -- the segment-level view of a repricing
    exercise: which groups of policyholders (by region, age band, vehicle
    type, ...) see the rate move, and which direction.

    Returns
    -------
    DataFrame, one row per level: ``level``, ``n_obs``, ``exposure``,
    ``old_rate``, ``new_rate``, ``pct_change`` (the *aggregate*
    exposure-weighted change for the level -- ``new_rate / old_rate -
    1`` -- not the average of each row's own individual pct change, so a
    level with rows moving in both directions isn't misleadingly netted
    to look unchanged at the individual level: see
    ``pct_exposure_increasing``/``pct_exposure_decreasing`` for that),
    ``pct_exposure_increasing`` and ``pct_exposure_decreasing`` (share of
    this level's exposure whose own *row-level* predicted rate went up /
    down, regardless of the level's aggregate direction -- these two need
    not sum to 1 if some rows are exactly unchanged).

    Notes
    -----
    A row where ``y_pred_old`` is exactly 0 makes that row's individual
    pct change undefined (division by zero); such rows are excluded from
    ``pct_exposure_increasing``/``pct_exposure_decreasing``'s denominator
    (their exposure isn't counted as "unchanged" either) but still
    contribute to ``old_rate``/``new_rate``/the level's aggregate
    ``pct_change``.
    """
    if feature not in pf.policy_schema.feature_roles:
        raise ValueError(
            f"{feature!r} is not a declared feature on this PolicyFrame's "
            f"PolicySchema (feature_roles: {list(pf.policy_schema.feature_roles)})."
        )

    tf_new, tf_old = _predictions(pf, target, y_pred_new, y_pred_old, peril, coverage)
    role = pf.policy_schema.feature_roles[feature]
    levels = _level_series(tf_new.view[feature], role, n_bins=n_bins, binning=binning)

    with np.errstate(divide="ignore", invalid="ignore"):
        row_pct_change = tf_new.predicted / tf_old.predicted - 1
    defined = np.isfinite(row_pct_change)

    frame = pd.DataFrame(
        {
            "level": levels,
            "weight": tf_new.weight,
            "old_numerator": tf_old.predicted_numerator,
            "new_numerator": tf_new.predicted_numerator,
            "increasing_weight": np.where(defined & (row_pct_change > 0), tf_new.weight, 0.0),
            "decreasing_weight": np.where(defined & (row_pct_change < 0), tf_new.weight, 0.0),
            "defined_weight": np.where(defined, tf_new.weight, 0.0),
        }
    )

    rows = []
    for level, g in frame.groupby("level", observed=True, sort=True):
        weight_sum = float(g["weight"].sum())
        old_rate = float(g["old_numerator"].sum() / weight_sum)
        new_rate = float(g["new_numerator"].sum() / weight_sum)
        defined_weight_sum = float(g["defined_weight"].sum())
        rows.append(
            {
                "level": str(level),
                "n_obs": len(g),
                "exposure": weight_sum,
                "old_rate": old_rate,
                "new_rate": new_rate,
                "pct_change": new_rate / old_rate - 1 if old_rate != 0 else float("nan"),
                "pct_exposure_increasing": (
                    float(g["increasing_weight"].sum() / defined_weight_sum)
                    if defined_weight_sum > 0
                    else float("nan")
                ),
                "pct_exposure_decreasing": (
                    float(g["decreasing_weight"].sum() / defined_weight_sum)
                    if defined_weight_sum > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def rate_change_distribution(
    pf: PolicyFrame,
    target: Target,
    y_pred_new,
    y_pred_old,
    *,
    bands: tuple[float, ...] = (-1.0, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, np.inf),
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> pd.DataFrame:
    """The overall shape of the book's disruption -- what share of
    exposure falls into each row-level pct-change band -- independent of
    any rating factor. The single most common number asked for in a
    rate-change sign-off: "what percentage of the book sees more than a
    10% increase?"

    Parameters
    ----------
    bands:
        Ascending bin edges for row-level ``(new/old - 1)`` pct change,
        e.g. the default ``-0.20`` edge means the band labeled
        ``"(-20%, -10%]"`` catches rows whose rate fell by between 10%
        and 20%. ``-1.0`` and ``inf`` as the outer edges catch every
        possible decrease (down to -100%, a rate going to 0) and any
        increase beyond the last named band.

    Returns
    -------
    DataFrame, one row per band, in ascending order: ``band``, ``n_obs``,
    ``exposure``, ``exposure_share`` (of the total, across all bands
    including any rows with an undefined pct change -- see Notes).

    Notes
    -----
    Rows where ``y_pred_old`` is exactly 0 (pct change undefined) are
    reported in their own ``"undefined (old rate = 0)"`` band rather than
    silently dropped, so ``exposure_share`` always sums to 1.0 across the
    returned table.
    """
    tf_new, tf_old = _predictions(pf, target, y_pred_new, y_pred_old, peril, coverage)

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = tf_new.predicted / tf_old.predicted - 1
    defined = np.isfinite(pct_change)

    total_weight = float(tf_new.weight.sum())
    rows = []

    labels = [
        f"({_fmt_pct(lo)}, {_fmt_pct(hi)}]" for lo, hi in itertools.pairwise(bands)
    ]
    band_index = np.searchsorted(bands, pct_change, side="right") - 1
    band_index = np.clip(band_index, 0, len(labels) - 1)

    for i, label in enumerate(labels):
        mask = defined & (band_index == i)
        weight_sum = float(tf_new.weight[mask].sum())
        rows.append(
            {
                "band": label,
                "n_obs": int(mask.sum()),
                "exposure": weight_sum,
                "exposure_share": weight_sum / total_weight if total_weight > 0 else float("nan"),
            }
        )

    undefined_mask = ~defined
    if undefined_mask.any():
        weight_sum = float(tf_new.weight[undefined_mask].sum())
        rows.append(
            {
                "band": "undefined (old rate = 0)",
                "n_obs": int(undefined_mask.sum()),
                "exposure": weight_sum,
                "exposure_share": weight_sum / total_weight if total_weight > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _fmt_pct(x: float) -> str:
    if np.isinf(x):
        return "+inf" if x > 0 else "-100%"
    return f"{x * 100:+.0f}%"
