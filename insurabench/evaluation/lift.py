"""Single-model lift chart (design brief §6).

Fixes, and enforces structurally, the one convention design brief §6
calls out as varying silently project-to-project: deciles are formed by
cumulative *exposure* share after sorting by predicted value -- not by
policy count, and not by naively quantiling the predicted-value
distribution itself (which can produce wildly uneven-exposure buckets
whenever predictions are skewed).
"""
from __future__ import annotations

import pandas as pd

from insurabench.curves._common import Target, build_target_frame, exposure_deciles
from insurabench.data.policy_frame import PolicyFrame


def lift_chart(
    pf: PolicyFrame,
    target: Target,
    y_pred,
    *,
    n_deciles: int = 10,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> pd.DataFrame:
    """Exposure-weighted actual vs. predicted by predicted-value decile.

    Rows are sorted ascending by the predicted rate, then assigned to one
    of ``n_deciles`` buckets by *cumulative exposure share* -- each bucket
    holds ~1/``n_deciles`` of total exposure (earned exposure for
    frequency/pure_premium, claim count for severity), not ~1/``n_deciles``
    of the row count and not an equal-width split of the predicted values
    themselves. This is the one documented convention design brief §6
    requires be fixed and enforced, not left to vary by call site.

    Parameters
    ----------
    target, y_pred:
        See ``insurabench.curves._common.build_target_frame``.
    n_deciles:
        Number of buckets (the name "decile" assumes the conventional 10;
        pass a different value for finer/coarser buckets).

    Returns
    -------
    DataFrame, one row per bucket (``decile`` numbered 1..``n_deciles``,
    ascending predicted value): ``n_obs``, ``exposure``, ``actual``,
    ``predicted``.

    Notes
    -----
    Ties in the predicted value (e.g. many rows sharing the same
    categorical-only prediction) are broken by original row order (a
    stable sort), so a run of tied rows can span a bucket boundary rather
    than all landing in the same bucket -- expected, and part of why
    buckets are exposure-equal rather than row-count-equal.
    """
    if n_deciles < 1:
        raise ValueError(f"n_deciles must be >= 1, got {n_deciles!r}.")

    tf = build_target_frame(pf, target, y_pred, peril=peril, coverage=coverage)
    bucket = exposure_deciles(tf.weight, tf.predicted, n_deciles)

    frame = pd.DataFrame(
        {
            "decile": bucket,
            "numerator": tf.numerator,
            "predicted_numerator": tf.predicted_numerator,
            "weight": tf.weight,
        }
    )
    grouped = (
        frame.groupby("decile")
        .agg(
            n_obs=("weight", "size"),
            exposure=("weight", "sum"),
            actual_numerator=("numerator", "sum"),
            predicted_total=("predicted_numerator", "sum"),
        )
        .reset_index()
    )
    grouped["actual"] = grouped["actual_numerator"] / grouped["exposure"]
    grouped["predicted"] = grouped["predicted_total"] / grouped["exposure"]
    return grouped[["decile", "n_obs", "exposure", "actual", "predicted"]]
