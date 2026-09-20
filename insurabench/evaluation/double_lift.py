"""Direct model-vs-model comparison chart (design brief §6).

A single lift chart (``evaluation.lift.lift_chart``) checks one model
against itself. A double lift chart instead asks: on the rows where two
models disagree most about relative risk, which one actually tracks the
observed outcome? This primitive doesn't exist packaged anywhere
currently (design brief §6) even though it's the standard way actuaries
compare a candidate model against an incumbent before replacing it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurabench.curves._common import Target, build_target_frame, exposure_deciles
from insurabench.data.policy_frame import PolicyFrame


def double_lift_chart(
    pf: PolicyFrame,
    target: Target,
    y_pred_a,
    y_pred_b,
    *,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    n_deciles: int = 10,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> pd.DataFrame:
    """Exposure-weighted actual vs. both models' predictions, bucketed by
    where the two models disagree most.

    Rows are sorted ascending by the ratio ``y_pred_a / y_pred_b`` (both
    on the rate scale) -- the standard double-lift sort key, since it's
    exactly the quantity that's ~1 where the models agree and moves away
    from 1 where they diverge -- then assigned to one of ``n_deciles``
    buckets by cumulative exposure share (the same fixed convention as
    ``lift_chart``, via ``insurabench.curves._common.exposure_deciles``).

    Parameters
    ----------
    y_pred_a, y_pred_b:
        Two prediction arrays for the *same* ``target``/``peril``/
        ``coverage`` view of ``pf`` -- e.g. a GLM's and a GBM's
        predictions for the same held-out data (design brief §4's
        model-agnosticism is exactly what makes this comparison
        meaningful: both go through the same ``build_target_frame``
        regardless of which model type produced them).
    model_a_name, model_b_name:
        Column-name labels for the two models' predicted columns in the
        output.

    Returns
    -------
    DataFrame, one row per bucket (``decile`` numbered 1..``n_deciles``,
    ascending ``y_pred_a / y_pred_b`` ratio): ``n_obs``, ``exposure``,
    ``actual``, ``<model_a_name>``, ``<model_b_name>``,
    ``ratio_a_over_b`` (the exposure-weighted mean sort key within the
    bucket, for reference).

    Notes
    -----
    A row where both models predict (near-)zero makes the ratio
    ill-defined (0/0) or unstable; such rows are still included (sorted
    by whatever floating-point ratio results) since excluding them would
    bias buckets away from the lowest-risk end of the book. If either
    model routinely predicts exactly zero for a target, treat the bucket
    containing those rows with caution rather than dropping this check
    entirely.
    """
    tf_a = build_target_frame(pf, target, y_pred_a, peril=peril, coverage=coverage)
    tf_b = build_target_frame(pf, target, y_pred_b, peril=peril, coverage=coverage)

    if not np.array_equal(tf_a.numerator, tf_b.numerator) or not np.array_equal(
        tf_a.weight, tf_b.weight
    ):
        raise ValueError(
            "y_pred_a and y_pred_b must be predictions for the exact same "
            "rows (same PolicyFrame/target/peril/coverage) -- their "
            "observed numerator/weight don't match, which usually means "
            "one of them was computed against a different view."
        )

    ratio = tf_a.predicted / tf_b.predicted
    bucket = exposure_deciles(tf_a.weight, ratio, n_deciles)

    frame = pd.DataFrame(
        {
            "decile": bucket,
            "numerator": tf_a.numerator,
            "weight": tf_a.weight,
            "predicted_a": tf_a.predicted_numerator,
            "predicted_b": tf_b.predicted_numerator,
            "ratio": ratio,
        }
    )
    grouped = (
        frame.groupby("decile")
        .agg(
            n_obs=("weight", "size"),
            exposure=("weight", "sum"),
            actual_numerator=("numerator", "sum"),
            predicted_a_total=("predicted_a", "sum"),
            predicted_b_total=("predicted_b", "sum"),
            ratio_a_over_b=("ratio", lambda r: np.average(r, weights=frame.loc[r.index, "weight"])),
        )
        .reset_index()
    )
    grouped["actual"] = grouped["actual_numerator"] / grouped["exposure"]
    grouped[model_a_name] = grouped["predicted_a_total"] / grouped["exposure"]
    grouped[model_b_name] = grouped["predicted_b_total"] / grouped["exposure"]
    return grouped[
        ["decile", "n_obs", "exposure", "actual", model_a_name, model_b_name, "ratio_a_over_b"]
    ]
