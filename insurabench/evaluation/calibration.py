"""Observed-vs-predicted calibration checks by predicted-value decile
(design brief §6).

Where ``lift_chart`` (and its cousin ``gini_index``) tell you whether a
model ranks risk well -- relative ordering -- calibration asks a
different question: at each level of predicted risk, does the *average*
observed outcome actually match the *average* prediction? A model can
rank well (high Gini) while still being systematically miscalibrated in
one part of the risk spectrum (e.g. consistently under-predicting the
highest-risk decile) -- that's exactly what this catches and ``gini``
cannot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurabench.curves._common import Target, build_target_frame, exposure_deciles
from insurabench.curves.one_way import _z_score
from insurabench.data.policy_frame import PolicyFrame


def calibration_table(
    pf: PolicyFrame,
    target: Target,
    y_pred,
    *,
    n_bins: int = 10,
    confidence: float = 0.95,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> pd.DataFrame:
    """Exposure-weighted actual vs. predicted by predicted-value decile,
    with a per-bucket dispersion check.

    Buckets are formed exactly as in ``lift_chart`` (ascending predicted
    value, cumulative-exposure-share deciles via
    ``insurabench.curves._common.exposure_deciles``) -- calibration and
    lift deliberately share a bucketing convention so the two charts are
    directly comparable side by side.

    Returns
    -------
    DataFrame, one row per bucket: ``decile``, ``n_obs``, ``exposure``,
    ``observed``, ``fitted``, ``standard_error`` (of ``observed``, same
    distribution-agnostic reliability-weighted approximation as
    ``one_way_curve`` -- see its docstring), ``z`` (``(observed -
    fitted) / standard_error``: near 0 means well-calibrated in that
    bucket; beyond roughly +/-2 for a 95% check is a bucket worth
    scrutinizing).

    Notes
    -----
    ``z`` is ``nan`` for a bucket whose standard error is exactly 0 (e.g.
    every row in the bucket has an identical observed ratio) -- treat
    such a bucket as needing a visual/manual check rather than reading
    "nan" as either pass or fail.
    """
    tf = build_target_frame(pf, target, y_pred, peril=peril, coverage=coverage)
    bucket = exposure_deciles(tf.weight, tf.predicted, n_bins)
    ratio = tf.numerator / tf.weight

    frame = pd.DataFrame(
        {
            "decile": bucket,
            "numerator": tf.numerator,
            "weight": tf.weight,
            "predicted_numerator": tf.predicted_numerator,
            "ratio": ratio,
        }
    )

    z_crit = _z_score(confidence)
    rows = []
    for decile, g in frame.groupby("decile", sort=True):
        w = g["weight"].to_numpy()
        weight_sum = float(w.sum())
        observed = float(g["numerator"].sum() / weight_sum)
        fitted = float(g["predicted_numerator"].sum() / weight_sum)
        sum_w_sq = float(np.sum(w**2))
        n_eff = (weight_sum**2) / sum_w_sq if sum_w_sq > 0 else len(g)
        weighted_var = float(np.average((g["ratio"].to_numpy() - observed) ** 2, weights=w))
        se = float(np.sqrt(weighted_var / n_eff)) if n_eff > 0 else float("nan")
        z_stat = (observed - fitted) / se if se > 0 else float("nan")
        rows.append(
            {
                "decile": int(decile),
                "n_obs": len(g),
                "exposure": weight_sum,
                "observed": observed,
                "fitted": fitted,
                "standard_error": se,
                "z": z_stat,
                "significant": bool(abs(z_stat) > z_crit) if not np.isnan(z_stat) else False,
            }
        )
    return pd.DataFrame(rows)


def calibration_index(
    pf: PolicyFrame,
    target: Target,
    y_pred,
    *,
    n_bins: int = 10,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> float:
    """Single dispersion summary across all buckets: the exposure-weighted
    mean of ``z**2`` from ``calibration_table``.

    A value near 1 is what a well-calibrated model with correctly-sized
    confidence bands should produce (a squared standard-normal variate
    has expectation 1). Substantially above 1 means the model is
    miscalibrated (systematically off in one or more buckets) beyond what
    sampling noise alone would explain; substantially below 1 usually
    means the bands themselves are too wide (e.g. real overdispersion
    relative to the distribution-agnostic variance approximation used
    here) rather than that the model is unusually well calibrated.
    Buckets with ``nan``/zero standard error are excluded from the
    average rather than treated as 0.
    """
    table = calibration_table(pf, target, y_pred, n_bins=n_bins, peril=peril, coverage=coverage)
    valid = table[table["z"].notna()]
    if valid.empty:
        return float("nan")
    return float(np.average(valid["z"] ** 2, weights=valid["exposure"]))
