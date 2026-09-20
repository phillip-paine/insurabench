"""Exposure-weighted Lorenz curve and normalized Gini index (design brief
§6).

Follows the ordered-Lorenz-curve convention standard in actuarial pricing
literature (Frees, Meyers & Cummings): rows sorted ascending by predicted
value, plotting cumulative exposure share (x) against cumulative actual-
loss share (y) at that sort order. A model that ranks risk no better than
random traces the diagonal (Gini 0); a model that ranks identically to
sorting on the actual outcome itself (the best any model *could* do,
short of hindsight) is the normalizing "ideal" curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurabench.curves._common import Target, build_target_frame
from insurabench.data.policy_frame import PolicyFrame


def _lorenz_points(numerator: np.ndarray, weight: np.ndarray, predicted_numerator: np.ndarray) -> pd.DataFrame:
    order = np.argsort(predicted_numerator / weight, kind="mergesort")
    w = weight[order]
    num = numerator[order]

    cum_exposure = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    total_loss = num.sum()
    if total_loss > 0:
        cum_loss = np.concatenate([[0.0], np.cumsum(num) / total_loss])
    else:
        cum_loss = np.zeros(len(w) + 1)
    return pd.DataFrame({"cum_exposure": cum_exposure, "cum_loss": cum_loss})


def _raw_gini(curve: pd.DataFrame) -> float:
    # np.trapezoid (numpy>=2.0) -- np.trapz was removed, not just deprecated,
    # as of the numpy>=2.5 this project depends on.
    return float(2 * np.trapezoid(curve["cum_loss"], curve["cum_exposure"]) - 1)


def gini_curve(
    pf: PolicyFrame,
    target: Target,
    y_pred,
    *,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> pd.DataFrame:
    """The ordered Lorenz curve points for one model's predictions:
    ``cum_exposure``, ``cum_loss``, sorted ascending by predicted value,
    starting at (0, 0) and ending at (1, 1). See ``gini_index`` for the
    scalar summary. ``target``/``y_pred`` as in
    ``insurabench.curves._common.build_target_frame``.
    """
    tf = build_target_frame(pf, target, y_pred, peril=peril, coverage=coverage)
    return _lorenz_points(tf.numerator, tf.weight, tf.predicted_numerator)


def gini_index(
    pf: PolicyFrame,
    target: Target,
    y_pred,
    *,
    normalized: bool = True,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> float:
    """Scalar Gini from the ordered Lorenz curve: ``2 * AUC - 1`` against
    the diagonal, computed on exposure-weighted points.

    ``normalized=True`` (the default, design brief §6) divides the raw
    model Gini by the Gini of the *ideal* ordering -- sorting by the
    actual outcome itself instead of the predicted value, the best any
    model could do short of hindsight -- so 1.0 always means "as good as
    sorting on hindsight" and -1.0 means "as wrong as it's possible to be
    (perfectly anti-sorted)", rather than a raw, unbounded-in-practice
    AUC-style number whose scale depends on how concentrated the loss
    distribution itself happens to be. Pass ``normalized=False`` for the
    raw (non-normalized) Gini instead.

    Returns ``nan`` if the ideal Gini is exactly 0 (every row has
    identical loss, so there is nothing to rank).
    """
    tf = build_target_frame(pf, target, y_pred, peril=peril, coverage=coverage)
    model_gini = _raw_gini(_lorenz_points(tf.numerator, tf.weight, tf.predicted_numerator))
    if not normalized:
        return model_gini

    # The ideal ordering: predicted_numerator == numerator (a "perfect"
    # prediction) sorts by the actual outcome and reproduces the actual
    # cumulative-loss curve exactly -- the best-possible Lorenz curve to
    # normalize against.
    ideal_gini = _raw_gini(_lorenz_points(tf.numerator, tf.weight, tf.numerator))
    if ideal_gini == 0:
        return float("nan")
    return model_gini / ideal_gini
