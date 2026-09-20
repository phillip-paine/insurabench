"""Shared row-grain plumbing for the curves and evaluation layers (design
brief §5, §6).

Both layers need the same three things, aligned row-for-row, for a given
PolicyFrame + target: the observed response numerator, the row weight
(exposure for frequency/pure_premium, 1 per claim for severity), and the
model's prediction converted onto that same numerator scale. Getting this
alignment right in exactly one place is what lets ``one_way``,
``relativity_table``, ``lift``, ``gini`` (and future evaluation modules)
all be written against a single (numerator, weight, predicted_numerator)
triple instead of each re-deriving per-target plumbing -- and risking a
different, silently inconsistent convention each time (design brief
§9.4/§9.5).

Deliberately, nothing here calls a model. Every function takes ``y_pred``
as a plain array the caller already produced (e.g.
``model.predict_policy_frame(pf)``, whatever ``model`` is). That is what
makes the curves/evaluation layer genuinely model-agnostic (design brief
§4's "critical requirement", §5/§6): a GLM's and a GBM's predictions are
both just an array here, and this module never needs to know which
produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from insurabench.data.policy_frame import PolicyFrame

Target = Literal["frequency", "severity", "pure_premium"]


@dataclass(frozen=True)
class TargetFrame:
    """Row-aligned (view, numerator, weight, predicted_numerator) for one
    target.

    ``view`` is the full ``frequency_view``/``severity_view``/
    ``pure_premium_view`` output (every declared feature column included),
    so curves can bin/group by any rating factor without a second merge.
    ``numerator``/``weight``/``predicted_numerator`` are aligned to
    ``view``'s row order.
    """

    view: pd.DataFrame
    numerator: np.ndarray
    weight: np.ndarray
    predicted_numerator: np.ndarray

    @property
    def observed(self) -> np.ndarray:
        """Per-row observed value on the rate scale (claim_count/exposure,
        claim_amount, or claim_amount_total/exposure, depending on
        target) -- this is what a curve's "actual" line is built from."""
        return self.numerator / self.weight

    @property
    def predicted(self) -> np.ndarray:
        """Per-row predicted value on that same rate scale."""
        return self.predicted_numerator / self.weight


def build_target_frame(
    pf: PolicyFrame,
    target: Target,
    y_pred,
    *,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> TargetFrame:
    """Assemble the (numerator, weight, predicted_numerator) triple for
    ``target``, checked against ``y_pred``.

    The three targets put ``y_pred`` on different scales at fit time (see
    ``insurabench.models.glm._build_design``), so this is also where that
    difference gets normalized away, onto a common "numerator" scale:

    - ``"frequency"``: fit with ``offset=log(exposure)``, so a model's
      prediction is already claim-count-scale. ``predicted_numerator =
      y_pred`` directly.
    - ``"severity"``: fit directly on claim amount, one row per claim
      (weight is always 1). ``predicted_numerator = y_pred`` directly.
    - ``"pure_premium"``: fit with ``sample_weight=exposure`` on
      ``claim_amount_total / exposure``, so a model's prediction is a
      *rate*, not a total. ``predicted_numerator = y_pred * weight`` puts
      it back onto the same total-amount scale as ``numerator`` (the
      actual ``claim_amount_total``), so it aggregates with a plain sum
      exactly like the other two targets do.

    Raises
    ------
    ValueError
        ``target`` isn't one of the three known targets, or ``y_pred``'s
        length doesn't match the target's view row count -- almost always
        means predictions were computed on the wrong view (e.g.
        frequency-view predictions passed in for a severity curve).
    """
    if target == "frequency":
        view = pf.frequency_view(peril=peril, coverage=coverage)
        numerator = view["claim_count"].to_numpy(dtype=float)
        weight = view[pf.policy_schema.exposure_col].to_numpy(dtype=float)
        predicted_numerator = np.asarray(y_pred, dtype=float)
    elif target == "severity":
        view = pf.severity_view(peril=peril, coverage=coverage)
        numerator = view[pf.claims_schema.claim_amount_col].to_numpy(dtype=float)
        weight = np.ones(len(view), dtype=float)
        predicted_numerator = np.asarray(y_pred, dtype=float)
    elif target == "pure_premium":
        view = pf.pure_premium_view(peril=peril, coverage=coverage)
        numerator = view["claim_amount_total"].to_numpy(dtype=float)
        weight = view[pf.policy_schema.exposure_col].to_numpy(dtype=float)
        predicted_numerator = np.asarray(y_pred, dtype=float) * weight
    else:
        raise ValueError(
            f"Unknown target: {target!r}. Choose 'frequency', 'severity', or 'pure_premium'."
        )

    if len(np.asarray(y_pred)) != len(view):
        raise ValueError(
            f"y_pred has {len(np.asarray(y_pred))} row(s) but the {target!r} "
            f"view for this PolicyFrame has {len(view)} row(s) -- predictions "
            f"must come from calling predict on that same target/peril/"
            f"coverage view (e.g. model.predict_policy_frame(pf) after "
            f"model.fit_policy_frame(pf, target={target!r})); passing "
            f"predictions built for a different target/view is a common "
            f"cause of this mismatch."
        )

    return TargetFrame(
        view=view, numerator=numerator, weight=weight, predicted_numerator=predicted_numerator
    )


def exposure_deciles(weight: np.ndarray, sort_key: np.ndarray, n_buckets: int) -> np.ndarray:
    """Bucket assignment (1..``n_buckets``) by cumulative-``weight`` share
    after sorting ascending by ``sort_key``.

    This is the one fixed decile convention design brief §6 requires:
    each bucket holds ~1/``n_buckets`` of total exposure (or claim count
    for severity), not ~1/``n_buckets`` of the row count and not an
    equal-width split of ``sort_key`` itself. Shared by
    ``evaluation.lift.lift_chart``, ``evaluation.double_lift.double_lift_chart``,
    and ``evaluation.calibration.calibration_table`` so the same
    convention can't drift between them.

    Returns an array the same length as ``weight``/``sort_key``, in their
    *original* row order (not sorted) -- caller's row ``i`` belongs to
    bucket ``result[i]``.
    """
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets!r}.")
    order = np.argsort(sort_key, kind="mergesort")
    cum_weight = np.cumsum(weight[order])
    total_weight = cum_weight[-1]
    bucket_sorted = np.minimum((cum_weight / total_weight * n_buckets).astype(int), n_buckets - 1) + 1
    bucket = np.empty_like(bucket_sorted)
    bucket[order] = bucket_sorted
    return bucket
