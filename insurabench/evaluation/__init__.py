"""Model-agnostic, exposure-correct evaluation metrics (design brief §6).

``lift_chart``, ``double_lift_chart``, ``gini_curve``/``gini_index``, and
``calibration_table``/``calibration_index`` never call a model -- each
takes a plain ``y_pred`` array and shares the same fixed exposure-decile
convention (``insurabench.curves._common.exposure_deciles``) where
relevant, so charts are directly comparable across every function here.
``bootstrap_relativities``/``split_relativities``/``stability_summary``
(``evaluation.stability``) round this layer out (build order step 3).
"""
from __future__ import annotations

from insurabench.evaluation.calibration import calibration_index, calibration_table
from insurabench.evaluation.double_lift import double_lift_chart
from insurabench.evaluation.gini import gini_curve, gini_index
from insurabench.evaluation.lift import lift_chart
from insurabench.evaluation.stability import (
    bootstrap_relativities,
    split_relativities,
    stability_summary,
)

__all__ = [
    "bootstrap_relativities",
    "calibration_index",
    "calibration_table",
    "double_lift_chart",
    "gini_curve",
    "gini_index",
    "lift_chart",
    "split_relativities",
    "stability_summary",
]
