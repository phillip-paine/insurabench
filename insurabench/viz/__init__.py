"""Professional-looking, presentation-ready charts for every
``curves``/``evaluation`` output (design brief §8).

Every function here is a pure presentation step: it takes the plain
DataFrame (or float) a ``curves``/``evaluation`` function already
returned and draws it with ``insurabench.viz.theme``'s shared house
style. None of them call a model or touch a ``PolicyFrame`` -- compute
the curve/chart once, then plot it (and reuse it for an export, a table,
etc.) independently.

    from insurabench.curves import one_way_curve
    from insurabench.viz import plot_one_way_curve

    curve = one_way_curve(pf, "driver_age", "frequency", y_pred)
    fig, ax = plot_one_way_curve(curve, feature_name="driver_age", save_path="driver_age.png")

Every plotting function accepts ``ax=`` (draw onto an existing subplot,
for a multi-chart dashboard figure) and ``save_path=`` (write the figure
straight to a file at ``savefig``'s default 200 DPI, tight-cropped --
ready to drop into a slide).
"""
from __future__ import annotations

from insurabench.viz.curves import (
    plot_one_way_curve,
    plot_relativity_table,
    plot_two_way_curve,
)
from insurabench.viz.evaluation import (
    plot_calibration_table,
    plot_double_lift_chart,
    plot_gini_curve,
    plot_lift_chart,
    plot_stability_summary,
)
from insurabench.viz.theme import PALETTE, theme

__all__ = [
    "PALETTE",
    "plot_calibration_table",
    "plot_double_lift_chart",
    "plot_gini_curve",
    "plot_lift_chart",
    "plot_one_way_curve",
    "plot_relativity_table",
    "plot_stability_summary",
    "plot_two_way_curve",
    "theme",
]
