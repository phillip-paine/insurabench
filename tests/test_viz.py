"""Structural tests for insurabench.viz -- these don't (and can't sensibly)
golden-value-test pixel output, but they do check that every plotting
function runs headlessly, draws the shapes it claims to (right number of
axes/lines/bars, non-empty tick labels), respects ax=/save_path=, and
raises on the same misuse cases its docstring calls out.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from insurabench.curves.one_way import one_way_curve
from insurabench.curves.partial_dependence import partial_dependence
from insurabench.curves.relativity_table import relativity_table
from insurabench.curves.two_way import two_way_curve
from insurabench.evaluation.calibration import calibration_table
from insurabench.evaluation.double_lift import double_lift_chart
from insurabench.evaluation.gini import gini_curve
from insurabench.evaluation.lift import lift_chart
from insurabench.evaluation.rate_impact import (
    rate_change_by_level,
    rate_change_distribution,
)
from insurabench.evaluation.stability import bootstrap_relativities, stability_summary
from insurabench.viz import (
    plot_calibration_table,
    plot_double_lift_chart,
    plot_gini_curve,
    plot_lift_chart,
    plot_one_way_curve,
    plot_partial_dependence,
    plot_rate_change_by_level,
    plot_rate_change_distribution,
    plot_relativity_table,
    plot_stability_summary,
    plot_two_way_curve,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_one_way_curve_structure(toy_frequency_pf):
    curve = one_way_curve(toy_frequency_pf, "region", "frequency", np.array([1.0, 2.0, 0.5, 1.5]))
    fig, ax = plot_one_way_curve(curve, feature_name="region")

    assert len(fig.axes) == 2  # foreground rate axes + secondary exposure axes
    assert ax.get_xlabel() == "region"
    assert [t.get_text() for t in ax.get_xticklabels()] == list(curve["level"])
    assert len(ax.lines) == 2  # observed + fitted
    assert len(ax.collections) == 1  # the fill_between CI band
    assert "vehicle_power" not in ax.get_title()  # sanity: title uses feature_name, not garbage


def test_plot_one_way_curve_accepts_existing_ax(toy_frequency_pf):
    curve = one_way_curve(toy_frequency_pf, "region", "frequency", np.array([1.0, 2.0, 0.5, 1.5]))
    _fig, ax = plt.subplots()
    fig_out, ax_out = plot_one_way_curve(curve, ax=ax)
    assert fig_out is _fig
    assert ax_out is ax


def test_plot_one_way_curve_saves_file(toy_frequency_pf, tmp_path):
    curve = one_way_curve(toy_frequency_pf, "region", "frequency", np.array([1.0, 2.0, 0.5, 1.5]))
    out = tmp_path / "one_way.png"
    plot_one_way_curve(curve, save_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_two_way_curve_structure(toy_two_feature_pf):
    curve = two_way_curve(toy_two_feature_pf, "region", "vehicle_type", "frequency", np.array([1.0, 2.0, 0.5, 1.5]))
    _fig, ax = plot_two_way_curve(curve, x_label="region", y_label="vehicle_type")

    assert len(ax.images) == 1  # the heatmap itself
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["X", "Y"]
    # one annotated text per cell (2x2 grid, all cells populated in this fixture)
    assert len(ax.texts) == 4


def test_plot_two_way_curve_invalid_value_raises(toy_two_feature_pf):
    curve = two_way_curve(toy_two_feature_pf, "region", "vehicle_type", "frequency", np.array([1.0, 2.0, 0.5, 1.5]))
    with pytest.raises(ValueError, match="'fitted' or 'observed'"):
        plot_two_way_curve(curve, value="nonsense")


def test_plot_relativity_table_single_feature(toy_frequency_pf):
    table = relativity_table(toy_frequency_pf, "frequency", np.array([1.0, 2.0, 0.5, 1.5]))
    _fig, ax = plot_relativity_table(table, feature="region")
    assert len(ax.patches) == 2  # two bars, one per region level
    assert ax.get_ylabel() == "Relativity"


def test_plot_relativity_table_unknown_feature_raises(toy_frequency_pf):
    table = relativity_table(toy_frequency_pf, "frequency", np.array([1.0, 2.0, 0.5, 1.5]))
    with pytest.raises(ValueError, match="No rows for feature"):
        plot_relativity_table(table, feature="not_a_feature")


def test_plot_relativity_table_facet_rejects_ax(toy_frequency_pf):
    table = relativity_table(toy_frequency_pf, "frequency", np.array([1.0, 2.0, 0.5, 1.5]))
    _fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Faceting"):
        plot_relativity_table(table, ax=ax)


def test_plot_lift_chart_structure(toy_frequency_pf):
    chart = lift_chart(toy_frequency_pf, "frequency", np.array([1.0, 2.0, 0.5, 1.5]), n_deciles=2)
    fig, ax = plot_lift_chart(chart)
    assert len(fig.axes) == 2
    assert len(ax.lines) == 2  # actual + predicted
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2"]


def test_plot_double_lift_chart_structure(toy_frequency_pf):
    chart = double_lift_chart(
        toy_frequency_pf,
        "frequency",
        np.array([1.0, 2.0, 0.5, 1.5]),
        np.array([0.5, 1.0, 1.0, 1.0]),
        n_deciles=2,
    )
    _fig, ax = plot_double_lift_chart(chart)
    assert len(ax.lines) == 3  # actual + model_a + model_b


def test_plot_double_lift_chart_wrong_model_name_raises(toy_frequency_pf):
    chart = double_lift_chart(
        toy_frequency_pf,
        "frequency",
        np.array([1.0, 2.0, 0.5, 1.5]),
        np.array([0.5, 1.0, 1.0, 1.0]),
        model_a_name="glm",
        model_b_name="gbm",
    )
    with pytest.raises(ValueError, match="not a column"):
        plot_double_lift_chart(chart)  # defaults model_a/model_b don't exist here


def test_plot_gini_curve_structure(toy_frequency_pf):
    curve = gini_curve(toy_frequency_pf, "frequency", np.array([1.0, 2.0, 0.5, 1.5]))
    _fig, ax = plot_gini_curve(curve, gini_value=0.42)
    assert len(ax.lines) == 2  # model curve + diagonal
    texts = [t.get_text() for t in ax.texts]
    assert any("0.420" in t for t in texts)
    assert ax.get_xlim() == (0.0, 1.0)


def test_plot_calibration_table_structure(toy_frequency_pf):
    table = calibration_table(toy_frequency_pf, "frequency", np.array([1.0, 2.0, 0.5, 1.5]), n_bins=2)
    fig, ax = plot_calibration_table(table)
    assert len(fig.axes) == 2
    # errorbar creates one Line2D for the point markers plus cap/bar artists;
    # at minimum the fitted line and the errorbar's marker line should both exist.
    assert len(ax.lines) >= 2


def test_plot_stability_summary_structure(toy_frequency_pf):
    draws = bootstrap_relativities(
        toy_frequency_pf, "region", "frequency", np.array([1.0, 2.0, 0.5, 1.5]), n_bootstrap=20, seed=0
    )
    summary = stability_summary(draws)
    _fig, ax = plot_stability_summary(summary)
    assert [t.get_text() for t in ax.get_xticklabels()] == list(summary["level"])
    assert ax.get_ylabel() == "Relativity"


def test_plot_rate_change_by_level_structure(toy_frequency_pf):
    table = rate_change_by_level(
        toy_frequency_pf,
        "region",
        "frequency",
        np.array([1.0, 2.0, 0.5, 1.5]),
        np.array([0.5, 1.0, 1.0, 1.0]),
    )
    _fig, ax = plot_rate_change_by_level(table, feature_name="region")

    assert len(ax.patches) == 2  # one bar per region level
    assert len(ax.texts) == 2  # one "up/down" annotation per bar
    # every annotation must land within the axes' own ylim -- this is
    # exactly the bug that was caught and fixed (annotations drifting
    # outside a data-range-proportional padding and blowing out the
    # saved canvas via bbox_inches="tight").
    y_min, y_max = ax.get_ylim()
    for txt in ax.texts:
        _, y = txt.get_position()
        assert y_min <= y <= y_max


def test_plot_rate_change_by_level_saves_file(toy_frequency_pf, tmp_path):
    table = rate_change_by_level(
        toy_frequency_pf,
        "region",
        "frequency",
        np.array([1.0, 2.0, 0.5, 1.5]),
        np.array([0.5, 1.0, 1.0, 1.0]),
    )
    out = tmp_path / "rate_change.png"
    plot_rate_change_by_level(table, save_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_rate_change_distribution_structure(toy_frequency_pf):
    table = rate_change_distribution(
        toy_frequency_pf,
        "frequency",
        np.array([1.0, 2.0, 0.5, 1.5]),
        np.array([0.5, 1.0, 1.0, 1.0]),
    )
    _fig, ax = plot_rate_change_distribution(table)
    assert len(ax.patches) == len(table)  # one bar per band
    # only bands with nonzero share get a text label
    nonzero_bands = int((table["exposure_share"] > 0).sum())
    assert len(ax.texts) == nonzero_bands


class _StubRateModel:
    """Minimal PricingModel-shaped stand-in (mirrors tests/test_curves.py's
    stub) -- predicts a fixed rate per region, scaled to the count scale
    by offset. Used here only to exercise plot_partial_dependence without
    depending on glum.
    """

    def __init__(self, rates: dict[str, float]):
        self.rates = rates

    def predict(self, X, *, offset=None):
        base_rate = X["region"].map(self.rates).to_numpy(dtype=float)
        if offset is None:
            return base_rate
        return base_rate * np.exp(np.asarray(offset))


def test_plot_partial_dependence_structure(toy_frequency_pf):
    model = _StubRateModel(rates={"A": 1.0, "B": 3.0})
    pdp = partial_dependence(toy_frequency_pf, "region", "frequency", model)
    fig, ax = plot_partial_dependence(pdp, feature_name="region")

    assert len(fig.axes) == 1  # no exposure axis here, unlike plot_one_way_curve
    assert len(ax.lines) == 1  # a single fitted line, no observed/CI band
    assert [t.get_text() for t in ax.get_xticklabels()] == list(pdp["level"])
    assert ax.get_xlabel() == "region"
    assert ax.get_ylabel() == "Partial dependence (rate)"
    # the line's y-data must match the pdp values exactly (values 1.0, 3.0)
    assert np.allclose(ax.lines[0].get_ydata(), pdp["partial_dependence"])


def test_plot_partial_dependence_saves_file(toy_frequency_pf, tmp_path):
    model = _StubRateModel(rates={"A": 1.0, "B": 3.0})
    pdp = partial_dependence(toy_frequency_pf, "region", "frequency", model)
    out = tmp_path / "pdp.png"
    plot_partial_dependence(pdp, save_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_finalize_calls_plt_show(monkeypatch):
    """Regression test for a real notebook-display bug found while
    building the Workbook notebooks: in a Jupyter/inline-backend session,
    matplotlib_inline auto-displays open figures via a post-execute hook
    -- but confirmed directly that this hook silently stops firing after
    the *first* figure drawn inside a `with plt.rc_context(...):` block
    in that kernel session (every subsequent insurabench chart in the
    same notebook produced no output at all, no error either, since
    every insurabench.viz function wraps its drawing in `theme()`, which
    is exactly such an rc_context). An explicit `plt.show()` in
    `finalize` sidesteps this -- this test pins that call, rather than
    only the structural checks the rest of this file does (which use the
    Agg backend and can't exercise this bug at all, agnostic to display).
    """
    from insurabench.viz.theme import finalize

    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))

    fig, _ax = plt.subplots()
    finalize(fig, None)

    assert calls == [1]
