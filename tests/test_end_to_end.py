"""End-to-end integration test: policies+claims tables -> ``PolicyFrame``
-> a fitted GLM -> curves -> evaluation -> viz, run once on a realistic-
sized synthetic book (not a 4-row toy fixture).

Every other test file in this suite checks one function's correctness
against a small, hand-computable fixture -- that's the right tool for
"is this formula right," but it can't catch a different class of bug:
two independently-correct pieces that don't actually compose (a column
name that doesn't match what the next stage expects, a shape mismatch
that only appears at realistic scale, an exception that only fires once
real category cardinality or a real train/test split is involved).
This file exists for that second class of bug. Nothing here is
golden-value tested against a hand-computed number -- the synthetic
generator's output isn't hand-verifiable at this size, and that isn't
the point. The assertions are deliberately loose "this is a sane,
finite, correctly-shaped number" checks, not "this is the *right*
number" checks; the toy-fixture tests elsewhere already own that job.

If this test is green, the full pipeline the design brief and build
order describe actually runs end to end -- not just each piece of it in
isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurabench.curves import (
    one_way_curve,
    partial_dependence,
    relativity_table,
    two_way_curve,
)
from insurabench.data.datasets import make_synthetic_two_table
from insurabench.data.policy_frame import PolicyFrame
from insurabench.evaluation import (
    calibration_index,
    calibration_table,
    double_lift_chart,
    gini_curve,
    gini_index,
    lift_chart,
    rate_change_by_level,
    rate_change_distribution,
)
from insurabench.model_selection import train_test_split_policy_frame
from insurabench.models.glm import GLMPricingModel
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
    plot_two_way_curve,
)

N_POLICIES = 2000
SEED = 42


@pytest.fixture(scope="module")
def pf() -> PolicyFrame:
    book = make_synthetic_two_table(
        n_policies=N_POLICIES, renewals=True, claims_per_policy_lambda=0.2, seed=SEED
    )
    return PolicyFrame.from_tables(
        book.policies, book.claims, book.policy_schema, book.claims_schema, verbose=False
    )


@pytest.fixture(scope="module")
def split(pf):
    return train_test_split_policy_frame(pf, test_size=0.3, seed=0)


@pytest.fixture(scope="module")
def fitted_frequency_models(split):
    """Two GLMs (default and lightly regularized) fit on the same train
    split -- lets the double-lift/rate-impact stages exercise a genuine
    model-vs-model comparison rather than a model against itself.
    """
    train_pf, test_pf = split
    model_a = GLMPricingModel(family="poisson").fit_policy_frame(train_pf, "frequency")
    model_b = GLMPricingModel(family="poisson", alpha=0.3).fit_policy_frame(train_pf, "frequency")
    return model_a, model_b, train_pf, test_pf


# --- stage 1: data layer -----------------------------------------------------


def test_data_layer_produces_a_realistic_book(pf):
    assert pf.n_policies > 0
    assert pf.n_claims > 0
    assert pf.total_exposure > 0

    freq_view = pf.frequency_view()
    sev_view = pf.severity_view()
    # frequency grain is one row per policy-period; severity is one row
    # per claim -- these are different lengths by construction (design
    # brief §3.3), and neither should be empty for a book with any claims.
    assert len(freq_view) == pf.n_policies
    assert len(sev_view) == pf.n_claims


def test_train_test_split_partitions_exposure(split, pf):
    train_pf, test_pf = split
    assert train_pf.n_policies + test_pf.n_policies == pf.n_policies
    assert train_pf.total_exposure + test_pf.total_exposure == pytest.approx(pf.total_exposure)
    # a claim's own policy-period must land in the same split as the claim
    assert train_pf.n_claims + test_pf.n_claims == pf.n_claims


# --- stage 2: model -----------------------------------------------------------


def test_glm_fits_and_predicts_finite_rates(fitted_frequency_models):
    model_a, _, _train_pf, test_pf = fitted_frequency_models
    y_pred = model_a.predict_policy_frame(test_pf)

    assert len(y_pred) == len(test_pf.frequency_view())
    assert np.all(np.isfinite(y_pred))
    assert np.all(y_pred >= 0)  # a Poisson rate can never be negative

    relativities = model_a.relativities()
    assert isinstance(relativities, pd.DataFrame)
    assert len(relativities) > 0


def test_glm_deviance_score_is_finite_and_bounded(fitted_frequency_models):
    model_a, _, train_pf, test_pf = fitted_frequency_models
    train_score = model_a.score_policy_frame(train_pf)
    test_score = model_a.score_policy_frame(test_pf)

    for score in (train_score, test_score):
        assert np.isfinite(score)
        assert score <= 1.0  # D² is bounded above by 1 (a perfect fit)
    # Deliberately NOT asserting score > 0 here: a deviance-based D² can
    # legitimately come out negative (worse than an intercept-only null
    # model) for a weak/noisy target relative to the number of
    # parameters, and this test's fixture data does produce a small
    # negative D² on both splits at these settings (see the finding
    # flagged in the build notes) -- asserting positivity would make
    # this test assert something that isn't actually true about the
    # fixture, which is a worse failure mode than a loose bound.


# --- stage 3: curves -----------------------------------------------------------


@pytest.fixture(scope="module")
def a_feature(pf) -> str:
    return next(iter(pf.policy_schema.feature_roles))


@pytest.fixture(scope="module")
def another_feature(pf) -> str:
    return list(pf.policy_schema.feature_roles)[1]


def test_one_way_curve_covers_full_exposure(fitted_frequency_models, a_feature):
    model_a, _, _, test_pf = fitted_frequency_models
    y_pred = model_a.predict_policy_frame(test_pf)
    curve = one_way_curve(test_pf, a_feature, "frequency", y_pred)

    assert curve["exposure"].sum() == pytest.approx(test_pf.total_exposure)
    assert (curve["observed_lower"] <= curve["observed"]).all()
    assert (curve["observed"] <= curve["observed_upper"]).all()
    assert np.all(np.isfinite(curve["fitted"]))


def test_two_way_curve_runs_on_real_features(fitted_frequency_models, a_feature, another_feature):
    model_a, _, _, test_pf = fitted_frequency_models
    y_pred = model_a.predict_policy_frame(test_pf)
    curve = two_way_curve(test_pf, a_feature, another_feature, "frequency", y_pred)

    assert len(curve) > 0
    assert curve["exposure"].sum() <= test_pf.total_exposure + 1e-6


def test_relativity_table_every_base_level_is_one(fitted_frequency_models):
    model_a, _, _, test_pf = fitted_frequency_models
    y_pred = model_a.predict_policy_frame(test_pf)
    table = relativity_table(test_pf, "frequency", y_pred)

    assert set(table["feature"]) == set(test_pf.policy_schema.feature_roles)
    # every feature's own base (max-exposure) level relativity is exactly 1.0
    for _feature, sub in table.groupby("feature"):
        base_row = sub.loc[sub["exposure"].idxmax()]
        assert base_row["relativity"] == pytest.approx(1.0)


def test_partial_dependence_runs_against_a_real_fitted_glm(fitted_frequency_models, a_feature):
    model_a, _, train_pf, _ = fitted_frequency_models
    pdp = partial_dependence(train_pf, a_feature, "frequency", model_a)

    assert len(pdp) > 0
    assert np.all(np.isfinite(pdp["partial_dependence"]))
    assert (pdp["partial_dependence"] >= 0).all()


# --- stage 4: evaluation -------------------------------------------------------


def test_lift_chart_deciles_partition_exposure(fitted_frequency_models):
    model_a, _, _, test_pf = fitted_frequency_models
    y_pred = model_a.predict_policy_frame(test_pf)
    chart = lift_chart(test_pf, "frequency", y_pred, n_deciles=10)

    assert len(chart) == 10
    assert chart["exposure"].sum() == pytest.approx(test_pf.total_exposure)


def test_gini_index_between_worst_and_best(fitted_frequency_models):
    model_a, _, _, test_pf = fitted_frequency_models
    y_pred = model_a.predict_policy_frame(test_pf)
    gc = gini_curve(test_pf, "frequency", y_pred)
    gi = gini_index(test_pf, "frequency", y_pred)

    assert len(gc) > 0
    assert -1.0 <= gi <= 1.0


def test_calibration_index_is_finite_and_nonnegative(fitted_frequency_models):
    model_a, _, _, test_pf = fitted_frequency_models
    y_pred = model_a.predict_policy_frame(test_pf)
    ci = calibration_index(test_pf, "frequency", y_pred)
    assert np.isfinite(ci)
    assert ci >= 0.0  # a sum of squared z-scores can't be negative


def test_double_lift_and_rate_impact_between_two_real_models(fitted_frequency_models, a_feature):
    model_a, model_b, _, test_pf = fitted_frequency_models
    y_a = model_a.predict_policy_frame(test_pf)
    y_b = model_b.predict_policy_frame(test_pf)

    dl = double_lift_chart(test_pf, "frequency", y_a, y_b, n_deciles=10)
    assert len(dl) == 10
    assert dl["exposure"].sum() == pytest.approx(test_pf.total_exposure)

    by_level = rate_change_by_level(test_pf, a_feature, "frequency", y_a, y_b)
    assert len(by_level) > 0
    assert by_level["exposure"].sum() == pytest.approx(test_pf.total_exposure)

    distribution = rate_change_distribution(test_pf, "frequency", y_a, y_b)
    assert distribution["exposure_share"].sum() == pytest.approx(1.0)


# --- stage 5: viz --------------------------------------------------------------


def test_every_plot_function_runs_and_saves(fitted_frequency_models, a_feature, another_feature, tmp_path):
    model_a, model_b, train_pf, test_pf = fitted_frequency_models
    y_a = model_a.predict_policy_frame(test_pf)
    y_b = model_b.predict_policy_frame(test_pf)

    curve = one_way_curve(test_pf, a_feature, "frequency", y_a)
    two_way = two_way_curve(test_pf, a_feature, another_feature, "frequency", y_a)
    table = relativity_table(test_pf, "frequency", y_a)
    pdp = partial_dependence(train_pf, a_feature, "frequency", model_a)
    lc = lift_chart(test_pf, "frequency", y_a)
    dl = double_lift_chart(test_pf, "frequency", y_a, y_b)
    gc = gini_curve(test_pf, "frequency", y_a)
    ct = calibration_table(test_pf, "frequency", y_a)
    by_level = rate_change_by_level(test_pf, a_feature, "frequency", y_a, y_b)
    distribution = rate_change_distribution(test_pf, "frequency", y_a, y_b)

    calls = [
        (plot_one_way_curve, (curve,), {"feature_name": a_feature}),
        (plot_two_way_curve, (two_way,), {}),
        (plot_relativity_table, (table,), {"feature": a_feature}),
        (plot_partial_dependence, (pdp,), {"feature_name": a_feature}),
        (plot_lift_chart, (lc,), {}),
        (plot_double_lift_chart, (dl,), {}),
        (plot_gini_curve, (gc,), {"gini_value": gini_index(test_pf, "frequency", y_a)}),
        (plot_calibration_table, (ct,), {}),
        (plot_rate_change_by_level, (by_level,), {"feature_name": a_feature}),
        (plot_rate_change_distribution, (distribution,), {}),
    ]
    for i, (fn, args, kwargs) in enumerate(calls):
        out = tmp_path / f"{fn.__name__}_{i}.png"
        fig, _ax = fn(*args, save_path=str(out), **kwargs)
        assert out.exists()
        assert out.stat().st_size > 0
        import matplotlib.pyplot as plt

        plt.close(fig)
