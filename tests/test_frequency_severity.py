"""Tests for insurabench.models.frequency_severity.

Three concerns:

1. Basic fit/predict/score plumbing, GLM+GLM (the simplest composition).
2. Mixed model types (GLM frequency leg + GBM severity leg, and vice
   versa) -- the actual point of the module: either leg can be either
   model type.
3. Model-agnosticism at the curves/evaluation layer, exactly mirroring
   ``tests/test_gbm.py``'s section 3 -- a composed model's
   ``predict_policy_frame`` output must run through curves/evaluation
   unchanged, and must be D²/Gini-comparable to a direct single-model
   Tweedie pure-premium fit.
"""
from __future__ import annotations

import numpy as np
import pytest

from insurabench.curves import one_way_curve, relativity_table
from insurabench.data.datasets import make_synthetic_two_table
from insurabench.data.policy_frame import PolicyFrame
from insurabench.evaluation import gini_index, lift_chart
from insurabench.models.frequency_severity import FrequencySeverityModel
from insurabench.models.gbm import GBMPricingModel
from insurabench.models.glm import GLMPricingModel


@pytest.fixture(scope="module")
def pf() -> PolicyFrame:
    book = make_synthetic_two_table(n_policies=1200, renewals=True, seed=11)
    return PolicyFrame.from_tables(
        book.policies, book.claims, book.policy_schema, book.claims_schema, verbose=False
    )


# --- 1. Basic plumbing, GLM+GLM ------------------------------------------------


def test_glm_plus_glm_fits_and_predicts_finite_rates(pf):
    model = FrequencySeverityModel(
        frequency_model=GLMPricingModel(family="poisson"),
        severity_model=GLMPricingModel(family="gamma"),
    ).fit_policy_frame(pf)

    y_pred = model.predict_policy_frame(pf)
    assert len(y_pred) == len(pf.pure_premium_view())
    assert np.all(np.isfinite(y_pred))
    assert np.all(y_pred >= 0)


def test_score_policy_frame_is_finite_and_comparable_to_direct_tweedie(pf):
    composed = FrequencySeverityModel(
        frequency_model=GLMPricingModel(family="poisson"),
        severity_model=GLMPricingModel(family="gamma"),
    ).fit_policy_frame(pf)
    direct = GLMPricingModel(family="tweedie", power=1.5).fit_policy_frame(pf, "pure_premium")

    d2_composed = composed.score_policy_frame(pf, power=1.5)
    d2_direct = direct.score_policy_frame(pf)

    assert np.isfinite(d2_composed)
    assert np.isfinite(d2_direct)


def test_score_policy_frame_requires_explicit_power(pf):
    model = FrequencySeverityModel(
        frequency_model=GLMPricingModel(family="poisson"),
        severity_model=GLMPricingModel(family="gamma"),
    ).fit_policy_frame(pf)
    with pytest.raises(TypeError):
        model.score_policy_frame(pf)  # power is keyword-only, no default


def test_predict_before_fit_raises(pf):
    model = FrequencySeverityModel(
        frequency_model=GLMPricingModel(family="poisson"),
        severity_model=GLMPricingModel(family="gamma"),
    )
    with pytest.raises(RuntimeError, match="fit_policy_frame"):
        model.predict_policy_frame(pf)


def test_same_instance_for_both_legs_raises():
    shared = GLMPricingModel(family="poisson")
    with pytest.raises(ValueError, match="two separate instances"):
        FrequencySeverityModel(frequency_model=shared, severity_model=shared)


def test_relativities_returns_both_legs_separately(pf):
    model = FrequencySeverityModel(
        frequency_model=GLMPricingModel(family="poisson"),
        severity_model=GLMPricingModel(family="gamma"),
    ).fit_policy_frame(pf)
    rel = model.relativities()
    assert set(rel) == {"frequency", "severity"}
    assert "relativity" in rel["frequency"].columns
    assert "relativity" in rel["severity"].columns


# --- 2. Mixed model types -------------------------------------------------------


def test_glm_frequency_gbm_severity(pf):
    model = FrequencySeverityModel(
        frequency_model=GLMPricingModel(family="poisson"),
        severity_model=GBMPricingModel(family="gamma", iterations=100),
    ).fit_policy_frame(pf)
    y_pred = model.predict_policy_frame(pf)
    assert np.all(np.isfinite(y_pred))
    assert np.all(y_pred >= 0)


def test_gbm_frequency_glm_severity(pf):
    model = FrequencySeverityModel(
        frequency_model=GBMPricingModel(family="poisson", iterations=100),
        severity_model=GLMPricingModel(family="gamma"),
    ).fit_policy_frame(pf)
    y_pred = model.predict_policy_frame(pf)
    assert np.all(np.isfinite(y_pred))
    assert np.all(y_pred >= 0)


def test_gbm_plus_gbm(pf):
    model = FrequencySeverityModel(
        frequency_model=GBMPricingModel(family="zip", zip_gamma=1.0, iterations=100),
        severity_model=GBMPricingModel(family="gamma", iterations=100),
    ).fit_policy_frame(pf)
    y_pred = model.predict_policy_frame(pf)
    assert np.all(np.isfinite(y_pred))
    assert np.all(y_pred >= 0)
    assert np.isfinite(model.score_policy_frame(pf, power=1.5))


def test_underlying_family_target_mismatch_still_raises(pf):
    # FrequencySeverityModel doesn't duplicate GBMPricingModel's own
    # family/target guard -- fit_policy_frame delegates straight to each
    # leg's own fit_policy_frame, so the existing guard should fire
    # unchanged.
    model = FrequencySeverityModel(
        frequency_model=GBMPricingModel(family="gamma"),  # wrong family for frequency
        severity_model=GLMPricingModel(family="gamma"),
    )
    with pytest.raises(ValueError, match="needs family"):
        model.fit_policy_frame(pf)


# --- 3. Model-agnosticism at the curves/evaluation layer -----------------------


def test_curves_and_evaluation_run_unchanged_against_a_composed_model(pf):
    model = FrequencySeverityModel(
        frequency_model=GLMPricingModel(family="poisson"),
        severity_model=GLMPricingModel(family="gamma"),
    ).fit_policy_frame(pf)
    y_pred = model.predict_policy_frame(pf)
    feature = next(iter(pf.policy_schema.feature_roles))

    curve = one_way_curve(pf, feature, "pure_premium", y_pred)
    assert curve["exposure"].sum() == pytest.approx(pf.total_exposure)

    table = relativity_table(pf, "pure_premium", y_pred)
    assert len(table) > 0

    chart = lift_chart(pf, "pure_premium", y_pred, n_deciles=10)
    assert len(chart) == 10
    assert chart["exposure"].sum() == pytest.approx(pf.total_exposure)

    gini = gini_index(pf, "pure_premium", y_pred)
    assert -1.0 <= gini <= 1.0


def test_composed_and_direct_tweedie_gini_are_directly_comparable(pf):
    composed = FrequencySeverityModel(
        frequency_model=GLMPricingModel(family="poisson"),
        severity_model=GLMPricingModel(family="gamma"),
    ).fit_policy_frame(pf)
    direct = GLMPricingModel(family="tweedie", power=1.5).fit_policy_frame(pf, "pure_premium")

    gini_composed = gini_index(pf, "pure_premium", composed.predict_policy_frame(pf))
    gini_direct = gini_index(pf, "pure_premium", direct.predict_policy_frame(pf))

    assert np.isfinite(gini_composed)
    assert np.isfinite(gini_direct)


def test_peril_filter_is_applied_to_both_legs(pf):
    perils = pf.available_perils
    if not perils:
        pytest.skip("synthetic book has no peril_col configured")
    peril = perils[0]
    model = FrequencySeverityModel(
        frequency_model=GLMPricingModel(family="poisson"),
        severity_model=GLMPricingModel(family="gamma"),
    ).fit_policy_frame(pf, peril=peril)
    assert model.frequency_model.peril_ == peril
    assert model.severity_model.peril_ == peril
    y_pred = model.predict_policy_frame(pf)
    assert len(y_pred) == len(pf.pure_premium_view(peril=peril))
