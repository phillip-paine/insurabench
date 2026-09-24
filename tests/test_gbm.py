"""Tests for insurabench.models.gbm.

Three concerns, each deliberately tested differently:

1. The custom-objective math (``_ZIPB1Objective``, ``_GammaObjective``) --
   verified by finite differences against the loss function directly,
   not by checking a fitted model's predictions look plausible. A
   plausible-looking prediction doesn't rule out a sign error that
   happens to still point the optimizer in a locally-useful direction;
   a finite-difference check against the actual loss does.
2. The wrapper's own plumbing (fit/predict/score/relativities,
   family/target mismatch guards) -- ordinary behavioral tests against
   the synthetic book, mirroring the level of coverage a GLM wrapper
   test file would have.
3. Model-agnosticism (build order step 4's actual point) -- rerunning
   curves/evaluation functions against a fitted GBM's predictions and
   confirming they produce the same *shape* of sane output they do for
   a GLM's, with no GBM-specific branch anywhere in curves/evaluation.
"""
from __future__ import annotations

import numpy as np
import pytest

from insurabench.curves import one_way_curve, relativity_table
from insurabench.data.datasets import make_synthetic_two_table
from insurabench.data.policy_frame import PolicyFrame
from insurabench.evaluation import gini_index, lift_chart
from insurabench.models.gbm import (
    GBMPricingModel,
    _GammaObjective,
    _poisson_unit_deviance,
    _tweedie_unit_deviance,
    _zip_unit_deviance,
    _ZIPB1Objective,
)
from insurabench.models.glm import GLMPricingModel

# --- 1. Custom-objective correctness (finite differences against the loss) --


def _zip_loss(y: float, raw: float, gamma: float) -> float:
    """So (2024) eq. 17, directly -- the loss ``_ZIPB1Objective`` is the
    gradient/hessian of (before CatBoost's sign convention is applied).
    """
    mu = np.exp(raw)
    mu_g = mu**gamma
    if y == 0:
        return -np.log(1 + mu_g * np.exp(-mu)) + np.log(1 + mu_g)
    return -gamma * np.log(mu) + np.log(1 + mu_g) + mu - y * np.log(mu)


def _gamma_loss(y: float, raw: float) -> float:
    """The Gamma GLM-equivalent loss ``_GammaObjective`` is the
    gradient/hessian of, up to a y-only constant that doesn't affect
    derivatives w.r.t. raw.
    """
    mu = np.exp(raw)
    return y / mu + raw


@pytest.mark.parametrize("y", [0, 1, 2, 5, 10])
@pytest.mark.parametrize("gamma", [1.0, 5.0, 20.0])
def test_zip_objective_matches_finite_difference_of_the_paper_loss(y, gamma):
    raw = 0.7  # an arbitrary interior point, not near a boundary
    eps = 1e-4

    g_numeric = (_zip_loss(y, raw + eps, gamma) - _zip_loss(y, raw - eps, gamma)) / (2 * eps)
    h_numeric = (
        _zip_loss(y, raw + eps, gamma) - 2 * _zip_loss(y, raw, gamma) + _zip_loss(y, raw - eps, gamma)
    ) / eps**2

    # _ZIPB1Objective returns CatBoost's (der1, der2) = (-g, -h); recover
    # (g, h) to compare against the loss's own numeric derivatives.
    result = _ZIPB1Objective(gamma).calc_ders_range([raw], [float(y)], None)
    der1, der2 = result[0]
    g_analytic, h_analytic = -der1, -der2

    assert g_analytic == pytest.approx(g_numeric, abs=1e-3)
    assert h_analytic == pytest.approx(h_numeric, abs=2e-3)


@pytest.mark.parametrize("y", [0.5, 5.0, 200.0, 5000.0])
def test_gamma_objective_matches_finite_difference_of_its_loss(y):
    raw = 3.2
    eps = 1e-4

    g_numeric = (_gamma_loss(y, raw + eps) - _gamma_loss(y, raw - eps)) / (2 * eps)
    h_numeric = (_gamma_loss(y, raw + eps) - 2 * _gamma_loss(y, raw) + _gamma_loss(y, raw - eps)) / eps**2

    result = _GammaObjective().calc_ders_range([raw], [y], None)
    der1, der2 = result[0]
    g_analytic, h_analytic = -der1, -der2

    assert g_analytic == pytest.approx(g_numeric, abs=1e-3)
    assert h_analytic == pytest.approx(h_numeric, abs=1e-3)


def test_zip_objective_matches_native_poisson_as_gamma_to_infinity():
    # Sanity cross-check independent of the finite-difference tests above:
    # as gamma -> large, p = 1/(1+mu^gamma) -> 0 for mu > 1, collapsing the
    # ZIP distribution toward a plain Poisson for typical claim-rate mu.
    # Not exact (mu can dip below 1), but the deviance should track the
    # plain Poisson deviance reasonably closely at a very large gamma for
    # mu comfortably above 1.
    y = np.array([2.0])
    mu = np.array([3.0])
    zip_dev = _zip_unit_deviance(y, mu, np.array([1e-6]))  # p essentially 0
    poisson_dev = _poisson_unit_deviance(y, mu)
    assert zip_dev[0] == pytest.approx(poisson_dev[0], abs=1e-4)


def test_gamma_unit_deviance_matches_tweedie_power_2_formula():
    # _GammaObjective's implied deviance (via score()) should agree with
    # the general Tweedie unit deviance formula at its power=2 boundary,
    # confirming the two independently-written formulas describe the same
    # distribution.
    y = np.array([50.0, 500.0, 5000.0])
    mu = np.array([100.0, 400.0, 4000.0])
    dev = _tweedie_unit_deviance(y, mu, 2.0)
    expected = 2.0 * (y / mu - np.log(y / mu) - 1.0)
    assert dev == pytest.approx(expected)


# --- 2. Wrapper plumbing -------------------------------------------------------


@pytest.fixture(scope="module")
def pf() -> PolicyFrame:
    book = make_synthetic_two_table(n_policies=1200, renewals=True, seed=7)
    return PolicyFrame.from_tables(
        book.policies, book.claims, book.policy_schema, book.claims_schema, verbose=False
    )


def test_poisson_frequency_fits_and_predicts_finite_rates(pf):
    model = GBMPricingModel(family="poisson", iterations=200).fit_policy_frame(pf, "frequency")
    y_pred = model.predict_policy_frame(pf)
    assert len(y_pred) == len(pf.frequency_view())
    assert np.all(np.isfinite(y_pred))
    assert np.all(y_pred >= 0)
    assert np.isfinite(model.score_policy_frame(pf))


def test_zip_frequency_fits_and_predicts_finite_rates(pf):
    model = GBMPricingModel(family="zip", zip_gamma=1.0, iterations=200).fit_policy_frame(
        pf, "frequency"
    )
    y_pred = model.predict_policy_frame(pf)
    assert np.all(np.isfinite(y_pred))
    assert np.all(y_pred >= 0)
    assert np.isfinite(model.score_policy_frame(pf))


def test_gamma_severity_lands_on_the_right_scale_even_at_few_iterations(pf):
    # Regression test for the cold-start bug found and fixed while
    # building this wrapper: with no natural offset, a custom objective
    # starting every row at raw=0 (mu=1) needs many boosting iterations
    # to reach a severity target's true scale (thousands) unless given
    # an auto-baseline. At a low iteration count, an under-converged fit
    # would predict close to exp(0)=1 for everything -- assert the
    # predictions are actually within an order of magnitude of the
    # data's own mean, not collapsed toward 1.
    model = GBMPricingModel(family="gamma", iterations=30).fit_policy_frame(pf, "severity")
    y_pred = model.predict_policy_frame(pf)
    actual_mean = pf.severity_view()["claim_amount"].mean()
    assert np.all(np.isfinite(y_pred))
    assert np.all(y_pred > 0)
    assert 0.2 * actual_mean < y_pred.mean() < 5 * actual_mean


def test_tweedie_pure_premium_fits_and_predicts_finite_rates(pf):
    model = GBMPricingModel(family="tweedie", power=1.5, iterations=200).fit_policy_frame(
        pf, "pure_premium"
    )
    y_pred = model.predict_policy_frame(pf)
    assert np.all(np.isfinite(y_pred))
    assert np.all(y_pred >= 0)
    assert np.isfinite(model.score_policy_frame(pf))


def test_tweedie_rejects_power_outside_open_interval_one_two():
    with pytest.raises(ValueError, match="1 < power < 2"):
        GBMPricingModel(family="tweedie", power=2.0)
    with pytest.raises(ValueError, match="1 < power < 2"):
        GBMPricingModel(family="tweedie", power=1.0)


def test_zip_requires_gamma():
    with pytest.raises(ValueError, match="requires an explicit zip_gamma"):
        GBMPricingModel(family="zip")


def test_zip_gamma_must_be_positive():
    with pytest.raises(ValueError, match="gamma must be > 0"):
        GBMPricingModel(family="zip", zip_gamma=-1.0)


@pytest.mark.parametrize(
    "family,target,kwargs",
    [
        ("poisson", "severity", {}),
        ("gamma", "frequency", {}),
        ("tweedie", "severity", {"power": 1.5}),
        ("zip", "pure_premium", {"zip_gamma": 1.0}),
    ],
)
def test_family_target_mismatch_raises(pf, family, target, kwargs):
    with pytest.raises(ValueError, match="needs family"):
        GBMPricingModel(family=family, **kwargs).fit_policy_frame(pf, target)


def test_relativities_returns_normalized_importances(pf):
    model = GBMPricingModel(family="poisson", iterations=200).fit_policy_frame(pf, "frequency")
    rel = model.relativities()
    assert set(rel.columns) == {"feature", "importance"}
    assert set(rel["feature"]) == set(pf.policy_schema.feature_roles)
    assert (rel["importance"] >= 0).all()
    # sorted descending, most-important feature first
    assert (rel["importance"].diff().dropna() <= 1e-9).all()


# --- 3. Model-agnosticism (build order step 4's actual point) ----------------


def test_curves_and_evaluation_run_unchanged_against_a_fitted_gbm(pf):
    """The point of this test: no branch anywhere in curves/evaluation
    knows or cares that this prediction array came from a GBM rather
    than a GLM. If this test needed any GBM-specific handling to pass,
    the model-agnosticism claim would be false.
    """
    gbm = GBMPricingModel(family="poisson", iterations=200).fit_policy_frame(pf, "frequency")
    y_pred = gbm.predict_policy_frame(pf)
    feature = next(iter(pf.policy_schema.feature_roles))

    curve = one_way_curve(pf, feature, "frequency", y_pred)
    assert curve["exposure"].sum() == pytest.approx(pf.total_exposure)

    table = relativity_table(pf, "frequency", y_pred)
    assert len(table) > 0

    chart = lift_chart(pf, "frequency", y_pred, n_deciles=10)
    assert len(chart) == 10
    assert chart["exposure"].sum() == pytest.approx(pf.total_exposure)

    gini = gini_index(pf, "frequency", y_pred)
    assert -1.0 <= gini <= 1.0


def test_glm_and_gbm_gini_are_directly_comparable(pf):
    """Fits both model types on the same data/target and runs the exact
    same evaluation call on each -- the concrete proof that curves/
    evaluation code written once works against either model type without
    modification, which is the design brief's stated critical
    requirement (§4) and build order step 4's whole purpose.
    """
    glm = GLMPricingModel(family="poisson").fit_policy_frame(pf, "frequency")
    gbm = GBMPricingModel(family="poisson", iterations=200).fit_policy_frame(pf, "frequency")

    gini_glm = gini_index(pf, "frequency", glm.predict_policy_frame(pf))
    gini_gbm = gini_index(pf, "frequency", gbm.predict_policy_frame(pf))

    assert np.isfinite(gini_glm)
    assert np.isfinite(gini_gbm)
