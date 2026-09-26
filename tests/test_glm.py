"""Tests for insurabench.models.glm.

Focused specifically on ``GLMPricingModel.information_criteria`` (added
alongside the reporting layer, for ``reporting.model_card``) -- the rest
of ``GLMPricingModel`` has always had its only direct coverage via
``tests/test_end_to_end.py``, still flagged as a real gap in the progress
notes; not closed here, out of scope for this change.
"""
from __future__ import annotations

import numpy as np
import pytest

from insurabench.data.datasets import make_synthetic_two_table
from insurabench.data.policy_frame import PolicyFrame
from insurabench.models.base import build_design
from insurabench.models.glm import GLMPricingModel


@pytest.fixture(scope="module")
def pf() -> PolicyFrame:
    book = make_synthetic_two_table(n_policies=800, renewals=True, seed=5)
    return PolicyFrame.from_tables(
        book.policies, book.claims, book.policy_schema, book.claims_schema, verbose=False
    )


def test_information_criteria_matches_manual_offset_aware_computation(pf):
    """Regression test for the glum footgun found while building the
    reporting layer: glum's own ``.aic()``/``.bic()`` have no ``offset``
    parameter and silently evaluate log-likelihood at the wrong mu for an
    offset-fit (frequency) model. Recompute AIC by hand, offset included,
    and check ``information_criteria`` matches it -- not glum's own
    ``.aic()``.
    """
    model = GLMPricingModel(family="poisson").fit_policy_frame(pf, "frequency")
    X, y, _, offset = build_design(pf, "frequency", peril=None, coverage=None)

    aic, bic = model.information_criteria(X, y, offset=offset)

    mu = model.predict(X, offset=offset)
    ll = model._estimator.family_instance.log_likelihood(np.asarray(y, dtype=float), mu)
    coef = np.asarray(model._estimator.coef_)
    ddof = int(np.sum(np.abs(coef) > np.finfo(coef.dtype).eps))
    k_params = ddof + int(model.fit_intercept)
    nobs = len(y)
    expected_aic = -2 * ll + 2 * k_params
    expected_bic = -2 * ll + np.log(nobs) * k_params

    assert aic == pytest.approx(expected_aic)
    assert bic == pytest.approx(expected_bic)


def test_information_criteria_differs_from_glums_own_broken_aic(pf):
    """The actual bug this guards against: calling glum's native
    ``.aic()`` without an offset (its signature has none) silently
    computes a materially different, wrong number for an offset-fit
    model. If this test ever starts failing because the two numbers
    match, that means glum fixed the underlying method upstream -- worth
    investigating, not just deleting the test.
    """
    model = GLMPricingModel(family="poisson").fit_policy_frame(pf, "frequency")
    X, y, _, offset = build_design(pf, "frequency", peril=None, coverage=None)

    aic_correct, _ = model.information_criteria(X, y, offset=offset)
    aic_native_broken = model._estimator.aic(X, y)  # no offset param exists to pass

    assert aic_correct != pytest.approx(aic_native_broken, rel=1e-6)


def test_information_criteria_policy_frame_matches_low_level_call(pf):
    model = GLMPricingModel(family="poisson").fit_policy_frame(pf, "frequency")
    X, y, sw, offset = build_design(pf, "frequency", peril=None, coverage=None)

    aic1, bic1 = model.information_criteria_policy_frame(pf)
    aic2, bic2 = model.information_criteria(X, y, sample_weight=sw, offset=offset)

    assert aic1 == pytest.approx(aic2)
    assert bic1 == pytest.approx(bic2)


def test_information_criteria_before_fit_raises(pf):
    model = GLMPricingModel(family="poisson")
    with pytest.raises(RuntimeError, match="fit_policy_frame"):
        model.information_criteria_policy_frame(pf)


def test_information_criteria_finite_for_severity_and_pure_premium(pf):
    # Neither of these targets is offset-fit, so unlike frequency there's
    # no bug to regress-test here -- just confirming the method works for
    # every target, not just the one with the offset footgun.
    severity_model = GLMPricingModel(family="gamma").fit_policy_frame(pf, "severity")
    aic, bic = severity_model.information_criteria_policy_frame(pf)
    assert np.isfinite(aic)
    assert np.isfinite(bic)

    pp_model = GLMPricingModel(family="tweedie", power=1.5).fit_policy_frame(pf, "pure_premium")
    aic, bic = pp_model.information_criteria_policy_frame(pf)
    assert np.isfinite(aic)
    assert np.isfinite(bic)


def test_information_criteria_warns_under_ridge_regularization(pf):
    model = GLMPricingModel(family="poisson", alpha=0.5, l1_ratio=0.0).fit_policy_frame(pf, "frequency")
    with pytest.warns(UserWarning, match="degrees of freedom"):
        model.information_criteria_policy_frame(pf)
