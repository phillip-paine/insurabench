from __future__ import annotations

import numpy as np
import pytest

from insurabench.data.policy_frame import PolicyFrame
from insurabench.model_selection import train_test_split_policy_frame
from insurabench.models.glm import GLMPricingModel, _resolve_family


@pytest.fixture
def big_book():
    """A larger synthetic book -- small fixtures make Tweedie/Poisson fits
    on near-all-zero claim data numerically unstable, not wrong."""
    from insurabench.data.datasets import make_synthetic_two_table

    return make_synthetic_two_table(n_policies=2000, renewals=True, seed=7)


@pytest.fixture
def big_pf(big_book):
    return PolicyFrame.from_tables(
        big_book.policies, big_book.claims, big_book.policy_schema, big_book.claims_schema,
        verbose=False,
    )


def test_tweedie_without_power_raises():
    with pytest.raises(ValueError, match="power"):
        _resolve_family("tweedie", None)


def test_power_with_non_tweedie_family_raises():
    with pytest.raises(ValueError, match="power"):
        _resolve_family("poisson", 1.5)


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="Unknown family"):
        _resolve_family("bogus", None)


def test_pure_premium_fit_predict_on_holdout(big_pf):
    train, test = train_test_split_policy_frame(big_pf, test_size=0.25, seed=0)

    model = GLMPricingModel(family="tweedie", power=1.5, alpha=0.01)
    model.fit_policy_frame(train, target="pure_premium")

    preds = model.predict_policy_frame(test)
    assert len(preds) == len(test.policies)
    assert np.all(np.isfinite(preds))
    assert (preds >= 0).all()  # log-link Tweedie mean must be non-negative

    d2 = model.score_policy_frame(test)
    assert np.isfinite(d2)


def test_frequency_fit_predict_on_holdout(big_pf):
    train, test = train_test_split_policy_frame(big_pf, test_size=0.25, seed=1)

    model = GLMPricingModel(family="poisson", alpha=0.01)
    model.fit_policy_frame(train, target="frequency")

    preds = model.predict_policy_frame(test)
    assert len(preds) == len(test.policies)
    assert (preds >= 0).all()


def test_severity_fit_predict_on_holdout(big_pf):
    train, test = train_test_split_policy_frame(big_pf, test_size=0.25, seed=2)

    model = GLMPricingModel(family="gamma", alpha=0.01)
    model.fit_policy_frame(train, target="severity")

    if test.n_claims == 0:
        pytest.skip("no claims in this holdout split for this seed")

    preds = model.predict_policy_frame(test)
    assert len(preds) == test.n_claims
    assert (preds > 0).all()  # Gamma mean must be strictly positive


def test_relativities_shape_and_log_link(big_pf):
    model = GLMPricingModel(family="tweedie", power=1.5)
    model.fit_policy_frame(big_pf, target="pure_premium")
    rel = model.relativities()

    assert "feature" in rel.columns
    assert "coefficient" in rel.columns
    assert "relativity" in rel.columns
    assert (rel["feature"] == "(intercept)").sum() == 1
    # multiplicative relativity must be strictly positive under a log link
    assert (rel["relativity"] > 0).all()
    np.testing.assert_allclose(rel["relativity"], np.exp(rel["coefficient"]))


def test_peril_specific_fit(big_pf):
    perils = big_pf.available_perils
    if not perils:
        pytest.skip("no claims generated for this seed")
    model = GLMPricingModel(family="tweedie", power=1.5)
    model.fit_policy_frame(big_pf, target="pure_premium", peril=perils[0])
    preds = model.predict_policy_frame(big_pf)
    assert len(preds) == big_pf.n_policies


def test_predict_before_fit_raises(big_pf):
    model = GLMPricingModel(family="tweedie", power=1.5)
    with pytest.raises(RuntimeError):
        model.predict_policy_frame(big_pf)


def test_categorical_feature_encoding_matches_between_fit_and_predict():
    """A regression-style guard: fitting on one book and predicting on a
    disjoint book with the same categories (but not the exact same rows)
    must not error over dtype/encoding mismatches."""
    from insurabench.data.datasets import make_synthetic_two_table

    book_a = make_synthetic_two_table(n_policies=300, renewals=False, seed=10)
    book_b = make_synthetic_two_table(n_policies=300, renewals=False, seed=11)

    pf_a = PolicyFrame.from_tables(
        book_a.policies, book_a.claims, book_a.policy_schema, book_a.claims_schema, verbose=False
    )
    pf_b = PolicyFrame.from_tables(
        book_b.policies, book_b.claims, book_b.policy_schema, book_b.claims_schema, verbose=False
    )
    model = GLMPricingModel(family="tweedie", power=1.5)
    model.fit_policy_frame(pf_a, target="pure_premium")
    preds = model.predict_policy_frame(pf_b)
    assert len(preds) == pf_b.n_policies
