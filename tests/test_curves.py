"""Golden-value tests for insurabench.curves (build order step 3).

Uses small, fully hand-computable fixtures (not the noisy synthetic
generator) so expected values can be checked by hand rather than by
re-running the code under test -- this is what makes these tests catch
silent wrong-answer bugs in exposure weighting / binning / normalization,
per the build order's explicit ask for golden-value tests here.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from insurabench.curves.one_way import one_way_curve
from insurabench.curves.relativity_table import relativity_table

Z_95 = norm.ppf(0.975)

# toy_frequency_pf is defined in tests/conftest.py (shared with
# tests/test_evaluation.py).


def test_frequency_view_matches_fixture_design(toy_frequency_pf):
    # Sanity check on the fixture itself, in policies' row order (p1..p4).
    view = toy_frequency_pf.frequency_view()
    assert view["claim_count"].tolist() == [1, 3, 0, 2]
    assert view["exposure"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_one_way_curve_exposure_weighted_actual_and_fitted(toy_frequency_pf):
    # Predicted counts (not rates) aligned to policies' row order p1..p4,
    # chosen distinctly from the actual counts so fitted != observed.
    y_pred = np.array([1.0, 2.0, 0.5, 1.5])

    curve = one_way_curve(toy_frequency_pf, "region", "frequency", y_pred, confidence=0.95)
    curve = curve.set_index("level")

    # region A: numerator sum=4, weight sum=2 -> observed=2.0
    #           predicted_numerator sum=1+2=3 -> fitted=1.5
    assert curve.loc["A", "n_obs"] == 2
    assert curve.loc["A", "exposure"] == pytest.approx(2.0)
    assert curve.loc["A", "observed"] == pytest.approx(2.0)
    assert curve.loc["A", "fitted"] == pytest.approx(1.5)

    # region B: numerator sum=2, weight sum=4 -> observed=0.5
    #           predicted_numerator sum=0.5+1.5=2 -> fitted=0.5
    assert curve.loc["B", "n_obs"] == 2
    assert curve.loc["B", "exposure"] == pytest.approx(4.0)
    assert curve.loc["B", "observed"] == pytest.approx(0.5)
    assert curve.loc["B", "fitted"] == pytest.approx(0.5)

    # Confidence bands: reliability-weighted SE, hand-computed (see module
    # docstring for the formula). Region A: ratio=[1,3], weight=[1,1],
    # weighted_var=avg([(1-2)^2,(3-2)^2])=1.0, n_eff=(2^2)/(1^2+1^2)=2,
    # se=sqrt(1.0/2).
    se_a = np.sqrt(1.0 / 2)
    assert curve.loc["A", "observed_lower"] == pytest.approx(2.0 - Z_95 * se_a)
    assert curve.loc["A", "observed_upper"] == pytest.approx(2.0 + Z_95 * se_a)

    # Region B: ratio=[0,1], weight=[2,2], weighted_var=avg([(0-0.5)^2,(1-0.5)^2],
    # weights=[2,2])=0.25, n_eff=(4^2)/(2^2+2^2)=2, se=sqrt(0.25/2).
    se_b = np.sqrt(0.25 / 2)
    assert curve.loc["B", "observed_lower"] == pytest.approx(0.5 - Z_95 * se_b)
    assert curve.loc["B", "observed_upper"] == pytest.approx(0.5 + Z_95 * se_b)


def test_one_way_curve_unknown_feature_raises(toy_frequency_pf):
    with pytest.raises(ValueError, match="not a declared feature"):
        one_way_curve(toy_frequency_pf, "not_a_column", "frequency", np.zeros(4))


def test_one_way_curve_mismatched_y_pred_length_raises(toy_frequency_pf):
    with pytest.raises(ValueError, match="row.s. but the"):
        one_way_curve(toy_frequency_pf, "region", "frequency", np.zeros(3))


def test_relativity_table_normalizes_to_max_exposure_base(toy_frequency_pf):
    y_pred = np.array([1.0, 2.0, 0.5, 1.5])
    table = relativity_table(toy_frequency_pf, "frequency", y_pred, features=["region"])
    table = table.set_index("level")

    # region B has more exposure (4 > 2) so it's the base: relativity 1.0.
    # region A's fitted (1.5) relative to B's fitted (0.5) -> 3.0.
    assert table.loc["B", "relativity"] == pytest.approx(1.0)
    assert table.loc["A", "relativity"] == pytest.approx(3.0)


def test_relativity_table_first_base(toy_frequency_pf):
    y_pred = np.array([1.0, 2.0, 0.5, 1.5])
    # Categorical levels come back sorted by descending exposure from
    # one_way_curve, so "first" here is region B (exposure 4), same base
    # as max_exposure for this fixture -- included as a regression check
    # that the "first" option is wired through at all, not a claim that
    # it always agrees with "max_exposure".
    table = relativity_table(
        toy_frequency_pf, "frequency", y_pred, features=["region"], base="first"
    ).set_index("level")
    assert table.loc["B", "relativity"] == pytest.approx(1.0)
    assert table.loc["A", "relativity"] == pytest.approx(3.0)


def test_relativity_table_unknown_base_raises(toy_frequency_pf):
    with pytest.raises(ValueError, match="Unknown base"):
        relativity_table(
            toy_frequency_pf, "frequency", np.zeros(4), features=["region"], base="nonsense"
        )


# --- two_way_curve ---------------------------------------------------------

from insurabench.curves.two_way import two_way_curve


def test_two_way_curve_exposure_weighted_grid(toy_two_feature_pf):
    # p1..p4 (exposure [1,1,2,2]) each occupy a distinct (region, vtype)
    # cell, so each output row is a single policy-period -- fully
    # hand-computable without any binning/grouping ambiguity.
    y_pred = np.array([1.0, 2.0, 0.5, 1.5])

    curve = two_way_curve(toy_two_feature_pf, "region", "vehicle_type", "frequency", y_pred)
    curve = curve.set_index(["level_x", "level_y"])

    assert curve.loc[("A", "X"), "n_obs"] == 1
    assert curve.loc[("A", "X"), "exposure"] == pytest.approx(1.0)
    assert curve.loc[("A", "X"), "observed"] == pytest.approx(1.0)  # p1: 1/1
    assert curve.loc[("A", "X"), "fitted"] == pytest.approx(1.0)  # p1: 1.0/1

    assert curve.loc[("A", "Y"), "observed"] == pytest.approx(3.0)  # p2: 3/1
    assert curve.loc[("A", "Y"), "fitted"] == pytest.approx(2.0)  # p2: 2.0/1

    assert curve.loc[("B", "X"), "exposure"] == pytest.approx(2.0)
    assert curve.loc[("B", "X"), "observed"] == pytest.approx(0.0)  # p3: 0/2
    assert curve.loc[("B", "X"), "fitted"] == pytest.approx(0.25)  # p3: 0.5/2

    assert curve.loc[("B", "Y"), "observed"] == pytest.approx(1.0)  # p4: 2/2
    assert curve.loc[("B", "Y"), "fitted"] == pytest.approx(0.75)  # p4: 1.5/2


def test_two_way_curve_same_feature_twice_raises(toy_two_feature_pf):
    with pytest.raises(ValueError, match="must differ"):
        two_way_curve(toy_two_feature_pf, "region", "region", "frequency", np.zeros(4))


def test_two_way_curve_unknown_feature_raises(toy_two_feature_pf):
    with pytest.raises(ValueError, match="not a declared feature"):
        two_way_curve(toy_two_feature_pf, "region", "not_a_column", "frequency", np.zeros(4))


# --- partial_dependence -----------------------------------------------------

from insurabench.curves.partial_dependence import partial_dependence


class _StubRateModel:
    """A minimal PricingModel-shaped stand-in: predicts a fixed rate per
    ``region`` level, scaled to the count scale by ``offset`` exactly the
    way ``GLMPricingModel`` does for target='frequency'. Lets
    ``partial_dependence`` be tested without depending on glum being
    installed, and gives an exactly hand-computable answer.
    """

    def __init__(self, rates: dict[str, float]):
        self.rates = rates

    def predict(self, X, *, offset=None):
        base_rate = X["region"].map(self.rates).to_numpy(dtype=float)
        if offset is None:
            return base_rate
        return base_rate * np.exp(np.asarray(offset))


def test_partial_dependence_categorical_recovers_exact_rate(toy_frequency_pf):
    # Every row's region is overwritten with each grid value in turn, so
    # the model's per-row predicted *rate* is the same constant for every
    # row regardless of exposure -- the exposure-weighted average must
    # therefore equal that constant exactly, for any weight distribution.
    model = _StubRateModel(rates={"A": 1.0, "B": 3.0})
    pdp = partial_dependence(toy_frequency_pf, "region", "frequency", model)
    pdp = pdp.set_index("level")

    assert pdp.loc["A", "partial_dependence"] == pytest.approx(1.0)
    assert pdp.loc["B", "partial_dependence"] == pytest.approx(3.0)


def test_partial_dependence_unknown_feature_raises(toy_frequency_pf):
    with pytest.raises(ValueError, match="not a declared feature"):
        partial_dependence(toy_frequency_pf, "not_a_column", "frequency", _StubRateModel({}))
