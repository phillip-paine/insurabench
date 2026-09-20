"""Golden-value tests for insurabench.evaluation (build order step 3).

Uses the same small, hand-computable fixtures as tests/test_curves.py
(defined in conftest.py) rather than the noisy synthetic generator, so
expected values are checked against independently hand-computed numbers,
not against a second call into the code under test.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from insurabench.evaluation.gini import gini_curve, gini_index
from insurabench.evaluation.lift import lift_chart


def test_lift_chart_deciles_by_cumulative_exposure(toy_frequency_pf):
    # Predicted counts aligned to policies' row order p1..p4 (exposure
    # [1,1,2,2]); predicted *rate* = y_pred/exposure = [1.0, 2.0, 0.25, 0.75].
    y_pred = np.array([1.0, 2.0, 0.5, 1.5])

    chart = lift_chart(toy_frequency_pf, "frequency", y_pred, n_deciles=2)
    chart = chart.set_index("decile")

    # Sorted ascending by predicted rate: p3(0.25,w=2), p4(0.75,w=2),
    # p1(1.0,w=1), p2(2.0,w=1). Total weight=6, so cumulative-exposure
    # share after each row is 2/6, 4/6, 5/6, 6/6 -> with n_deciles=2,
    # bucket = floor(share*2) capped at 1, +1:
    #   p3: floor(2/6*2)=0 -> bucket 1
    #   p4: floor(4/6*2)=1 -> bucket 2
    #   p1: floor(5/6*2)=1 -> bucket 2
    #   p2: floor(6/6*2)=2, capped to 1 -> bucket 2
    # bucket 1 = {p3}: numerator=0, weight=2, predicted_numerator=0.5
    #   -> actual=0/2=0.0, predicted=0.5/2=0.25
    # bucket 2 = {p4,p1,p2}: numerator=2+1+3=6, weight=2+1+1=4,
    #   predicted_numerator=1.5+1+2=4.5 -> actual=6/4=1.5, predicted=4.5/4=1.125
    assert chart.loc[1, "n_obs"] == 1
    assert chart.loc[1, "exposure"] == pytest.approx(2.0)
    assert chart.loc[1, "actual"] == pytest.approx(0.0)
    assert chart.loc[1, "predicted"] == pytest.approx(0.25)

    assert chart.loc[2, "n_obs"] == 3
    assert chart.loc[2, "exposure"] == pytest.approx(4.0)
    assert chart.loc[2, "actual"] == pytest.approx(1.5)
    assert chart.loc[2, "predicted"] == pytest.approx(1.125)


def test_lift_chart_n_deciles_validation(toy_frequency_pf):
    with pytest.raises(ValueError, match="n_deciles"):
        lift_chart(toy_frequency_pf, "frequency", np.zeros(4), n_deciles=0)


def test_gini_index_perfect_prediction_is_one(toy_frequency_pf):
    # predicted_numerator == numerator exactly (frequency: y_pred IS the
    # numerator scale directly) -> model curve == ideal curve by
    # construction, for any dataset, regardless of ties.
    perfect_y_pred = toy_frequency_pf.frequency_view()["claim_count"].to_numpy()
    assert gini_index(toy_frequency_pf, "frequency", perfect_y_pred) == pytest.approx(1.0)


def test_gini_index_anti_correlated_prediction_golden_value(toy_severity_pf):
    # claim_amount = [1, 3, 5] (already increasing); predict the exact
    # reverse order -> the worst possible ranking. Severity's weight is
    # always 1, so this is a plain (unweighted) Gini, hand-computable.
    y_pred = np.array([5.0, 3.0, 1.0])

    raw = gini_index(toy_severity_pf, "severity", y_pred, normalized=False)
    # Sorted ascending by predicted: claim order becomes [5, 3, 1] (amounts).
    # cum_exposure = [1/3, 2/3, 1], cum_loss = [5/9, 8/9, 1] (loss total=9).
    # Trapezoidal AUC against (0,0): 5/54 + 13/54 + 17/54 = 35/54.
    # raw_gini = 2*(35/54) - 1 = 8/27.
    assert raw == pytest.approx(float(Fraction(8, 27)))

    ideal = gini_index(toy_severity_pf, "severity", np.array([1.0, 3.0, 5.0]), normalized=False)
    # The ideal (correctly-ordered) prediction: cum_exposure=[1/3,2/3,1],
    # cum_loss=[1/9,4/9,1]. AUC = 1/54+5/54+13/54 = 19/54.
    # ideal_gini = 2*(19/54) - 1 = -8/27.
    assert ideal == pytest.approx(float(Fraction(-8, 27)))

    normalized = gini_index(toy_severity_pf, "severity", y_pred, normalized=True)
    assert normalized == pytest.approx(-1.0)


def test_gini_curve_endpoints(toy_frequency_pf):
    y_pred = np.array([1.0, 2.0, 0.5, 1.5])
    curve = gini_curve(toy_frequency_pf, "frequency", y_pred)
    assert curve["cum_exposure"].iloc[0] == pytest.approx(0.0)
    assert curve["cum_loss"].iloc[0] == pytest.approx(0.0)
    assert curve["cum_exposure"].iloc[-1] == pytest.approx(1.0)
    assert curve["cum_loss"].iloc[-1] == pytest.approx(1.0)


# --- double_lift_chart -------------------------------------------------------

from insurabench.evaluation.double_lift import double_lift_chart


def test_double_lift_chart_sorts_by_model_ratio(toy_frequency_pf):
    # Same fixture/weights as test_lift_chart_deciles_by_cumulative_exposure.
    # Predicted *counts*: model_a=[1,2,0.5,1.5] -> rate_a=[1.0,2.0,0.25,0.75];
    # model_b=[0.5,1,1,1] -> rate_b=[0.5,1.0,0.5,0.5].
    # ratio = rate_a/rate_b = [2.0, 2.0, 0.5, 1.5] for p1,p2,p3,p4.
    y_pred_a = np.array([1.0, 2.0, 0.5, 1.5])
    y_pred_b = np.array([0.5, 1.0, 1.0, 1.0])

    chart = double_lift_chart(
        toy_frequency_pf, "frequency", y_pred_a, y_pred_b, n_deciles=2
    ).set_index("decile")

    # Sorted ascending by ratio: p3(0.5,w=2), p4(1.5,w=2), p1(2.0,w=1),
    # p2(2.0,w=1) (tie broken by original/stable order). Total weight=6;
    # bucket = floor(cum_share*2) capped at 1, +1 (same convention as lift):
    #   p3: cum=2/6 -> bucket 1; p4: cum=4/6 -> bucket 2;
    #   p1: cum=5/6 -> bucket 2; p2: cum=6/6 -> bucket 2.
    # bucket 1 = {p3}: numerator=0, weight=2, pred_a_num=0.5, pred_b_num=1.0
    #   -> actual=0, model_a=0.25, model_b=0.5, ratio_a_over_b=0.5
    assert chart.loc[1, "n_obs"] == 1
    assert chart.loc[1, "exposure"] == pytest.approx(2.0)
    assert chart.loc[1, "actual"] == pytest.approx(0.0)
    assert chart.loc[1, "model_a"] == pytest.approx(0.25)
    assert chart.loc[1, "model_b"] == pytest.approx(0.5)
    assert chart.loc[1, "ratio_a_over_b"] == pytest.approx(0.5)

    # bucket 2 = {p4,p1,p2}: numerator=2+1+3=6, weight=2+1+1=4,
    #   pred_a_num=1.5+1+2=4.5, pred_b_num=1.0+0.5+1.0=2.5
    #   -> actual=1.5, model_a=1.125, model_b=0.625
    #   ratio_a_over_b weighted avg (weights 2,1,1 of ratios 1.5,2.0,2.0)
    #   = (1.5*2 + 2.0*1 + 2.0*1) / 4 = 7/4 = 1.75
    assert chart.loc[2, "n_obs"] == 3
    assert chart.loc[2, "exposure"] == pytest.approx(4.0)
    assert chart.loc[2, "actual"] == pytest.approx(1.5)
    assert chart.loc[2, "model_a"] == pytest.approx(1.125)
    assert chart.loc[2, "model_b"] == pytest.approx(0.625)
    assert chart.loc[2, "ratio_a_over_b"] == pytest.approx(1.75)


def test_double_lift_chart_length_mismatch_raises(toy_frequency_pf):
    # y_pred_b built for the wrong number of rows -- caught by the same
    # per-array length check every build_target_frame call does.
    with pytest.raises(ValueError, match="row.s. but the"):
        double_lift_chart(
            toy_frequency_pf,
            "frequency",
            np.array([1.0, 2.0, 0.5, 1.5]),
            np.array([1.0, 2.0, 0.5]),
        )


def test_double_lift_chart_identical_predictions_agree_everywhere(toy_frequency_pf):
    y_pred = np.array([1.0, 2.0, 0.5, 1.5])
    chart = double_lift_chart(toy_frequency_pf, "frequency", y_pred, y_pred, n_deciles=2)
    assert np.allclose(chart["model_a"], chart["model_b"])
    assert np.allclose(chart["ratio_a_over_b"], 1.0)


# --- calibration -------------------------------------------------------------

from insurabench.evaluation.calibration import calibration_index, calibration_table


def test_calibration_table_matches_lift_actual_and_fitted(toy_frequency_pf):
    # Same setup/buckets as test_lift_chart_deciles_by_cumulative_exposure:
    # observed/fitted here must equal that test's actual/predicted exactly,
    # since both bucket by the same exposure_deciles convention.
    y_pred = np.array([1.0, 2.0, 0.5, 1.5])
    table = calibration_table(toy_frequency_pf, "frequency", y_pred, n_bins=2).set_index("decile")

    assert table.loc[1, "observed"] == pytest.approx(0.0)
    assert table.loc[1, "fitted"] == pytest.approx(0.25)
    assert table.loc[2, "observed"] == pytest.approx(1.5)
    assert table.loc[2, "fitted"] == pytest.approx(1.125)

    # bucket 1 = {p3} only: a single-row bucket has zero within-bucket
    # variance around its own mean -> se=0 -> z is undefined (nan).
    assert np.isnan(table.loc[1, "z"])
    assert table.loc[1, "significant"] == False

    # bucket 2 = {p4,p1,p2}, weight=[2,1,1], ratio=[1.0,1.0,3.0]:
    # weighted_var = avg((ratio-1.5)^2, weights=[2,1,1])
    #   = (0.25*2 + 0.25*1 + 2.25*1) / 4 = 3.0/4 = 0.75
    # n_eff = 4^2 / (2^2+1^2+1^2) = 16/6
    # se = sqrt(0.75 / (16/6)) = sqrt(0.28125) = 0.5303300858...
    # z = (1.5 - 1.125) / se = 0.375 / 0.53033... = 0.7071067811... = 1/sqrt(2)
    se_2 = np.sqrt(0.75 / (16 / 6))
    assert table.loc[2, "standard_error"] == pytest.approx(se_2)
    assert table.loc[2, "z"] == pytest.approx(1 / np.sqrt(2))


def test_calibration_index_excludes_nan_buckets(toy_frequency_pf):
    y_pred = np.array([1.0, 2.0, 0.5, 1.5])
    # Only bucket 2 has a defined z (0.70710678...); bucket 1's nan is
    # excluded rather than counted as 0, so the index is exactly z**2.
    assert calibration_index(toy_frequency_pf, "frequency", y_pred, n_bins=2) == pytest.approx(0.5)


# --- stability ----------------------------------------------------------------

from insurabench.evaluation.stability import (
    bootstrap_relativities,
    split_relativities,
    stability_summary,
)


def test_bootstrap_relativities_base_level_always_exactly_one(toy_frequency_pf):
    # region B has more exposure (4 > 2) so it's always the fixed base
    # level (chosen once from the full data, not re-picked per draw) --
    # its own relativity to itself must be exactly 1.0 in every draw that
    # includes it at all, regardless of which rows got resampled.
    y_pred = np.array([1.0, 2.0, 0.5, 1.5])
    draws = bootstrap_relativities(
        toy_frequency_pf, "region", "frequency", y_pred, n_bootstrap=100, seed=0
    )
    b_draws = draws[draws["level"] == "B"]
    assert len(b_draws) > 0
    assert np.allclose(b_draws["relativity"], 1.0)

    summary = stability_summary(draws).set_index("level")
    assert summary.loc["B", "relativity_mean"] == pytest.approx(1.0)
    assert summary.loc["B", "relativity_range"] == pytest.approx(0.0)


def test_split_relativities_hand_computed(toy_frequency_pf):
    # split assigns p1(A)->0, p2(A)->1, p3(B)->0, p4(B)->1, so both
    # splits contain one row of each region -- fully hand-computable.
    y_pred = np.array([1.0, 2.0, 0.5, 1.5])
    split = np.array([0, 1, 0, 1])

    long = split_relativities(toy_frequency_pf, "region", "frequency", y_pred, split)
    long = long.set_index(["split", "level"])

    # split 0: fitted_A = p1's pred_num/weight = 1/1 = 1.0
    #          fitted_B = p3's pred_num/weight = 0.5/2 = 0.25
    #          base is B -> relativity_A = 1.0/0.25 = 4.0, relativity_B = 1.0
    assert long.loc[(0, "A"), "relativity"] == pytest.approx(4.0)
    assert long.loc[(0, "B"), "relativity"] == pytest.approx(1.0)

    # split 1: fitted_A = p2's pred_num/weight = 2/1 = 2.0
    #          fitted_B = p4's pred_num/weight = 1.5/2 = 0.75
    #          relativity_A = 2.0/0.75 = 8/3, relativity_B = 1.0
    assert long.loc[(1, "A"), "relativity"] == pytest.approx(8 / 3)
    assert long.loc[(1, "B"), "relativity"] == pytest.approx(1.0)

    summary = stability_summary(long.reset_index()).set_index("level")
    # level A: values [4.0, 8/3] -> mean=10/3, std (ddof=1) computed from
    # deviations of +/-2/3 -> var = (2*(2/3)^2)/1 = 8/9, std = sqrt(8/9).
    assert summary.loc["A", "relativity_mean"] == pytest.approx(10 / 3)
    assert summary.loc["A", "relativity_std"] == pytest.approx(np.sqrt(8 / 9))
    assert summary.loc["A", "relativity_range"] == pytest.approx(4.0 - 8 / 3)

    assert summary.loc["B", "relativity_mean"] == pytest.approx(1.0)
    assert summary.loc["B", "relativity_std"] == pytest.approx(0.0)
    assert summary.loc["B", "relativity_range"] == pytest.approx(0.0)


def test_split_relativities_length_mismatch_raises(toy_frequency_pf):
    with pytest.raises(ValueError, match="aligned to that same view"):
        split_relativities(
            toy_frequency_pf, "region", "frequency", np.zeros(4), split=np.array([0, 1])
        )


# --- rate_impact --------------------------------------------------------------

from insurabench.evaluation.rate_impact import (
    rate_change_by_level,
    rate_change_distribution,
)


def test_rate_change_by_level_reveals_split_hidden_by_aggregate(toy_frequency_pf):
    # Same fixture/predictions as the double_lift tests: new rate =
    # [1.0, 2.0, 0.25, 0.75], old rate = [0.5, 1.0, 0.5, 0.5] for
    # p1,p2,p3,p4. Row-level pct change: p1/p2 = +100%, p3 = -50%,
    # p4 = +50%.
    y_new = np.array([1.0, 2.0, 0.5, 1.5])
    y_old = np.array([0.5, 1.0, 1.0, 1.0])

    table = rate_change_by_level(toy_frequency_pf, "region", "frequency", y_new, y_old).set_index(
        "level"
    )

    # region A = {p1, p2}, both weight 1, both +100% -> aggregate and
    # individual agree perfectly here.
    assert table.loc["A", "old_rate"] == pytest.approx(0.75)  # (0.5+1.0)/2
    assert table.loc["A", "new_rate"] == pytest.approx(1.5)  # (1.0+2.0)/2
    assert table.loc["A", "pct_change"] == pytest.approx(1.0)
    assert table.loc["A", "pct_exposure_increasing"] == pytest.approx(1.0)
    assert table.loc["A", "pct_exposure_decreasing"] == pytest.approx(0.0)

    # region B = {p3 (-50%, weight 2), p4 (+50%, weight 2)}: the two
    # moves exactly cancel in aggregate (old_rate == new_rate == 0.5,
    # pct_change == 0%) even though every single policy-period in the
    # region moved -- this is exactly the case
    # pct_exposure_increasing/decreasing exists to surface.
    assert table.loc["B", "old_rate"] == pytest.approx(0.5)  # (1.0+1.0)/4
    assert table.loc["B", "new_rate"] == pytest.approx(0.5)  # (0.5+1.5)/4
    assert table.loc["B", "pct_change"] == pytest.approx(0.0)
    assert table.loc["B", "pct_exposure_increasing"] == pytest.approx(0.5)
    assert table.loc["B", "pct_exposure_decreasing"] == pytest.approx(0.5)


def test_rate_change_by_level_length_mismatch_raises(toy_frequency_pf):
    with pytest.raises(ValueError, match="row.s. but the"):
        rate_change_by_level(
            toy_frequency_pf,
            "region",
            "frequency",
            np.array([1.0, 2.0, 0.5, 1.5]),
            np.array([1.0, 2.0, 0.5]),
        )


def test_rate_change_distribution_hand_computed(toy_frequency_pf):
    # Row-level pct changes: p1=+100%, p2=+100%, p3=-50%, p4=+50%,
    # weights [1,1,2,2], total weight 6.
    y_new = np.array([1.0, 2.0, 0.5, 1.5])
    y_old = np.array([0.5, 1.0, 1.0, 1.0])

    table = rate_change_distribution(toy_frequency_pf, "frequency", y_new, y_old).set_index("band")

    # p3 (-50%, weight 2) falls in (-100%, -20%]
    assert table.loc["(-100%, -20%]", "exposure"] == pytest.approx(2.0)
    assert table.loc["(-100%, -20%]", "exposure_share"] == pytest.approx(2 / 6)
    # p1, p2, p4 (all >= +50%, weight 1+1+2=4) fall in (+20%, +inf]
    assert table.loc["(+20%, +inf]", "exposure"] == pytest.approx(4.0)
    assert table.loc["(+20%, +inf]", "exposure_share"] == pytest.approx(4 / 6)
    # every other band is empty
    other_bands = table.drop(index=["(-100%, -20%]", "(+20%, +inf]"])
    assert (other_bands["exposure"] == 0).all()
    # shares always sum to 1.0 across the whole table
    assert table["exposure_share"].sum() == pytest.approx(1.0)


def test_rate_change_distribution_reports_undefined_band(toy_frequency_pf):
    # Force p1's old rate to exactly 0 (undefined pct change) by giving
    # it a 0 predicted count in y_old.
    y_new = np.array([1.0, 2.0, 0.5, 1.5])
    y_old = np.array([0.0, 1.0, 1.0, 1.0])

    table = rate_change_distribution(toy_frequency_pf, "frequency", y_new, y_old).set_index("band")

    assert table.loc["undefined (old rate = 0)", "exposure"] == pytest.approx(1.0)  # p1's weight
    assert table["exposure_share"].sum() == pytest.approx(1.0)
