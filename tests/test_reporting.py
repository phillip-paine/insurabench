"""Tests for insurabench.reporting (model_card, export).

Both modules are thin consumers of already-tested curves/evaluation/model
functions, so these tests focus on: the assembly is correct (right
numbers land in the right fields), the model-type-specific branches
(AIC/BIC only for GLM, family reporting for a composed model) behave as
documented, and the export round-trips into a readable file correctly
(including the is_base marker, which is genuinely new logic, not just
reused output).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurabench.data.datasets import make_synthetic_two_table
from insurabench.data.policy_frame import PolicyFrame
from insurabench.model_selection import train_test_split_policy_frame
from insurabench.models.frequency_severity import FrequencySeverityModel
from insurabench.models.gbm import GBMPricingModel
from insurabench.models.glm import GLMPricingModel
from insurabench.reporting import ModelCard, export_rating_table, generate_model_card


@pytest.fixture(scope="module")
def split_pf():
    book = make_synthetic_two_table(n_policies=1500, renewals=True, seed=13)
    pf = PolicyFrame.from_tables(
        book.policies, book.claims, book.policy_schema, book.claims_schema, verbose=False
    )
    return train_test_split_policy_frame(pf, test_size=0.3, seed=13)


# --- model_card ----------------------------------------------------------------


def test_glm_card_has_aic_bic_and_matches_underlying_metrics(split_pf):
    train_pf, test_pf = split_pf
    model = GLMPricingModel(family="poisson").fit_policy_frame(train_pf, "frequency")

    card = generate_model_card(model, "frequency", test_pf, train_pf=train_pf)

    assert isinstance(card, ModelCard)
    assert card.model_type == "GLMPricingModel"
    assert card.family == "poisson"
    assert card.aic is not None and np.isfinite(card.aic)
    assert card.bic is not None and np.isfinite(card.bic)
    assert card.train_d2 == pytest.approx(model.score_policy_frame(train_pf))
    assert card.test_d2 == pytest.approx(model.score_policy_frame(test_pf))
    assert card.n_policies == test_pf.n_policies
    assert card.n_claims == test_pf.n_claims
    assert card.total_exposure == pytest.approx(test_pf.total_exposure)
    assert len(card.top_relativities) <= 10


def test_gbm_card_has_no_aic_bic(split_pf):
    train_pf, test_pf = split_pf
    model = GBMPricingModel(family="poisson", iterations=100).fit_policy_frame(train_pf, "frequency")

    card = generate_model_card(model, "frequency", test_pf)

    assert card.model_type == "GBMPricingModel"
    assert card.aic is None
    assert card.bic is None
    assert np.isfinite(card.gini)


def test_composed_card_reports_both_leg_families_and_needs_power(split_pf):
    train_pf, test_pf = split_pf
    model = FrequencySeverityModel(
        frequency_model=GLMPricingModel(family="poisson"),
        severity_model=GLMPricingModel(family="gamma"),
    ).fit_policy_frame(train_pf)

    with pytest.raises(ValueError, match="power is required"):
        generate_model_card(model, "pure_premium", test_pf)

    card = generate_model_card(model, "pure_premium", test_pf, power=1.5)
    assert card.model_type == "FrequencySeverityModel"
    assert card.family == {"frequency": "poisson", "severity": "gamma"}
    assert card.aic is None and card.bic is None
    assert set(card.raw_relativities) == {"frequency", "severity"}


def test_top_relativities_are_sorted_by_deviation_from_base(split_pf):
    train_pf, test_pf = split_pf
    model = GLMPricingModel(family="poisson").fit_policy_frame(train_pf, "frequency")
    card = generate_model_card(model, "frequency", test_pf, top_n=5)

    deviation = np.abs(np.log(card.top_relativities["relativity"].to_numpy(dtype=float)))
    assert (np.diff(deviation) <= 1e-9).all()  # non-increasing
    assert len(card.top_relativities) == 5


def test_to_markdown_renders_without_error(split_pf):
    train_pf, test_pf = split_pf
    model = GLMPricingModel(family="poisson").fit_policy_frame(train_pf, "frequency")
    card = generate_model_card(model, "frequency", test_pf, train_pf=train_pf)

    md = card.to_markdown()
    assert isinstance(md, str)
    assert "GLMPricingModel" in md
    assert "Train D²" in md
    assert "AIC" in md


# --- export_rating_table ---------------------------------------------------------


def test_export_csv_has_exactly_one_base_row_per_feature(split_pf, tmp_path):
    train_pf, test_pf = split_pf
    model = GLMPricingModel(family="poisson").fit_policy_frame(train_pf, "frequency")
    y_pred = model.predict_policy_frame(test_pf)

    path = export_rating_table(test_pf, "frequency", y_pred, tmp_path / "rating_table.csv")
    out = pd.read_csv(path)

    assert set(out["feature"]) == set(test_pf.policy_schema.feature_roles)
    base_counts = out.groupby("feature")["is_base"].sum()
    assert (base_counts == 1).all()
    # the flagged base row should itself have relativity == 1.0
    base_rel = out.loc[out["is_base"], "relativity"].to_numpy(dtype=float)
    assert np.allclose(base_rel, 1.0)


def test_export_xlsx_has_one_sheet_per_feature(split_pf, tmp_path):
    train_pf, test_pf = split_pf
    model = GLMPricingModel(family="poisson").fit_policy_frame(train_pf, "frequency")
    y_pred = model.predict_policy_frame(test_pf)

    path = export_rating_table(test_pf, "frequency", y_pred, tmp_path / "rating_table.xlsx")
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")

    assert set(sheets) == set(test_pf.policy_schema.feature_roles)
    for sheet in sheets.values():
        assert "feature" not in sheet.columns  # dropped -- redundant with the sheet name
        assert sheet["is_base"].sum() == 1


def test_export_unsupported_format_raises(split_pf, tmp_path):
    train_pf, test_pf = split_pf
    model = GLMPricingModel(family="poisson").fit_policy_frame(train_pf, "frequency")
    y_pred = model.predict_policy_frame(test_pf)

    with pytest.raises(ValueError, match="Unsupported output format"):
        export_rating_table(test_pf, "frequency", y_pred, tmp_path / "rating_table.json")


def test_export_pure_premium_works_for_composed_model(split_pf, tmp_path):
    train_pf, test_pf = split_pf
    model = FrequencySeverityModel(
        frequency_model=GLMPricingModel(family="poisson"),
        severity_model=GLMPricingModel(family="gamma"),
    ).fit_policy_frame(train_pf)
    y_pred = model.predict_policy_frame(test_pf)

    path = export_rating_table(test_pf, "pure_premium", y_pred, tmp_path / "pp_rating_table.csv")
    out = pd.read_csv(path)
    assert len(out) > 0
