from __future__ import annotations

import pandas as pd
import pytest

from insurabench.aggregate import to_frequency_frame, to_severity_frame
from insurabench.data.schema import ClaimsSchema, FeatureRole, PolicySchema
from insurabench.exceptions import SchemaError


def test_frequency_frame_shape_and_zero_fill(simple_book):
    freq = to_frequency_frame(
        simple_book.policies, simple_book.claims, simple_book.policy_schema, simple_book.claims_schema
    )
    # one row per policy-period, even for policies with zero claims
    assert len(freq) == len(simple_book.policies)
    assert "claim_count" in freq.columns
    assert freq["claim_count"].sum() == len(simple_book.claims)
    assert (freq["claim_count"] >= 0).all()


def test_severity_frame_one_row_per_claim_never_summed(simple_book):
    sev = to_severity_frame(
        simple_book.policies, simple_book.claims, simple_book.policy_schema, simple_book.claims_schema
    )
    assert len(sev) == len(simple_book.claims)
    # features must be present on every claim row
    for col in simple_book.policy_schema.feature_cols:
        assert col in sev.columns
        assert sev[col].notna().all()


def test_renewal_terms_claim_count_not_double_counted():
    """A policy_id with two non-overlapping renewal terms and one claim in
    each term must show claim_count == 1 for each term, not 2 for both
    (the bug this design explicitly guards against -- design brief §3.3)."""
    policies = pd.DataFrame(
        {
            "policy_id": [1, 1],
            "term_start": [pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")],
            "term_end": [pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")],
            "exposure": [1.0, 1.0],
            "region": ["North", "North"],
        }
    )
    claims = pd.DataFrame(
        {
            "claim_id": [101, 102],
            "policy_id": [1, 1],
            "claim_date": [pd.Timestamp("2023-06-01"), pd.Timestamp("2024-06-01")],
            "claim_amount": [500.0, 700.0],
        }
    )
    policy_schema = PolicySchema(
        policy_id_col="policy_id",
        exposure_col="exposure",
        term_start_col="term_start",
        term_end_col="term_end",
        feature_roles={"region": FeatureRole.CATEGORICAL},
    )
    claims_schema = ClaimsSchema(
        claim_id_col="claim_id",
        policy_id_col="policy_id",
        claim_date_col="claim_date",
        claim_amount_col="claim_amount",
    )

    freq = to_frequency_frame(policies, claims, policy_schema, claims_schema)
    assert len(freq) == 2
    assert (freq["claim_count"] == 1).all()

    sev = to_severity_frame(policies, claims, policy_schema, claims_schema)
    assert len(sev) == 2
    assert sorted(sev["claim_amount"].tolist()) == [500.0, 700.0]


def test_claim_id_and_policy_id_columns_can_share_a_name():
    """Regression test: claims_schema.policy_id_col and
    policy_schema.policy_id_col are both literally 'policy_id' here (the
    common case) -- the merge logic must not collide on that name."""
    policies = pd.DataFrame(
        {"policy_id": [1, 2], "exposure": [1.0, 1.0], "region": ["North", "South"]}
    )
    claims = pd.DataFrame(
        {
            "claim_id": [1],
            "policy_id": [1],
            "claim_date": [pd.Timestamp("2023-06-01")],
            "claim_amount": [500.0],
        }
    )
    policy_schema = PolicySchema(
        policy_id_col="policy_id",
        exposure_col="exposure",
        feature_roles={"region": FeatureRole.CATEGORICAL},
    )
    claims_schema = ClaimsSchema(
        claim_id_col="claim_id",
        policy_id_col="policy_id",
        claim_date_col="claim_date",
        claim_amount_col="claim_amount",
    )
    freq = to_frequency_frame(policies, claims, policy_schema, claims_schema)
    assert freq.loc[freq["policy_id"] == 1, "claim_count"].iloc[0] == 1
    assert freq.loc[freq["policy_id"] == 2, "claim_count"].iloc[0] == 0


def test_peril_filter_splits_claims_correctly(simple_book):
    perils = simple_book.claims["peril"].unique().tolist()
    if not perils:
        pytest.skip("no claims generated for this seed")
    one_peril = perils[0]

    freq_all = to_frequency_frame(
        simple_book.policies, simple_book.claims, simple_book.policy_schema, simple_book.claims_schema
    )
    freq_one = to_frequency_frame(
        simple_book.policies, simple_book.claims, simple_book.policy_schema, simple_book.claims_schema,
        peril=one_peril,
    )
    expected = (simple_book.claims["peril"] == one_peril).sum()
    assert freq_one["claim_count"].sum() == expected
    assert freq_one["claim_count"].sum() <= freq_all["claim_count"].sum()

    sev_one = to_severity_frame(
        simple_book.policies, simple_book.claims, simple_book.policy_schema, simple_book.claims_schema,
        peril=one_peril,
    )
    assert (sev_one["peril"] == one_peril).all()
    assert len(sev_one) == expected


def test_filtering_by_unset_column_raises(simple_book):
    claims_schema_no_peril = ClaimsSchema(
        claim_id_col=simple_book.claims_schema.claim_id_col,
        policy_id_col=simple_book.claims_schema.policy_id_col,
        claim_date_col=simple_book.claims_schema.claim_date_col,
        claim_amount_col=simple_book.claims_schema.claim_amount_col,
    )
    with pytest.raises(SchemaError, match="peril_col"):
        to_frequency_frame(
            simple_book.policies, simple_book.claims, simple_book.policy_schema,
            claims_schema_no_peril, peril="theft",
        )
