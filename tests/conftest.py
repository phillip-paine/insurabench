from __future__ import annotations

import pandas as pd
import pytest

from insurabench.data.datasets import make_synthetic_two_table
from insurabench.data.policy_frame import PolicyFrame
from insurabench.data.schema import ClaimsSchema, FeatureRole, PolicySchema


@pytest.fixture
def simple_book():
    """A synthetic book with one term per policy -- bare policy_id is unique."""
    return make_synthetic_two_table(n_policies=50, renewals=False, seed=1)


@pytest.fixture
def renewal_book():
    """A synthetic book with renewal terms -- policy_id recurs, term_start
    disambiguates."""
    return make_synthetic_two_table(n_policies=50, renewals=True, seed=2)


@pytest.fixture
def toy_frequency_pf() -> PolicyFrame:
    """4 policy-periods, 2 regions, no renewals (bare policy_id is
    already unique -> no term_start/term_end needed) -- small and fully
    hand-computable, for golden-value curves/evaluation tests
    (tests/test_curves.py, tests/test_evaluation.py).

    region A: p1 (exposure 1, 1 claim), p2 (exposure 1, 3 claims)
      -> claim_count = [1, 3], exposure = [1, 1]
    region B: p3 (exposure 2, 0 claims), p4 (exposure 2, 2 claims)
      -> claim_count = [0, 2], exposure = [2, 2]
    """
    policies = pd.DataFrame(
        {
            "policy_id": ["p1", "p2", "p3", "p4"],
            "exposure": [1.0, 1.0, 2.0, 2.0],
            "region": ["A", "A", "B", "B"],
        }
    )
    claims = pd.DataFrame(
        {
            "claim_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "policy_id": ["p1", "p2", "p2", "p2", "p4", "p4"],
            "claim_date": pd.Timestamp("2024-01-01"),
            "claim_amount": 100.0,
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
    return PolicyFrame.from_tables(policies, claims, policy_schema, claims_schema, verbose=False)


@pytest.fixture
def toy_two_feature_pf() -> PolicyFrame:
    """Same 4-policy-period frequency structure as ``toy_frequency_pf``,
    but with a second categorical feature (``vehicle_type``) crossed with
    ``region`` so every (region, vehicle_type) cell has exactly one
    policy-period -- for hand-computable ``two_way_curve`` tests
    (tests/test_curves.py).

    region A, vtype X: p1 (exposure 1, 1 claim)
    region A, vtype Y: p2 (exposure 1, 3 claims)
    region B, vtype X: p3 (exposure 2, 0 claims)
    region B, vtype Y: p4 (exposure 2, 2 claims)
    """
    policies = pd.DataFrame(
        {
            "policy_id": ["p1", "p2", "p3", "p4"],
            "exposure": [1.0, 1.0, 2.0, 2.0],
            "region": ["A", "A", "B", "B"],
            "vehicle_type": ["X", "Y", "X", "Y"],
        }
    )
    claims = pd.DataFrame(
        {
            "claim_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "policy_id": ["p1", "p2", "p2", "p2", "p4", "p4"],
            "claim_date": pd.Timestamp("2024-01-01"),
            "claim_amount": 100.0,
        }
    )
    policy_schema = PolicySchema(
        policy_id_col="policy_id",
        exposure_col="exposure",
        feature_roles={"region": FeatureRole.CATEGORICAL, "vehicle_type": FeatureRole.CATEGORICAL},
    )
    claims_schema = ClaimsSchema(
        claim_id_col="claim_id",
        policy_id_col="policy_id",
        claim_date_col="claim_date",
        claim_amount_col="claim_amount",
    )
    return PolicyFrame.from_tables(policies, claims, policy_schema, claims_schema, verbose=False)


@pytest.fixture
def toy_severity_pf() -> PolicyFrame:
    """3 policies, 1 claim each, chosen so claim amounts (and predictions
    passed against them) reduce to clean fractions for hand-computed
    Gini golden values (tests/test_evaluation.py). All exposures are
    irrelevant to severity's grain (weight is always 1 per claim) but set
    anyway to keep the policies table realistic."""
    policies = pd.DataFrame(
        {
            "policy_id": ["p1", "p2", "p3"],
            "exposure": [1.0, 1.0, 1.0],
            "region": ["A", "A", "A"],
        }
    )
    claims = pd.DataFrame(
        {
            "claim_id": ["c1", "c2", "c3"],
            "policy_id": ["p1", "p2", "p3"],
            "claim_date": pd.Timestamp("2024-01-01"),
            "claim_amount": [1.0, 3.0, 5.0],
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
    return PolicyFrame.from_tables(policies, claims, policy_schema, claims_schema, verbose=False)
