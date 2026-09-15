from __future__ import annotations

import pandas as pd
import pytest

from insurabench.exceptions import LinkingError
from insurabench.linking import check_linkage, validate_linkage


def test_valid_simple_book_passes(simple_book):
    report = validate_linkage(
        simple_book.policies, simple_book.claims, simple_book.policy_schema, simple_book.claims_schema
    )
    assert report.is_clean
    assert report.n_policies == len(simple_book.policies)
    assert report.n_claims == len(simple_book.claims)


def test_valid_renewal_book_passes(renewal_book):
    report = validate_linkage(
        renewal_book.policies, renewal_book.claims, renewal_book.policy_schema, renewal_book.claims_schema
    )
    assert report.is_clean


def test_orphan_claim_raises(simple_book):
    claims = simple_book.claims.copy()
    orphan = claims.iloc[[0]].copy()
    orphan["policy_id"] = -999
    orphan["claim_id"] = claims["claim_id"].max() + 1
    claims = pd.concat([claims, orphan], ignore_index=True)

    with pytest.raises(LinkingError, match="orphan|reference a policy id"):
        validate_linkage(simple_book.policies, claims, simple_book.policy_schema, simple_book.claims_schema)


def test_out_of_term_claim_raises(renewal_book):
    claims = renewal_book.claims.copy()
    policies = renewal_book.policies
    policy_id = policies["policy_id"].iloc[0]
    earliest_start = policies.loc[policies["policy_id"] == policy_id, "term_start"].min()

    bad_claim = claims.iloc[[0]].copy()
    bad_claim["policy_id"] = policy_id
    bad_claim["claim_date"] = earliest_start - pd.Timedelta(days=10)
    bad_claim["claim_id"] = claims["claim_id"].max() + 1
    claims = pd.concat([claims, bad_claim], ignore_index=True)

    with pytest.raises(LinkingError, match="dated outside"):
        validate_linkage(policies, claims, renewal_book.policy_schema, renewal_book.claims_schema)


def test_check_linkage_reports_without_raising(simple_book):
    claims = simple_book.claims.copy()
    orphan = claims.iloc[[0]].copy()
    orphan["policy_id"] = -999
    orphan["claim_id"] = claims["claim_id"].max() + 1
    claims = pd.concat([claims, orphan], ignore_index=True)

    report = check_linkage(simple_book.policies, claims, simple_book.policy_schema, simple_book.claims_schema)
    assert not report.is_clean
    assert len(report.orphan_claim_ids) == 1
