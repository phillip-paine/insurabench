from __future__ import annotations

import pandas as pd
import pytest

from insurabench.data.policy_frame import RESERVING_ASSUMPTION, PolicyFrame
from insurabench.exceptions import LinkingError


def test_from_tables_simple_book(simple_book, capsys):
    pf = PolicyFrame.from_tables(
        simple_book.policies, simple_book.claims, simple_book.policy_schema, simple_book.claims_schema
    )
    captured = capsys.readouterr()
    assert RESERVING_ASSUMPTION in captured.out

    assert pf.n_policies == len(simple_book.policies)
    assert pf.n_claims == len(simple_book.claims)
    assert pf.total_exposure == pytest.approx(simple_book.policies["exposure"].sum())


def test_from_tables_renewal_book(renewal_book):
    pf = PolicyFrame.from_tables(
        renewal_book.policies,
        renewal_book.claims,
        renewal_book.policy_schema,
        renewal_book.claims_schema,
        verbose=False,
    )
    assert pf.linkage_report.is_clean


def test_frequency_and_severity_views_are_consistent(renewal_book):
    pf = PolicyFrame.from_tables(
        renewal_book.policies,
        renewal_book.claims,
        renewal_book.policy_schema,
        renewal_book.claims_schema,
        verbose=False,
    )
    freq = pf.frequency_view()
    sev = pf.severity_view()

    assert len(freq) == pf.n_policies
    assert len(sev) == pf.n_claims
    assert freq["claim_count"].sum() == len(sev)


def test_from_tables_raises_on_bad_join(simple_book):
    claims = simple_book.claims.copy()
    orphan = claims.iloc[[0]].copy() if not claims.empty else pd.DataFrame(
        {
            "claim_id": [999999],
            "policy_id": [-1],
            "claim_date": [pd.Timestamp("2023-01-01")],
            "claim_amount": [100.0],
            "peril": ["collision"],
            "coverage": ["Collision"],
        }
    )
    orphan["policy_id"] = -1
    orphan["claim_id"] = 999999
    claims = pd.concat([claims, orphan], ignore_index=True)

    with pytest.raises(LinkingError):
        PolicyFrame.from_tables(
            simple_book.policies, claims, simple_book.policy_schema, simple_book.claims_schema
        )


def test_describe_includes_reserving_assumption(simple_book):
    pf = PolicyFrame.from_tables(
        simple_book.policies, simple_book.claims, simple_book.policy_schema, simple_book.claims_schema,
        verbose=False,
    )
    assert RESERVING_ASSUMPTION in pf.describe()


def test_available_perils_and_per_peril_views(simple_book):
    pf = PolicyFrame.from_tables(
        simple_book.policies, simple_book.claims, simple_book.policy_schema, simple_book.claims_schema,
        verbose=False,
    )
    perils = pf.available_perils
    coverages = pf.available_coverages
    assert perils is not None and coverages is not None

    if not perils:
        pytest.skip("no claims generated for this seed")

    total_claims = 0
    for p in perils:
        sev_p = pf.severity_view(peril=p)
        total_claims += len(sev_p)
        assert (sev_p["peril"] == p).all()
    assert total_claims == pf.n_claims
