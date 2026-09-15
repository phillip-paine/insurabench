from __future__ import annotations

import pytest

from insurabench.claims.loader import load_claims
from insurabench.exceptions import SchemaError


def test_load_claims_passes(simple_book):
    df = load_claims(simple_book.claims, simple_book.claims_schema)
    assert len(df) == len(simple_book.claims)


def test_missing_column_raises(simple_book):
    df = simple_book.claims.drop(columns=["peril"])
    with pytest.raises(SchemaError, match="peril"):
        load_claims(df, simple_book.claims_schema)


def test_negative_amount_raises(simple_book):
    if simple_book.claims.empty:
        pytest.skip("no claims generated for this seed")
    df = simple_book.claims.copy()
    df.loc[0, "claim_amount"] = -10
    with pytest.raises(SchemaError, match="negative"):
        load_claims(df, simple_book.claims_schema)


def test_duplicate_claim_id_raises(simple_book):
    if len(simple_book.claims) < 2:
        pytest.skip("need at least 2 claims for this test")
    df = simple_book.claims.copy()
    df.loc[1, "claim_id"] = df.loc[0, "claim_id"]
    with pytest.raises(SchemaError, match="not unique"):
        load_claims(df, simple_book.claims_schema)


def test_non_datetime_claim_date_raises(simple_book):
    df = simple_book.claims.copy()
    df["claim_date"] = df["claim_date"].astype(str)
    with pytest.raises(SchemaError, match="datetime"):
        load_claims(df, simple_book.claims_schema)
