from __future__ import annotations

import pytest

from insurabench.model_selection import (
    train_test_split_policy_frame,
    train_val_test_split_policy_frame,
)


def _pf(book):
    from insurabench.data.policy_frame import PolicyFrame

    return PolicyFrame.from_tables(
        book.policies, book.claims, book.policy_schema, book.claims_schema, verbose=False
    )


def test_train_test_split_no_overlap_and_full_coverage(renewal_book):
    pf = _pf(renewal_book)
    train, test = train_test_split_policy_frame(pf, test_size=0.3, seed=0)

    key_cols = pf.policy_schema.period_key_cols
    train_keys = set(map(tuple, train.policies[key_cols].itertuples(index=False, name=None)))
    test_keys = set(map(tuple, test.policies[key_cols].itertuples(index=False, name=None)))
    assert train_keys.isdisjoint(test_keys)
    assert len(train.policies) + len(test.policies) == len(pf.policies)
    assert len(train.claims) + len(test.claims) == len(pf.claims)


def test_train_test_split_claims_never_leak(renewal_book):
    """Every claim in the test split must belong to a policy-period that is
    actually in the test split's policies table (design brief §3.3 -- a
    naive independent shuffle of the two tables could violate this)."""
    pf = _pf(renewal_book)
    _train, test = train_test_split_policy_frame(pf, test_size=0.3, seed=1)

    from insurabench.aggregate import match_claims_to_periods

    matched_test = match_claims_to_periods(
        test.policies, test.claims, test.policy_schema, test.claims_schema
    )
    key_cols = test.policy_schema.period_key_cols
    test_keys = set(map(tuple, test.policies[key_cols].itertuples(index=False, name=None)))
    matched_keys = set(map(tuple, matched_test[key_cols].itertuples(index=False, name=None)))
    assert matched_keys <= test_keys


def test_train_val_test_split_sizes_roughly_correct(renewal_book):
    pf = _pf(renewal_book)
    train, val, test = train_val_test_split_policy_frame(
        pf, val_size=0.2, test_size=0.2, seed=2
    )
    total = len(pf.policies)
    assert len(train.policies) + len(val.policies) + len(test.policies) == total
    assert abs(len(val.policies) / total - 0.2) < 0.1
    assert abs(len(test.policies) / total - 0.2) < 0.1


def test_train_val_test_split_rejects_invalid_fractions(renewal_book):
    pf = _pf(renewal_book)
    with pytest.raises(ValueError):
        train_val_test_split_policy_frame(pf, val_size=0.6, test_size=0.6)


def test_split_is_reproducible_with_seed(renewal_book):
    pf = _pf(renewal_book)
    train_a, test_a = train_test_split_policy_frame(pf, test_size=0.25, seed=42)
    train_b, test_b = train_test_split_policy_frame(pf, test_size=0.25, seed=42)
    assert train_a.policies.equals(train_b.policies)
    assert test_a.policies.equals(test_b.policies)
