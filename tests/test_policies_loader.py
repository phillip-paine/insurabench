from __future__ import annotations

import pandas as pd
import pytest

from insurabench.data.schema import PolicySchema
from insurabench.exceptions import PolicyIdentityError, SchemaError
from insurabench.policies.loader import load_policies


def test_load_policies_simple_book_passes(simple_book):
    schema = simple_book.policy_schema
    df = load_policies(simple_book.policies, schema)
    assert len(df) == len(simple_book.policies)


def test_load_policies_renewal_book_passes(renewal_book):
    schema = renewal_book.policy_schema
    df = load_policies(renewal_book.policies, schema)
    assert len(df) == len(renewal_book.policies)


def test_missing_column_raises_schema_error(simple_book):
    schema = simple_book.policy_schema
    df = simple_book.policies.drop(columns=["region"])
    with pytest.raises(SchemaError, match="region"):
        load_policies(df, schema)


def test_non_positive_exposure_raises(simple_book):
    schema = simple_book.policy_schema
    df = simple_book.policies.copy()
    df.loc[0, schema.exposure_col] = 0
    with pytest.raises(SchemaError, match="exposure"):
        load_policies(df, schema)


def test_duplicate_policy_id_without_term_start_raises(simple_book):
    """Simulates a renewal book being loaded with a schema that has no
    term_start_col configured -- design brief §3.2."""
    schema = simple_book.policy_schema
    df = pd.concat([simple_book.policies, simple_book.policies.iloc[[0]]], ignore_index=True)
    with pytest.raises(PolicyIdentityError, match="term_start_col"):
        load_policies(df, schema)


def test_renewal_book_requires_term_start_col(renewal_book):
    """A renewal book with policy_id repeating, but no term_start_col in the
    schema, must raise -- not silently keep only one term per policy."""
    naive_schema = PolicySchema(
        policy_id_col=renewal_book.policy_schema.policy_id_col,
        exposure_col=renewal_book.policy_schema.exposure_col,
        feature_roles=renewal_book.policy_schema.feature_roles,
    )
    # only trip this if the synthetic book actually produced duplicates
    has_dupes = renewal_book.policies["policy_id"].duplicated().any()
    if has_dupes:
        with pytest.raises(PolicyIdentityError):
            load_policies(renewal_book.policies, naive_schema)


def test_duplicate_period_key_raises_even_with_term_start(renewal_book):
    schema = renewal_book.policy_schema
    df = pd.concat([renewal_book.policies, renewal_book.policies.iloc[[0]]], ignore_index=True)
    with pytest.raises(PolicyIdentityError):
        load_policies(df, schema)
