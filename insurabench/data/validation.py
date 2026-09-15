"""Schema-level validation for policies and claims tables.

This module only checks that a single table is internally well-formed
(required columns present, sane dtypes, positive exposure/amounts, unique
ids). It does NOT check anything about how the two tables relate to each
other -- that's ``insurabench.linking``.
"""
from __future__ import annotations

import pandas as pd

from insurabench.data.schema import ClaimsSchema, PolicySchema
from insurabench.exceptions import PolicyIdentityError, SchemaError


def _require_columns(df: pd.DataFrame, columns: list[str | None], *, table_name: str) -> None:
    missing = [c for c in columns if c is not None and c not in df.columns]
    if missing:
        raise SchemaError(
            f"{table_name} is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_policy_schema(df: pd.DataFrame, schema: PolicySchema) -> None:
    """Check that ``df`` satisfies ``schema`` structurally.

    Raises
    ------
    SchemaError
        Missing columns, non-numeric/non-datetime dtypes, or non-positive
        exposure.
    """
    required: list[str | None] = [
        schema.policy_id_col,
        schema.exposure_col,
        schema.term_start_col,
        schema.term_end_col,
        schema.premium_col,
        *schema.feature_cols,
    ]
    _require_columns(df, required, table_name="policies")

    if not pd.api.types.is_numeric_dtype(df[schema.exposure_col]):
        raise SchemaError(
            f"'{schema.exposure_col}' (exposure_col) must be numeric, got "
            f"dtype {df[schema.exposure_col].dtype}."
        )
    non_positive = df[schema.exposure_col] <= 0
    if non_positive.any():
        bad_ids = df.loc[non_positive, schema.policy_id_col].head(10).tolist()
        raise SchemaError(
            f"{int(non_positive.sum())} row(s) have non-positive exposure in "
            f"'{schema.exposure_col}'. Example policy id(s): {bad_ids}. "
            "Exposure must be strictly positive (design brief §3.1)."
        )

    for col in (schema.term_start_col, schema.term_end_col):
        if col is not None and not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise SchemaError(
                f"'{col}' must be a datetime column, got dtype "
                f"{df[col].dtype}. Parse dates before constructing the schema, "
                "e.g. pd.to_datetime(df[col])."
            )

    if schema.term_start_col is not None:
        bad_window = df[schema.term_start_col] >= df[schema.term_end_col]
        if bad_window.any():
            bad_ids = df.loc[bad_window, schema.policy_id_col].head(10).tolist()
            raise SchemaError(
                f"{int(bad_window.sum())} row(s) have term_end_col <= "
                f"term_start_col. Example policy id(s): {bad_ids}."
            )


def validate_claims_schema(df: pd.DataFrame, schema: ClaimsSchema) -> None:
    """Check that ``df`` satisfies ``schema`` structurally.

    Raises
    ------
    SchemaError
        Missing columns, wrong dtypes, negative amounts, or duplicate claim
        ids.
    """
    required: list[str | None] = [
        schema.claim_id_col,
        schema.policy_id_col,
        schema.claim_date_col,
        schema.claim_amount_col,
        schema.peril_col,
        schema.coverage_col,
    ]
    _require_columns(df, required, table_name="claims")

    if not pd.api.types.is_datetime64_any_dtype(df[schema.claim_date_col]):
        raise SchemaError(
            f"'{schema.claim_date_col}' (claim_date_col) must be a datetime "
            f"column, got dtype {df[schema.claim_date_col].dtype}."
        )
    if not pd.api.types.is_numeric_dtype(df[schema.claim_amount_col]):
        raise SchemaError(
            f"'{schema.claim_amount_col}' (claim_amount_col) must be numeric, "
            f"got dtype {df[schema.claim_amount_col].dtype}."
        )
    negative = df[schema.claim_amount_col] < 0
    if negative.any():
        bad_ids = df.loc[negative, schema.claim_id_col].head(10).tolist()
        raise SchemaError(
            f"{int(negative.sum())} claim(s) have negative amount. Example "
            f"claim id(s): {bad_ids}."
        )

    dup = df[schema.claim_id_col].duplicated()
    if dup.any():
        bad_ids = df.loc[dup, schema.claim_id_col].head(10).tolist()
        raise SchemaError(
            f"'{schema.claim_id_col}' is not unique: {int(dup.sum())} "
            f"duplicate claim id(s), e.g. {bad_ids}."
        )


def resolve_policy_period_key(df: pd.DataFrame, schema: PolicySchema) -> PolicySchema:
    """Confirm that ``schema.period_key_cols`` uniquely identifies each row
    of ``df`` (design brief §3.2).

    - If ``term_start_col`` is set: the (policy_id, term_start) pair must be
      unique, or this raises ``PolicyIdentityError``.
    - If ``term_start_col`` is not set: bare ``policy_id`` must be unique.
      If it is, the schema is valid as-is (a single-term book). If it is
      NOT unique, this raises rather than guessing -- the caller must
      re-load with ``term_start_col`` set to disambiguate renewal terms.

    Returns the schema unchanged; this function only validates. It is kept
    separate from the loader so the identity check is independently
    testable and its failure mode is explicit and specific.
    """
    key_cols = schema.period_key_cols
    dup = df.duplicated(subset=key_cols, keep=False)
    if dup.any():
        example = df.loc[dup, key_cols].head(10)
        if schema.term_start_col is None:
            raise PolicyIdentityError(
                f"'{schema.policy_id_col}' is not unique ({int(dup.sum())} "
                "duplicated rows) and no term_start_col was supplied to "
                "disambiguate policy-periods. This usually means policy_id "
                "recurs across renewal terms (design brief §3.2). Re-load "
                "with PolicySchema(term_start_col=..., term_end_col=...) "
                "set to the columns marking each term's window.\n"
                f"Example duplicated key(s):\n{example}"
            )
        raise PolicyIdentityError(
            f"({schema.policy_id_col!r}, {schema.term_start_col!r}) is not a "
            f"unique key ({int(dup.sum())} duplicated rows) even with "
            "term_start_col set. This means the same policy has two rows "
            "claiming the same term start -- check for duplicate ingestion "
            f"or a genuinely ambiguous source system.\nExample:\n{example}"
        )
    return schema
