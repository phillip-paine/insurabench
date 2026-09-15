"""Loader for the claims table (design brief §3.1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from insurabench.data.schema import ClaimsSchema
from insurabench.data.validation import validate_claims_schema


def load_claims(source: str | Path | pd.DataFrame, schema: ClaimsSchema) -> pd.DataFrame:
    """Load and validate a claims table.

    Same "fail loudly, don't clean silently" contract as
    ``insurabench.policies.loader.load_policies``.

    Raises
    ------
    SchemaError
        Missing/malformed required columns, negative claim amounts, or
        duplicate claim ids.
    """
    df = _read(source)
    validate_claims_schema(df, schema)
    return df


def _read(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
