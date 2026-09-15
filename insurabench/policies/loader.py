"""Loader for the policies table (design brief §3.1-3.2)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from insurabench.data.schema import PolicySchema
from insurabench.data.validation import (
    resolve_policy_period_key,
    validate_policy_schema,
)


def load_policies(source: str | Path | pd.DataFrame, schema: PolicySchema) -> pd.DataFrame:
    """Load and validate a policies table.

    Parameters
    ----------
    source:
        A path to a CSV/Parquet file, or an already-loaded DataFrame.
    schema:
        Column mapping -- see ``PolicySchema``.

    Returns
    -------
    The DataFrame, unchanged except that ``source`` is read from disk if a
    path was given. No rows are dropped or modified: validation failures
    raise rather than silently cleaning the data (design brief §3.1, §9.1).

    Raises
    ------
    SchemaError
        Missing or malformed required columns.
    PolicyIdentityError
        ``policy_id`` (or (policy_id, term_start)) is not a unique key --
        see design brief §3.2.
    """
    df = _read(source)
    validate_policy_schema(df, schema)
    resolve_policy_period_key(df, schema)
    return df


def _read(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
