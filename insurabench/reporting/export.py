"""Export a fitted model's rating factors into the format actuaries
already expect (design brief §5's ``relativity_table.py`` note, §8's
``reporting/export.py``).

Builds directly on ``curves.relativity_table`` -- the base-level
normalization design brief §5 asks for is already done there; this module
only adds the two things a rating-table *export* needs that a tidy
analysis DataFrame doesn't: an explicit ``is_base`` marker (rather than
leaving the base level implicit as "whichever row has relativity == 1.0",
fragile under floating-point equality) and, for ``format="xlsx"``, one
sheet per rating factor -- the layout an actuarial pricing team
reviewing/loading a rating table would actually expect, rather than one
long tidy table they'd have to pivot themselves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from insurabench.curves import relativity_table as _relativity_table
from insurabench.data.policy_frame import PolicyFrame

Target = Literal["frequency", "severity", "pure_premium"]


def _with_is_base(table: pd.DataFrame) -> pd.DataFrame:
    """Add an explicit ``is_base`` column: the row within each feature
    whose ``relativity`` is closest to 1.0 -- not an exact
    ``relativity == 1.0`` floating-point equality check, which
    ``curves.relativity_table``'s own division (``fitted / base_fitted``)
    isn't guaranteed to produce exactly even for the base row itself.
    """
    table = table.copy()
    table["is_base"] = False
    for idx in table.groupby("feature").groups.values():
        sub = table.loc[idx]
        closest = (sub["relativity"] - 1.0).abs().idxmin()
        table.loc[closest, "is_base"] = True
    return table


def export_rating_table(
    pf: PolicyFrame,
    target: Target,
    y_pred,
    output_path: str | Path,
    *,
    features: list[str] | None = None,
    base: Literal["max_exposure", "first"] = "max_exposure",
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
    n_bins: int = 10,
    binning: Literal["quantile", "uniform"] = "quantile",
) -> Path:
    """Build a rating table (``curves.relativity_table``, plus an explicit
    ``is_base`` column) and write it to ``output_path``.

    Format is chosen from ``output_path``'s suffix:

    - ``.csv`` -- one long, tidy table (``feature``, ``level``, ``n_obs``,
      ``exposure``, ``observed``, ``fitted``, ``relativity``, ``is_base``),
      sorted by feature then descending exposure. Round-trips into a
      spreadsheet tool as easily as any other CSV.
    - ``.xlsx`` -- one sheet per rating factor, named after the feature
      (truncated to Excel's 31-character sheet-name limit), each sorted
      by descending exposure with ``is_base`` still present -- the layout
      a rating-table workbook actuaries hand-maintain typically has.
      Requires ``openpyxl`` (an optional dependency -- ``pip install
      openpyxl``, or ``pip install insurabench[excel]``); raises a clear
      ``ImportError`` with that instruction if it isn't installed, rather
      than a cryptic one from deep inside pandas.

    Returns the ``Path`` written to (so callers can immediately
    ``present_files``/log it without re-deriving the path).
    """
    output_path = Path(output_path)
    table = _relativity_table(
        pf, target, y_pred,
        features=features, base=base, peril=peril, coverage=coverage,
        n_bins=n_bins, binning=binning,
    )
    table = _with_is_base(table)
    table = table.sort_values(["feature", "exposure"], ascending=[True, False]).reset_index(drop=True)

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        table.to_csv(output_path, index=False)
    elif suffix == ".xlsx":
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "export_rating_table(..., '.xlsx') requires openpyxl. "
                "Install it with `pip install openpyxl` (or "
                "`pip install insurabench[excel]`)."
            ) from exc
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for feature, group in table.groupby("feature", sort=False):
                sheet_name = str(feature)[:31]  # Excel's own sheet-name length limit
                group.drop(columns="feature").to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        raise ValueError(
            f"Unsupported output format {suffix!r} (from {output_path!r}) -- "
            f"use a '.csv' or '.xlsx' path."
        )
    return output_path
