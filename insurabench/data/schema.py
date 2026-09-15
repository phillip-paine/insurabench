"""Column-role contracts for policies and claims tables.

These dataclasses are the single place that says "which column in your
dataframe means what". Every other module (loaders, linking, aggregate,
PolicyFrame) reads columns through a schema object rather than hardcoding
column names, so the same code works regardless of what the source system
happens to call things -- this is what lets the rest of the design brief
promise that curves/evaluation are model- and source-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FeatureRole(str, Enum):
    """How a rating-factor column should be treated downstream (curves,
    models, geo). Declared once here rather than re-inferred by each
    consumer from dtype, which is fragile (e.g. an integer-coded region
    isn't continuous just because it's numeric)."""

    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    SPATIAL = "spatial"


@dataclass(frozen=True)
class PolicySchema:
    """Column mapping for the policies table.

    Parameters
    ----------
    policy_id_col:
        Column identifying the policy/contract. May repeat across renewal
        terms (design brief §3.2) -- that's fine as long as
        ``term_start_col`` is also supplied to disambiguate.
    exposure_col:
        Earned exposure for the policy-period, in years. Must be strictly
        positive.
    term_start_col, term_end_col:
        Start/end of the policy-period. ``term_start_col`` doubles as the
        disambiguator for policy identity when ``policy_id`` recurs across
        terms. ``term_end_col`` is required to validate that claim dates
        fall within the covered period (design brief §3.1) and to attribute
        a claim to the correct term when policy_id recurs.
    premium_col:
        Optional. Charged premium for the period.
    feature_roles:
        Maps rating-factor column name -> FeatureRole. Only columns listed
        here are treated as model features; anything else on the table is
        carried through as metadata but ignored by curves/models.
    """

    policy_id_col: str
    exposure_col: str
    term_start_col: str | None = None
    term_end_col: str | None = None
    premium_col: str | None = None
    feature_roles: dict[str, FeatureRole] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.term_start_col is None) != (self.term_end_col is None):
            raise ValueError(
                "PolicySchema requires term_start_col and term_end_col to be "
                "set together (or both left as None) -- claim-to-term "
                "attribution needs the full window, not just a start date."
            )

    @property
    def feature_cols(self) -> list[str]:
        return list(self.feature_roles)

    @property
    def period_key_cols(self) -> list[str]:
        """Columns that together uniquely identify a policy-period.

        Falls back to bare ``policy_id`` when no ``term_start_col`` is
        configured. Callers are responsible for having verified that this
        is actually unique -- ``insurabench.data.validation.resolve_policy_period_key``
        does that check.
        """
        if self.term_start_col is not None:
            return [self.policy_id_col, self.term_start_col]
        return [self.policy_id_col]


@dataclass(frozen=True)
class ClaimsSchema:
    """Column mapping for the claims table.

    Parameters
    ----------
    claim_id_col:
        Unique claim identifier.
    policy_id_col:
        Foreign key into the policies table. Declared independently of the
        policies table's own ``policy_id_col`` name (they're often the same
        string, but loaders don't assume identical naming across source
        systems).
    claim_date_col:
        Date the claim occurred (or was reported -- insurabench does not
        distinguish; see the reserving-assumption note on ``PolicyFrame``).
    claim_amount_col:
        Final/as-of claim amount. insurabench treats this as a snapshot,
        not a maturing estimate -- no development modeling (design brief
        §3.4, §9.6).
    peril_col:
        Optional. The cause of loss (collision, theft, fire, weather, ...).
        Perils are frequently modeled separately -- their frequency/severity
        drivers differ -- and the resulting pure premiums summed at the
        policy level. Setting this column lets ``frequency_view``/
        ``severity_view`` be filtered to one peril (or a subset) at a time.
    coverage_col:
        Optional. Which section of the policy responds (e.g. Liability,
        Collision, Comprehensive) -- distinct from peril: a single coverage
        can be triggered by more than one peril (Comprehensive covers both
        theft and weather, say). Kept as its own column, filterable the
        same way as ``peril_col``, since pricing is often split by coverage
        too.
    """

    claim_id_col: str
    policy_id_col: str
    claim_date_col: str
    claim_amount_col: str
    peril_col: str | None = None
    coverage_col: str | None = None
