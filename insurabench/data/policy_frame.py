"""``PolicyFrame`` -- the spine every downstream insurabench module operates
on (design brief §3.5). It bundles validated, linked policies + claims
tables with their schemas, so a model, curve, or evaluation function that
receives a PolicyFrame always knows which column is exposure, which are
features and what role each plays, and never has to re-derive or guess at
that -- this is what structurally enforces correct exposure-weighting
everywhere instead of hoping each module remembers to do it (design brief
§9.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from insurabench.aggregate import to_frequency_frame, to_severity_frame
from insurabench.data.schema import ClaimsSchema, PolicySchema
from insurabench.linking import LinkageReport, validate_linkage

RESERVING_ASSUMPTION = (
    "insurabench treats claim amounts as final/as-of snapshot values: there "
    "is no paid/incurred/reserved distinction and no claim-development "
    "modeling. If your claims are still maturing, finalize/aggregate them "
    "upstream first (design brief §3.4, §9.6)."
)


@dataclass
class PolicyFrame:
    """Validated, linked policies + claims with an explicit schema.

    Construct via ``PolicyFrame.from_tables`` rather than the dataclass
    constructor directly -- that classmethod is what runs the linkage
    validation this class assumes has already passed.
    """

    policies: pd.DataFrame
    claims: pd.DataFrame
    policy_schema: PolicySchema
    claims_schema: ClaimsSchema
    linkage_report: LinkageReport

    @classmethod
    def from_tables(
        cls,
        policies: pd.DataFrame,
        claims: pd.DataFrame,
        policy_schema: PolicySchema,
        claims_schema: ClaimsSchema,
        *,
        verbose: bool = True,
    ) -> PolicyFrame:
        """Validate the policies/claims join and construct a PolicyFrame.

        Callers who already loaded each table via
        ``insurabench.policies.loader.load_policies`` /
        ``insurabench.claims.loader.load_claims`` have already passed the
        per-table schema checks; this additionally runs the cross-table
        join checks (``insurabench.linking.validate_linkage``) before
        anything downstream can consume the data.

        Raises
        ------
        PolicyIdentityError, LinkingError
            See ``insurabench.linking.validate_linkage``.
        """
        report = validate_linkage(policies, claims, policy_schema, claims_schema)
        if verbose:
            print(RESERVING_ASSUMPTION)
        return cls(
            policies=policies,
            claims=claims,
            policy_schema=policy_schema,
            claims_schema=claims_schema,
            linkage_report=report,
        )

    def frequency_view(
        self,
        *,
        peril: str | list[str] | None = None,
        coverage: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """One row per policy-period: ``claim_count``, exposure, features.
        See design brief §3.3. Recomputed on each call -- cheap relative to
        model fitting, and avoids stale-cache bugs if the underlying tables
        are mutated in place.

        Pass ``peril``/``coverage`` to count only claims matching that
        peril/coverage (or list of them) -- e.g. to fit a peril-specific
        frequency model. Requires the corresponding column to be set on
        ``claims_schema``.
        """
        return to_frequency_frame(
            self.policies, self.claims, self.policy_schema, self.claims_schema,
            peril=peril, coverage=coverage,
        )

    def severity_view(
        self,
        *,
        peril: str | list[str] | None = None,
        coverage: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """One row per claim: ``claim_amount`` + the policy-period's
        features, never rolled up. See design brief §3.3. ``peril``/
        ``coverage`` filter as in ``frequency_view``."""
        return to_severity_frame(
            self.policies, self.claims, self.policy_schema, self.claims_schema,
            peril=peril, coverage=coverage,
        )

    @property
    def available_perils(self) -> list | None:
        """Distinct values in ``claims_schema.peril_col``, or ``None`` if
        that column isn't configured. Useful for looping over perils to fit
        separate frequency/severity models and combine the results."""
        if self.claims_schema.peril_col is None:
            return None
        return sorted(self.claims[self.claims_schema.peril_col].dropna().unique().tolist())

    @property
    def available_coverages(self) -> list | None:
        """Distinct values in ``claims_schema.coverage_col``, or ``None`` if
        that column isn't configured."""
        if self.claims_schema.coverage_col is None:
            return None
        return sorted(self.claims[self.claims_schema.coverage_col].dropna().unique().tolist())

    @property
    def n_policies(self) -> int:
        return len(self.policies)

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    @property
    def total_exposure(self) -> float:
        return float(self.policies[self.policy_schema.exposure_col].sum())

    def describe(self) -> str:
        """Human-readable summary, including the reserving assumption --
        callers can print this on demand rather than only at construction
        time."""
        summary = (
            f"PolicyFrame: {self.n_policies} policy-period(s), "
            f"{self.n_claims} claim(s), {self.total_exposure:.1f} total exposure."
        )
        lines = [
            summary,
            f"Feature roles: {self.policy_schema.feature_roles}",
            RESERVING_ASSUMPTION,
        ]
        return "\n".join(lines)
