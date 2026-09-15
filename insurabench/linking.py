"""Join validation between policies and claims (design brief §3.1).

This is deliberately a separate step from validating either table
individually -- a policies table and a claims table can each be internally
valid and still not join correctly (orphan claims, claims dated outside
their policy's covered term, ambiguous period keys). Catching that here,
explicitly, means these failure modes can never be silently swallowed
inside a later ``.fit()`` call. Nothing in this module drops a row; it only
ever raises or reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from insurabench.data.schema import ClaimsSchema, PolicySchema
from insurabench.data.validation import resolve_policy_period_key
from insurabench.exceptions import LinkingError


@dataclass
class LinkageReport:
    """Diagnostic counts from ``check_linkage``.

    This is always computed; whether a problem is escalated to an exception
    is controlled separately by ``validate_linkage``, so callers who want to
    inspect issues (e.g. in a notebook, before deciding how to fix upstream
    data) can call ``check_linkage`` directly without triggering a raise.
    """

    n_policies: int
    n_claims: int
    orphan_claim_ids: list = field(default_factory=list)
    out_of_term_claim_ids: list = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.orphan_claim_ids and not self.out_of_term_claim_ids


def check_linkage(
    policies: pd.DataFrame,
    claims: pd.DataFrame,
    policy_schema: PolicySchema,
    claims_schema: ClaimsSchema,
) -> LinkageReport:
    """Compute (without raising) the set of join problems between
    ``policies`` and ``claims``.

    Checks performed:
      * orphan claims -- a claim's policy id has no matching policy row.
      * out-of-term claims -- claim_date falls outside
        [term_start, term_end) of every policy-period sharing its policy
        id (only checked when both term columns are configured).

    Duplicate policy-period keys are a property of the policies table alone
    and are checked separately, by ``resolve_policy_period_key``.
    """
    policy_ids = set(policies[policy_schema.policy_id_col])
    claim_policy_ids = claims[claims_schema.policy_id_col]
    is_orphan = ~claim_policy_ids.isin(policy_ids)
    orphan_ids = claims.loc[is_orphan, claims_schema.claim_id_col].tolist()

    out_of_term_ids: list = []
    if policy_schema.term_start_col is not None:
        policy_windows = policies[
            [policy_schema.policy_id_col, policy_schema.term_start_col, policy_schema.term_end_col]
        ].rename(columns={policy_schema.policy_id_col: "__policy_id"})

        candidates = claims.loc[~is_orphan].merge(
            policy_windows,
            left_on=claims_schema.policy_id_col,
            right_on="__policy_id",
            how="left",
        )
        in_window = (
            (candidates[claims_schema.claim_date_col] >= candidates[policy_schema.term_start_col])
            & (candidates[claims_schema.claim_date_col] < candidates[policy_schema.term_end_col])
        )
        # A claim is out-of-term only if NONE of the policy-periods sharing
        # its policy id contain its claim_date (a policy id can recur across
        # renewal terms, so a claim may legitimately have several candidate
        # windows and only needs to fit one).
        matches_any = in_window.groupby(candidates[claims_schema.claim_id_col]).any()
        out_of_term_ids = matches_any[~matches_any].index.tolist()

    return LinkageReport(
        n_policies=len(policies),
        n_claims=len(claims),
        orphan_claim_ids=orphan_ids,
        out_of_term_claim_ids=out_of_term_ids,
    )


def validate_linkage(
    policies: pd.DataFrame,
    claims: pd.DataFrame,
    policy_schema: PolicySchema,
    claims_schema: ClaimsSchema,
) -> LinkageReport:
    """Full join validation: raises on any problem rather than returning a
    report to inspect. This is what ``PolicyFrame.from_tables`` calls.

    Raises
    ------
    PolicyIdentityError
        Policy-period key is not unique (design brief §3.2).
    LinkingError
        Orphan claims, or claims dated outside every candidate policy-period's
        term.
    """
    resolve_policy_period_key(policies, policy_schema)

    report = check_linkage(policies, claims, policy_schema, claims_schema)
    problems = []
    if report.orphan_claim_ids:
        example = report.orphan_claim_ids[:10]
        problems.append(
            f"{len(report.orphan_claim_ids)} claim(s) reference a policy id "
            f"not present in the policies table. Example claim id(s): {example}."
        )
    if report.out_of_term_claim_ids:
        example = report.out_of_term_claim_ids[:10]
        problems.append(
            f"{len(report.out_of_term_claim_ids)} claim(s) are dated outside "
            f"every policy-period covering their policy id. Example claim "
            f"id(s): {example}."
        )
    if problems:
        raise LinkingError(
            "Policies/claims join failed validation:\n- " + "\n- ".join(problems)
            + "\nRows are never silently dropped -- fix the source data or "
            "adjust the schema, then re-run (design brief §3.1)."
        )
    return report
