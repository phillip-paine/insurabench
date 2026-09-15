"""Synthetic dataset generation for tests and demos.

Deliberately produces genuinely separate policies/claims tables (never a
pre-joined flat file) with renewal terms, so both the "simple" (bare
policy_id) and "renewal" (recurring policy_id, explicit term_start) paths
through the data layer get exercised (design brief §3.1-3.2).

A loader for a public benchmark dataset (e.g. freMTPL2) belongs here too,
per the design brief's proposed layout -- not yet implemented; see
``load_fremtpl2`` below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from insurabench.data.schema import ClaimsSchema, FeatureRole, PolicySchema

_VEHICLE_BRANDS = ["Alpha", "Beta", "Gamma", "Delta"]
_REGIONS = ["North", "South", "East", "West"]
_PERILS = ["collision", "theft", "weather", "liability"]
_COVERAGES = ["Collision", "Comprehensive", "Liability"]
_PERIL_TO_COVERAGE = {
    "collision": "Collision",
    "theft": "Comprehensive",
    "weather": "Comprehensive",
    "liability": "Liability",
}


@dataclass
class SyntheticBook:
    """A synthetic policies/claims book plus the schemas that describe it."""

    policies: pd.DataFrame
    claims: pd.DataFrame
    policy_schema: PolicySchema
    claims_schema: ClaimsSchema


def make_synthetic_two_table(
    n_policies: int = 200,
    *,
    renewals: bool = True,
    claims_per_policy_lambda: float = 0.15,
    seed: int = 0,
) -> SyntheticBook:
    """Generate a clean, jointly-valid synthetic policies/claims book.

    Parameters
    ----------
    n_policies:
        Number of distinct policy identities (contracts). With
        ``renewals=True`` each contract gets 1-3 terms, so the resulting
        policies table has more rows than ``n_policies``.
    renewals:
        If True, some contracts renew across multiple non-overlapping
        terms and ``policy_id`` recurs -- exercising the design brief §3.2
        disambiguation path. If False, every contract has exactly one term
        and bare ``policy_id`` is already unique.
    claims_per_policy_lambda:
        Poisson rate used to simulate claim counts per policy-period.
    seed:
        RNG seed, for reproducible tests.

    Returns
    -------
    A ``SyntheticBook`` with policies, claims, and both schemas already
    configured to match the generated columns. The data is guaranteed to
    pass ``insurabench.linking.validate_linkage`` as generated.
    """
    rng = np.random.default_rng(seed)

    policy_rows = []
    for policy_id in range(1, n_policies + 1):
        n_terms = int(rng.integers(1, 4)) if renewals else 1
        term_start = pd.Timestamp("2022-01-01") + pd.Timedelta(
            days=int(rng.integers(0, 300))
        )
        for _ in range(n_terms):
            term_end = term_start + pd.DateOffset(years=1)
            policy_rows.append(
                {
                    "policy_id": policy_id,
                    "term_start": term_start,
                    "term_end": term_end,
                    "exposure": round(rng.uniform(0.3, 1.0), 3),
                    "vehicle_power": int(rng.integers(4, 15)),
                    "driver_age": int(rng.integers(18, 80)),
                    "vehicle_brand": rng.choice(_VEHICLE_BRANDS),
                    "region": rng.choice(_REGIONS),
                    "premium": round(rng.uniform(200, 1500), 2),
                }
            )
            term_start = term_end  # next renewal term starts where this one ends

    policies = pd.DataFrame(policy_rows)

    claim_rows = []
    claim_id = 1
    for _, period in policies.iterrows():
        n_claims = rng.poisson(claims_per_policy_lambda)
        for _ in range(n_claims):
            offset_days = int(
                rng.integers(0, (period["term_end"] - period["term_start"]).days)
            )
            peril = rng.choice(_PERILS)
            claim_rows.append(
                {
                    "claim_id": claim_id,
                    "policy_id": period["policy_id"],
                    "claim_date": period["term_start"] + pd.Timedelta(days=offset_days),
                    "claim_amount": round(float(rng.gamma(shape=2.0, scale=800.0)), 2),
                    "peril": peril,
                    "coverage": _PERIL_TO_COVERAGE[peril],
                }
            )
            claim_id += 1

    claims = pd.DataFrame(
        claim_rows,
        columns=["claim_id", "policy_id", "claim_date", "claim_amount", "peril", "coverage"],
    )

    policy_schema = PolicySchema(
        policy_id_col="policy_id",
        exposure_col="exposure",
        term_start_col="term_start" if renewals else None,
        term_end_col="term_end" if renewals else None,
        premium_col="premium",
        feature_roles={
            "vehicle_power": FeatureRole.CONTINUOUS,
            "driver_age": FeatureRole.CONTINUOUS,
            "vehicle_brand": FeatureRole.CATEGORICAL,
            "region": FeatureRole.CATEGORICAL,
        },
    )
    claims_schema = ClaimsSchema(
        claim_id_col="claim_id",
        policy_id_col="policy_id",
        claim_date_col="claim_date",
        claim_amount_col="claim_amount",
        peril_col="peril",
        coverage_col="coverage",
    )
    return SyntheticBook(
        policies=policies, claims=claims, policy_schema=policy_schema, claims_schema=claims_schema
    )


def load_fremtpl2() -> SyntheticBook:
    """Load the freMTPL2 benchmark dataset, split into policies/claims.

    Not yet implemented -- freMTPL2 is the canonical "hello world" walkthrough
    referenced in the design brief (§8, docs/examples/), but ships in most
    sources as a single pre-joined frequency table with a separate severity
    table, not the two-table policies/claims shape insurabench expects.
    Building this loader means re-deriving explicit policy-period claim
    linkage from that source, which is future work, not part of the data
    layer itself.
    """
    raise NotImplementedError(
        "load_fremtpl2 is not yet implemented. Use "
        "insurabench.data.datasets.make_synthetic_two_table for now, or "
        "supply your own two-table data via insurabench.policies.loader / "
        "insurabench.claims.loader."
    )
