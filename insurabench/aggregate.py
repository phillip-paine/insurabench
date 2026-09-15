"""Reshape linked policies/claims into the two grains each model type wants
(design brief §3.3).

- Frequency modeling wants one row per policy-period: claim_count,
  exposure, rating factors.
- Severity modeling wants one row per *claim*, with its policy-period's
  rating factors joined on -- never rolled up to policy level.

This is a named, callable step deliberately: doing this reshape invisibly
inside a model's ``.fit()`` is exactly the kind of thing that hides bugs
(claims reported post-term-end, multiple claims per policy silently summed).

Every function here assumes ``insurabench.linking.validate_linkage`` has
already passed for the given tables/schemas -- they do not re-check for
orphans or out-of-term claims themselves, only for the ambiguity that
validate_linkage cannot detect without doing this same date-matching work
(a claim matching more than one policy-period).
"""
from __future__ import annotations

import pandas as pd

from insurabench.data.schema import ClaimsSchema, PolicySchema
from insurabench.exceptions import LinkingError, SchemaError


def _filter_claims(
    claims: pd.DataFrame,
    claims_schema: ClaimsSchema,
    *,
    peril: str | list[str] | None,
    coverage: str | list[str] | None,
) -> pd.DataFrame:
    """Restrict ``claims`` to one or more perils and/or coverages, so a user
    can build separate per-peril or per-coverage frequency/severity models
    (a common pattern: model each peril's frequency/severity independently,
    then sum the resulting pure premiums) while still being able to get the
    combined view by passing nothing.
    """
    if peril is not None:
        if claims_schema.peril_col is None:
            raise SchemaError(
                "peril filter requested but this ClaimsSchema has no "
                "peril_col configured."
            )
        values = [peril] if isinstance(peril, str) else list(peril)
        claims = claims[claims[claims_schema.peril_col].isin(values)]
    if coverage is not None:
        if claims_schema.coverage_col is None:
            raise SchemaError(
                "coverage filter requested but this ClaimsSchema has no "
                "coverage_col configured."
            )
        values = [coverage] if isinstance(coverage, str) else list(coverage)
        claims = claims[claims[claims_schema.coverage_col].isin(values)]
    return claims


def _match_claims_to_periods(
    policies: pd.DataFrame,
    claims: pd.DataFrame,
    policy_schema: PolicySchema,
    claims_schema: ClaimsSchema,
) -> pd.DataFrame:
    """Attach the policy-period key to every claim row.

    When ``policy_id`` alone is unique (no ``term_start_col`` configured),
    this is a plain merge -- unambiguous by construction. When policies
    recur across renewal terms, a claim must be matched to the *specific*
    term whose [term_start, term_end) window contains its claim_date --
    matching on policy_id alone would double-count claims/exposure across
    terms (design brief §3.3).

    Raises
    ------
    LinkingError
        A claim matches more than one policy-period (overlapping terms for
        the same policy id). This should have been caught by
        ``insurabench.linking.validate_linkage`` first; this is a backstop.
    """
    claim_policy_id_col = claims_schema.policy_id_col
    claim_date_col = claims_schema.claim_date_col
    claim_id_col = claims_schema.claim_id_col
    policy_id_col = policy_schema.policy_id_col
    term_start_col = policy_schema.term_start_col
    term_end_col = policy_schema.term_end_col

    # Bring the policy side in under a private column name so the merge
    # never depends on (or breaks due to) policy_id_col and
    # claim_policy_id_col happening to share -- or not share -- a name.
    if term_start_col is None:
        # Bare policy_id is already a unique key -> direct merge is unambiguous.
        policy_side = policies[[policy_id_col]].rename(columns={policy_id_col: "__policy_id"})
        matched = claims.merge(
            policy_side, left_on=claim_policy_id_col, right_on="__policy_id", how="left"
        )
    else:
        policy_side = policies[[policy_id_col, term_start_col, term_end_col]].rename(
            columns={policy_id_col: "__policy_id"}
        )
        candidates = claims.merge(
            policy_side, left_on=claim_policy_id_col, right_on="__policy_id", how="left"
        )
        in_window = (
            (candidates[claim_date_col] >= candidates[term_start_col])
            & (candidates[claim_date_col] < candidates[term_end_col])
        )
        matched = candidates.loc[in_window]

        dup = matched[claim_id_col].duplicated(keep=False)
        if dup.any():
            bad = matched.loc[dup, claim_id_col].unique()[:10].tolist()
            raise LinkingError(
                "Claim(s) match more than one policy-period -- overlapping "
                f"terms for the same policy id? Example claim id(s): {bad}. "
                "Run insurabench.linking.validate_linkage first to catch "
                "this before aggregating."
            )

    # Normalize the matched policy id back onto policy_id_col so downstream
    # code can group/merge by policy_schema.period_key_cols regardless of
    # whether the two tables happened to name the FK the same thing.
    if policy_id_col == claim_policy_id_col:
        matched = matched.drop(columns="__policy_id")
    else:
        matched = matched.rename(columns={"__policy_id": policy_id_col})
    return matched


def to_frequency_frame(
    policies: pd.DataFrame,
    claims: pd.DataFrame,
    policy_schema: PolicySchema,
    claims_schema: ClaimsSchema,
    *,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> pd.DataFrame:
    """One row per policy-period: ``claim_count``, exposure, and rating
    factors. This is what frequency models (Poisson/NB GLM or GBM with a
    count objective, exposure as offset) consume.

    Policies with zero claims are kept with ``claim_count == 0`` -- this is
    a left join from ``policies``, not an aggregation of ``claims``, so
    exposure with no claims is never silently dropped.

    ``peril``/``coverage`` restrict which claims are counted (e.g.
    ``peril="theft"`` to fit a theft-only frequency model), requiring the
    corresponding column to be set on ``claims_schema``. Leave both as
    ``None`` for the combined, all-perils/all-coverages count.
    """
    claims = _filter_claims(claims, claims_schema, peril=peril, coverage=coverage)
    key_cols = policy_schema.period_key_cols
    matched = _match_claims_to_periods(policies, claims, policy_schema, claims_schema)

    counts = matched.groupby(key_cols)[claims_schema.claim_id_col].count()
    out = policies.merge(
        counts.rename("claim_count"), left_on=key_cols, right_index=True, how="left"
    )
    out["claim_count"] = out["claim_count"].fillna(0).astype(int)

    cols = [*key_cols, policy_schema.exposure_col, *policy_schema.feature_cols, "claim_count"]
    if policy_schema.premium_col:
        cols.append(policy_schema.premium_col)
    cols = list(dict.fromkeys(cols))  # de-duplicate, preserve order
    return out[cols].reset_index(drop=True)


def to_severity_frame(
    policies: pd.DataFrame,
    claims: pd.DataFrame,
    policy_schema: PolicySchema,
    claims_schema: ClaimsSchema,
    *,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> pd.DataFrame:
    """One row per *claim*: ``claim_amount`` plus the rating factors of the
    exact policy-period it occurred under. Never rolled up to policy level
    -- a policy-period with three claims contributes three rows, each
    carrying the same features. This is what severity models (Gamma
    GLM/GBM) consume.

    ``peril``/``coverage`` restrict which claims are returned -- see
    ``to_frequency_frame`` for the same parameters. The ``peril_col``/
    ``coverage_col`` themselves (if configured) are always included in the
    output regardless of filtering, so a combined severity frame can still
    be split or grouped by them downstream.
    """
    claims = _filter_claims(claims, claims_schema, peril=peril, coverage=coverage)
    feature_cols = policy_schema.feature_cols
    key_cols = policy_schema.period_key_cols
    matched = _match_claims_to_periods(policies, claims, policy_schema, claims_schema)
    # Join on the full period key (not just policy_id) -- a claim is already
    # attributed to one specific policy-period by _match_claims_to_periods,
    # and joining on bare policy_id here would re-fan-out across every
    # renewal term that policy_id has, undoing that attribution.
    policy_side = policies[[*key_cols, *feature_cols]]
    merged = matched.merge(policy_side, on=key_cols, how="left")

    cols = [
        claims_schema.claim_id_col,
        claims_schema.policy_id_col,
        claims_schema.claim_date_col,
        claims_schema.claim_amount_col,
        *feature_cols,
    ]
    if claims_schema.peril_col:
        cols.append(claims_schema.peril_col)
    if claims_schema.coverage_col:
        cols.append(claims_schema.coverage_col)
    cols = list(dict.fromkeys(cols))
    return merged[cols].reset_index(drop=True)
