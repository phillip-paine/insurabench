"""Splitting a PolicyFrame for train/validation/test workflows.

The split happens at the *policy-period* grain, not by shuffling policies
and claims independently -- every claim must stay with the policy-period it
belongs to (design brief §3.3). Splitting policies and claims separately
(e.g. a naive random split of each table) would let a policy-period land in
the training set while one of its own claims lands in the test set,
silently leaking information between the two.

This is a simple random split: no stratification, no time-based holdout.
Both are natural extensions once the modeling layer needs one (a
time-based split, in particular, is usually the right choice for a real
pricing exercise, to check the model still holds up on more recent data --
that's future work, not part of the data layer itself).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurabench.aggregate import match_claims_to_periods
from insurabench.data.policy_frame import PolicyFrame


def _split_indices(n: int, fractions: list[float], seed: int | None) -> list[np.ndarray]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions} (sum={sum(fractions)}).")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut_points = np.cumsum([round(f * n) for f in fractions[:-1]])
    return np.split(order, cut_points)


def _subset(pf: PolicyFrame, policy_row_idx: np.ndarray, matched_claims: pd.DataFrame) -> PolicyFrame:
    key_cols = pf.policy_schema.period_key_cols
    sub_policies = pf.policies.iloc[policy_row_idx].reset_index(drop=True)

    sub_keys = sub_policies[key_cols].drop_duplicates()
    sub_claim_ids = matched_claims.merge(sub_keys, on=key_cols, how="inner")[
        pf.claims_schema.claim_id_col
    ]
    sub_claims = pf.claims[pf.claims[pf.claims_schema.claim_id_col].isin(sub_claim_ids)].reset_index(
        drop=True
    )
    return PolicyFrame.from_tables(
        sub_policies, sub_claims, pf.policy_schema, pf.claims_schema, verbose=False
    )


def train_test_split_policy_frame(
    pf: PolicyFrame, test_size: float = 0.2, *, seed: int | None = None
) -> tuple[PolicyFrame, PolicyFrame]:
    """Split ``pf`` into train/test PolicyFrames at the policy-period grain.

    Every claim stays with the policy-period it belongs to, so severity
    rows can never leak across the split (see module docstring).
    """
    matched = match_claims_to_periods(pf.policies, pf.claims, pf.policy_schema, pf.claims_schema)
    train_idx, test_idx = _split_indices(len(pf.policies), [1 - test_size, test_size], seed)
    return _subset(pf, train_idx, matched), _subset(pf, test_idx, matched)


def train_val_test_split_policy_frame(
    pf: PolicyFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    *,
    seed: int | None = None,
) -> tuple[PolicyFrame, PolicyFrame, PolicyFrame]:
    """Split ``pf`` into train/validation/test PolicyFrames at the
    policy-period grain. ``val_size``+``test_size`` must be < 1.0; the
    remainder is the training fraction.
    """
    train_size = 1 - val_size - test_size
    if train_size <= 0:
        raise ValueError(
            f"val_size + test_size must be < 1.0, got {val_size} + {test_size}."
        )
    matched = match_claims_to_periods(pf.policies, pf.claims, pf.policy_schema, pf.claims_schema)
    train_idx, val_idx, test_idx = _split_indices(
        len(pf.policies), [train_size, val_size, test_size], seed
    )
    return (
        _subset(pf, train_idx, matched),
        _subset(pf, val_idx, matched),
        _subset(pf, test_idx, matched),
    )
