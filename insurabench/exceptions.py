"""Custom exceptions for insurabench.

Design principle (see design brief §3.1, §9): data-layer problems must fail
loudly. Nothing in this package silently drops rows or coerces ambiguous
data into a guessed shape -- every failure mode below is meant to be raised
with enough detail (counts, example ids) that a user can go fix the source
data or their schema mapping.
"""
from __future__ import annotations


class InsurabenchError(Exception):
    """Base class for all insurabench errors."""


class SchemaError(InsurabenchError):
    """A table is missing required columns, or a column has the wrong dtype
    or an invalid value (e.g. non-positive exposure) for what insurabench
    needs to do with it."""


class PolicyIdentityError(InsurabenchError):
    """Policy-period identity is ambiguous (design brief §3.2).

    Typically this means ``policy_id`` repeats across rows (e.g. renewal
    terms) and no ``term_start_col`` was supplied to disambiguate, or the
    (policy_id, term_start) pair itself isn't unique.
    """


class LinkingError(InsurabenchError):
    """The policies/claims join failed validation: orphan claims, claims
    dated outside their policy-period's term, or a claim matching more than
    one policy-period. See design brief §3.1."""
