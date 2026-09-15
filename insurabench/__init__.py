"""insurabench: end-to-end non-life insurance statistical modeling for data scientists.

This top-level import surface only exposes the data layer so far (see the
project's build order: data layer -> GLM wrapper -> curves/evaluation ->
GBM wrapper -> geo -> reporting -> fairness stub). Modeling, curves, and
evaluation modules are added in later phases.
"""
from __future__ import annotations

from insurabench.data.policy_frame import PolicyFrame
from insurabench.data.schema import ClaimsSchema, FeatureRole, PolicySchema
from insurabench.exceptions import (
    InsurabenchError,
    LinkingError,
    PolicyIdentityError,
    SchemaError,
)

__all__ = [
    "ClaimsSchema",
    "FeatureRole",
    "InsurabenchError",
    "LinkingError",
    "PolicyFrame",
    "PolicyIdentityError",
    "PolicySchema",
    "SchemaError",
]

__version__ = "0.1.0.dev0"
