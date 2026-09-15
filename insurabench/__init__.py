"""insurabench: end-to-end non-life insurance statistical modeling for data scientists.

Implemented so far (see the project's build order): the data layer
(``PolicyFrame`` and friends) and the GLM wrapper. Curves, evaluation, the
GBM wrapper, geo, and reporting modules are not implemented yet.
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
from insurabench.model_selection import (
    train_test_split_policy_frame,
    train_val_test_split_policy_frame,
)
from insurabench.models.glm import GLMPricingModel

__all__ = [
    "ClaimsSchema",
    "FeatureRole",
    "GLMPricingModel",
    "InsurabenchError",
    "LinkingError",
    "PolicyFrame",
    "PolicyIdentityError",
    "PolicySchema",
    "SchemaError",
    "train_test_split_policy_frame",
    "train_val_test_split_policy_frame",
]

__version__ = "0.1.0.dev0"
