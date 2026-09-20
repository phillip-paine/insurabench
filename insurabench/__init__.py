"""insurabench: end-to-end non-life insurance statistical modeling for data scientists.

Implemented so far (see the project's build order): the data layer
(``PolicyFrame`` and friends), the GLM wrapper, and -- proven against that
GLM wrapper -- the core of the curves and evaluation layers
(``one_way_curve``, ``relativity_table``, ``lift_chart``, ``gini_index``).
Not yet built: ``curves.two_way``/``curves.partial_dependence``,
``evaluation.double_lift``/``calibration``/``stability``, the GBM wrapper,
geo, and reporting.
"""
from __future__ import annotations

from insurabench.curves import one_way_curve, relativity_table
from insurabench.data.policy_frame import PolicyFrame
from insurabench.data.schema import ClaimsSchema, FeatureRole, PolicySchema
from insurabench.evaluation import gini_curve, gini_index, lift_chart
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
    "gini_curve",
    "gini_index",
    "lift_chart",
    "one_way_curve",
    "relativity_table",
    "train_test_split_policy_frame",
    "train_val_test_split_policy_frame",
]

__version__ = "0.1.0.dev0"
