"""insurabench: end-to-end non-life insurance statistical modeling for
data scientists.

Implemented so far (see the project's build order): the data layer
(``PolicyFrame`` and friends), two swappable pricing model wrappers
(``GLMPricingModel``, ``GBMPricingModel`` -- the latter including the
zero-inflated Poisson boosted tree from So (2024) for frequency),
``FrequencySeverityModel`` (composing two independently-fit
``PricingModel``s -- either GLM or GBM -- into a pure-premium model,
the alternative to a single model's direct Tweedie
``target="pure_premium"`` route), and the full curves and evaluation
layers, proven model-agnostic against
both wrappers -- one-way/two-way curves, partial dependence, relativity
tables, lift/double-lift/Gini/calibration/stability, and repricing
rate-impact analysis -- plus a matching ``insurabench.viz`` plotting
layer for all of it, and ``insurabench.reporting`` (``generate_model_card``,
``export_rating_table``) for turning a fitted model into a one-page
diagnostic summary or an actuarial-format rating-table export. Not yet
built: geo and the fairness stub.

This top-level module re-exports the most commonly used names from each
layer; the submodules (``insurabench.curves``, ``insurabench.evaluation``,
``insurabench.viz``, ``insurabench.models``) are the complete, canonical
list in each case.
"""
from __future__ import annotations

from insurabench.curves import (
    one_way_curve,
    partial_dependence,
    relativity_table,
    two_way_curve,
)
from insurabench.data.policy_frame import PolicyFrame
from insurabench.data.schema import ClaimsSchema, FeatureRole, PolicySchema
from insurabench.evaluation import (
    bootstrap_relativities,
    calibration_index,
    calibration_table,
    double_lift_chart,
    gini_curve,
    gini_index,
    lift_chart,
    rate_change_by_level,
    rate_change_distribution,
    split_relativities,
    stability_summary,
)
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
from insurabench.models.frequency_severity import FrequencySeverityModel
from insurabench.models.gbm import GBMPricingModel
from insurabench.models.glm import GLMPricingModel
from insurabench.reporting import ModelCard, export_rating_table, generate_model_card

__all__ = [
    "ClaimsSchema",
    "FeatureRole",
    "FrequencySeverityModel",
    "GBMPricingModel",
    "GLMPricingModel",
    "InsurabenchError",
    "LinkingError",
    "ModelCard",
    "PolicyFrame",
    "PolicyIdentityError",
    "PolicySchema",
    "SchemaError",
    "bootstrap_relativities",
    "calibration_index",
    "calibration_table",
    "double_lift_chart",
    "export_rating_table",
    "generate_model_card",
    "gini_curve",
    "gini_index",
    "lift_chart",
    "one_way_curve",
    "partial_dependence",
    "rate_change_by_level",
    "rate_change_distribution",
    "relativity_table",
    "split_relativities",
    "stability_summary",
    "train_test_split_policy_frame",
    "train_val_test_split_policy_frame",
    "two_way_curve",
]

__version__ = "0.2.0.dev0"
