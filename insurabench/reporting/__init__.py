"""Thin consumers of the curves/evaluation/models layers -- packaging
already-trusted outputs into the two shapes people actually want to
consume them in (design brief §8, build order step 6): a one-page model
diagnostic summary (``model_card``), and an export of the fitted
rating factors into the format actuaries already expect
(``export_rating_table``).

Deliberately built last (build order step 6): both modules are consumers
of curves/evaluation/models, not new statistical primitives, so building
them before those layers settled would just mean rework as those APIs
changed. One exception to "no new correctness risk" surfaced while
building this: see ``model_card``'s AIC/BIC handling, and
``insurabench.models.glm.GLMPricingModel.information_criteria``'s
docstring, for a real glum footgun found and fixed along the way.
"""
from __future__ import annotations

from insurabench.reporting.export import export_rating_table
from insurabench.reporting.model_card import ModelCard, generate_model_card

__all__ = ["ModelCard", "export_rating_table", "generate_model_card"]
