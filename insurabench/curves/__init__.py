"""Model-agnostic rating-factor curves (design brief §5).

``one_way_curve`` / ``relativity_table`` / ``two_way_curve`` never call a
model -- they take a plain ``y_pred`` array (whatever
``model.predict_policy_frame(pf)`` produced for any ``PricingModel``) and
are correct for any model type by construction. ``partial_dependence`` is
the one exception in this package: a PDP inherently re-predicts after
perturbing a feature, so it takes a fitted ``PricingModel`` directly
instead (see its docstring for why that's still model-agnostic in the
sense that matters).
"""
from __future__ import annotations

from insurabench.curves.one_way import one_way_curve
from insurabench.curves.partial_dependence import partial_dependence
from insurabench.curves.relativity_table import relativity_table
from insurabench.curves.two_way import two_way_curve

__all__ = ["one_way_curve", "partial_dependence", "relativity_table", "two_way_curve"]
