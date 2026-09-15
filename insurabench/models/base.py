"""Common contract for pricing models (design brief §4).

Every pricing model -- GLM now, GBM in a later build-order step -- must
implement this same interface so curves/evaluation code written against
one works unchanged against the other. That interchangeability is the
design brief's stated critical requirement: whatever model type is fit, it
plugs into the curves/evaluation layer identically.

The ABC's methods are deliberately array/dataframe-based (X, y, weights) --
plain and sklearn-shaped -- rather than PolicyFrame-aware. PolicyFrame-aware
convenience (building X/y/weights from a PolicyFrame's views) lives on the
concrete model classes instead, so the ABC itself stays swappable and isn't
tied to insurabench's own data types.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from insurabench.data.schema import FeatureRole


class PricingModel(ABC):
    """Minimal fit/predict/relativities contract every wrapped model type
    must implement identically."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        sample_weight: np.ndarray | pd.Series | None = None,
        offset: np.ndarray | pd.Series | None = None,
    ) -> PricingModel:
        """Fit the model. Returns self (sklearn convention), so calls can
        be chained."""

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        *,
        offset: np.ndarray | pd.Series | None = None,
    ) -> np.ndarray:
        """Predict the mean response for X."""

    @abstractmethod
    def relativities(self) -> pd.DataFrame:
        """Model-level coefficient/relativity summary.

        NOTE: this is a raw model-coefficient view, not the curves-layer
        relativity table (design brief §5's ``relativity_table.py``, not
        built yet -- that's build order step 3). That one evaluates
        relativities against actual observed factor levels with base-level
        normalization and works identically for GLM or GBM; this is a
        quick per-model sanity check, specific to how this model type
        represents its coefficients.
        """


def prepare_features(df: pd.DataFrame, feature_roles: dict[str, FeatureRole]) -> pd.DataFrame:
    """Cast feature columns to the dtype a wrapped model expects, based on
    their declared ``FeatureRole`` (design brief §3.5 -- feature roles are
    declared once on the schema, not re-inferred per model, so every model
    treats a given column the same way).

    - CATEGORICAL columns -> pandas ``category`` dtype. glum (via tabmat)
      and LightGBM/XGBoost/CatBoost all encode ``category``-dtype columns
      directly without hand-rolled one-hot encoding -- but critically, a
      plain string/object-dtype column is *not* recognized: glum silently
      drops it from the design matrix (with only a warning, easy to miss),
      which would otherwise mean a configured rating factor quietly
      contributing nothing to the fit. Casting explicitly to ``category``
      here closes that off.
    - CONTINUOUS columns -> coerced to numeric.
    - SPATIAL columns are excluded. There is no spatial modeling component
      yet (design brief §7 / build order step 5); silently feeding a
      lat/long column into a GLM as an ordinary categorical or continuous
      term would hide the fact that it needs its own smoothed term, so it's
      dropped here rather than mishandled.

    Returns a new DataFrame containing only the declared, non-spatial
    feature columns -- in ``feature_roles`` order -- never the untouched
    input.
    """
    ordered_cols = [c for c, role in feature_roles.items() if role is not FeatureRole.SPATIAL]
    out = {}
    for col in ordered_cols:
        role = feature_roles[col]
        if role is FeatureRole.CATEGORICAL:
            out[col] = df[col].astype("category")
        else:
            out[col] = pd.to_numeric(df[col])
    return pd.DataFrame(out, index=df.index)[ordered_cols]
