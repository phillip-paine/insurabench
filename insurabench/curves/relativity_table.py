"""Standardized rating-factor relativity table (design brief §5).

Every rating-factor curve boils down, for actuarial consumption, to one
thing: each level's fitted relativity to a base/reference level. This is
the export format actuaries already expect -- a rating table -- built
directly on top of ``one_way_curve`` so it inherits the same exposure-
weighting and model-agnosticism without touching the model itself, only
the fitted values ``one_way_curve`` already computed.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from insurabench.curves._common import Target
from insurabench.curves.one_way import one_way_curve
from insurabench.data.policy_frame import PolicyFrame


def relativity_table(
    pf: PolicyFrame,
    target: Target,
    y_pred,
    *,
    features: list[str] | None = None,
    base: Literal["max_exposure", "first"] = "max_exposure",
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
    n_bins: int = 10,
    binning: Literal["quantile", "uniform"] = "quantile",
) -> pd.DataFrame:
    """One row per (feature, level): exposure, fitted value, and
    ``relativity`` -- the fitted value normalized to 1.0 at each feature's
    own base level, independently per feature (design brief §5's "base
    level normalization").

    Parameters
    ----------
    features:
        Which declared features to include; defaults to every feature in
        ``pf.policy_schema.feature_roles``.
    base:
        How to pick each feature's base level. ``"max_exposure"``
        (default) uses the highest-exposure level -- the common actuarial
        convention, and robust to which level happens to be encoded
        first. ``"first"`` uses whichever level ``one_way_curve`` returns
        first instead (the lowest bin for a continuous feature, the
        first category in encounter/category order for a categorical
        one).

    Notes
    -----
    A feature whose base-level fitted value is exactly 0 (e.g. a bin with
    a fitted rate of zero) produces an infinite or undefined relativity
    for every other level of that feature -- this is a property of the
    fitted model at that level, not a bug in the normalization, and is
    left as-is (inf/NaN) rather than silently clamped.
    """
    if features is None:
        features = list(pf.policy_schema.feature_roles)

    tables = []
    for feature in features:
        curve = one_way_curve(
            pf,
            feature,
            target,
            y_pred,
            peril=peril,
            coverage=coverage,
            n_bins=n_bins,
            binning=binning,
        ).copy()
        if base == "max_exposure":
            base_fitted = curve.loc[curve["exposure"].idxmax(), "fitted"]
        elif base == "first":
            base_fitted = curve["fitted"].iloc[0]
        else:
            raise ValueError(f"Unknown base {base!r}. Choose 'max_exposure' or 'first'.")

        curve.insert(0, "feature", feature)
        curve["relativity"] = curve["fitted"] / base_fitted
        tables.append(curve)

    out = pd.concat(tables, ignore_index=True)
    return out[["feature", "level", "n_obs", "exposure", "observed", "fitted", "relativity"]]
