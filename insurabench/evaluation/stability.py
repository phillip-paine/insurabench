"""Relativity stability across bootstrap resamples and time/other splits
(design brief §6).

A relativity table (``curves.relativity_table``) is a point estimate --
it says nothing about how much that estimate would move under a
different sample. This module answers that directly, two ways:
``bootstrap_relativities`` resamples rows to show sampling variability at
fixed data; ``split_relativities`` recomputes relativities independently
within caller-supplied groups (typically time periods, e.g. policy year)
to show whether a rating factor's effect is actually stable *in time*,
not just under resampling. Both return a long table with one row per
(group, level); ``stability_summary`` collapses either into a per-level
spread summary.

Neither function calls a model -- like the rest of ``curves``/
``evaluation``, they work from an already-computed ``y_pred`` array, so
they're equally usable for a GLM or a GBM's predictions (design brief
§4).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from insurabench.curves._common import Target, build_target_frame
from insurabench.curves.one_way import _level_series
from insurabench.data.policy_frame import PolicyFrame


def _level_totals(frame: pd.DataFrame) -> pd.Series:
    """Fitted value per level: sum(predicted_numerator) / sum(weight)."""
    agg = frame.groupby("level", observed=True).agg(
        weight_sum=("weight", "sum"), fitted_num=("predicted_numerator", "sum")
    )
    return agg["fitted_num"] / agg["weight_sum"]


def _base_level(fitted: pd.Series, weight_by_level: pd.Series, base: str):
    if base == "max_exposure":
        return weight_by_level.idxmax()
    if base == "first":
        return fitted.index[0]
    raise ValueError(f"Unknown base {base!r}. Choose 'max_exposure' or 'first'.")


def _prepare(
    pf: PolicyFrame,
    feature: str,
    target: Target,
    y_pred,
    *,
    n_bins: int,
    binning: str,
    peril,
    coverage,
):
    if feature not in pf.policy_schema.feature_roles:
        raise ValueError(
            f"{feature!r} is not a declared feature on this PolicyFrame's "
            f"PolicySchema (feature_roles: {list(pf.policy_schema.feature_roles)})."
        )
    tf = build_target_frame(pf, target, y_pred, peril=peril, coverage=coverage)
    role = pf.policy_schema.feature_roles[feature]
    levels = _level_series(tf.view[feature], role, n_bins=n_bins, binning=binning)
    frame = pd.DataFrame(
        {"level": levels, "weight": tf.weight, "predicted_numerator": tf.predicted_numerator}
    )
    full_fitted = _level_totals(frame)
    weight_by_level = frame.groupby("level", observed=True)["weight"].sum()
    return frame, full_fitted, weight_by_level


def bootstrap_relativities(
    pf: PolicyFrame,
    feature: str,
    target: Target,
    y_pred,
    *,
    base: Literal["max_exposure", "first"] = "max_exposure",
    n_bins: int = 10,
    binning: Literal["quantile", "uniform"] = "quantile",
    n_bootstrap: int = 200,
    seed: int | None = None,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> pd.DataFrame:
    """Row-level bootstrap resample of ``feature``'s relativities.

    The base level (see ``base``) is fixed once from the full, un-
    resampled data -- not re-selected inside each bootstrap draw -- so
    that draws are always normalized against the same reference level.
    Re-picking the base level per draw would let the normalization
    itself flip between draws for a feature with two similarly-exposed
    levels, which would inflate the apparent instability of every
    *other* level's relativity for a reason that has nothing to do with
    the model.

    Returns
    -------
    DataFrame, one row per (draw, level) with at least one row that
    resampled into it: ``draw`` (0-indexed), ``level``, ``relativity``.
    A draw in which the base level itself didn't get resampled at all,
    or resampled to a fitted value of exactly 0, is omitted entirely
    (relativity would be undefined) -- see ``n_bootstrap`` in
    ``stability_summary``'s output to check how many draws actually
    contributed to a given level's summary.
    """
    frame, full_fitted, weight_by_level = _prepare(
        pf, feature, target, y_pred, n_bins=n_bins, binning=binning, peril=peril, coverage=coverage
    )
    base_level = _base_level(full_fitted, weight_by_level, base)

    rng = np.random.default_rng(seed)
    n = len(frame)
    rows = []
    for draw in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        fitted = _level_totals(frame.iloc[idx])
        if base_level not in fitted.index or fitted.loc[base_level] == 0:
            continue
        relativity = fitted / fitted.loc[base_level]
        for level, value in relativity.items():
            rows.append({"draw": draw, "level": str(level), "relativity": float(value)})
    return pd.DataFrame(rows)


def split_relativities(
    pf: PolicyFrame,
    feature: str,
    target: Target,
    y_pred,
    split,
    *,
    base: Literal["max_exposure", "first"] = "max_exposure",
    n_bins: int = 10,
    binning: Literal["quantile", "uniform"] = "quantile",
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
) -> pd.DataFrame:
    """``feature``'s relativities computed independently within each group
    of ``split`` -- typically calendar/policy year, but any array-like
    grouping aligned to the target view's row order works (e.g. a
    train/test id, a geographic region held out one at a time).

    The base level is fixed from the full (unsplit) data, for the same
    reason given in ``bootstrap_relativities``.

    Parameters
    ----------
    split:
        Array-like the same length as the ``target`` view for this
        ``pf``/``peril``/``coverage`` (i.e. ``pf.frequency_view(...)``'s
        row count for ``target="frequency"``, etc.) -- raises if the
        length doesn't match, which usually means ``split`` was built
        against a different view.

    Returns
    -------
    DataFrame, one row per (split value, level) with at least one
    observation in it: ``split``, ``level``, ``exposure``,
    ``relativity``.
    """
    frame, full_fitted, weight_by_level = _prepare(
        pf, feature, target, y_pred, n_bins=n_bins, binning=binning, peril=peril, coverage=coverage
    )
    split = np.asarray(split)
    if len(split) != len(frame):
        raise ValueError(
            f"split has {len(split)} element(s) but the {target!r} view for "
            f"this PolicyFrame/peril/coverage has {len(frame)} row(s) -- "
            f"split must be aligned to that same view."
        )
    base_level = _base_level(full_fitted, weight_by_level, base)

    frame = frame.assign(split=split)
    rows = []
    for split_value, g in frame.groupby("split"):
        fitted = _level_totals(g)
        if base_level not in fitted.index or fitted.loc[base_level] == 0:
            continue
        weight_by_level_g = g.groupby("level", observed=True)["weight"].sum()
        relativity = fitted / fitted.loc[base_level]
        for level, value in relativity.items():
            rows.append(
                {
                    "split": split_value,
                    "level": str(level),
                    "exposure": float(weight_by_level_g.loc[level]),
                    "relativity": float(value),
                }
            )
    return pd.DataFrame(rows)


def stability_summary(long_table: pd.DataFrame) -> pd.DataFrame:
    """Collapse a ``bootstrap_relativities``/``split_relativities`` table
    to one row per level: ``n``, ``relativity_mean``, ``relativity_std``,
    ``relativity_min``, ``relativity_max``, ``relativity_range``
    (``max - min``, the simplest and most interpretable "how much does
    this move" number -- a large range on a level with a small ``n`` is
    itself informative, so both are reported side by side rather than
    only the spread).
    """
    grouped = long_table.groupby("level")["relativity"]
    out = grouped.agg(n="count", relativity_mean="mean", relativity_min="min", relativity_max="max")
    out["relativity_std"] = grouped.std(ddof=1).fillna(0.0)
    out["relativity_range"] = out["relativity_max"] - out["relativity_min"]
    return out.reset_index()[
        ["level", "n", "relativity_mean", "relativity_std", "relativity_min", "relativity_max", "relativity_range"]
    ]
