"""Chart functions for ``insurabench.curves`` outputs (design brief §8).

Every function here takes the plain DataFrame a ``curves`` function
already returned -- never a model, never a ``PolicyFrame`` -- and returns
``(fig, ax)`` (or ``(fig, axes)`` for a faceted plot). That keeps this
module a pure presentation layer: computing a curve and plotting it are
two separate, independently testable steps, and a caller building a
dashboard can recompute a curve once and hand it to both a plotting
function and, say, an export step without redoing the aggregation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurabench.viz.theme import (
    FIGSIZE,
    PALETTE,
    add_exposure_bars,
    combined_legend,
    finalize,
    new_axes,
    theme,
)


def _set_level_ticks(ax, levels, *, rotate: bool = True):
    x = np.arange(len(levels))
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=30 if rotate else 0, ha="right" if rotate else "center")
    return x


def plot_one_way_curve(
    curve: pd.DataFrame,
    *,
    feature_name: str | None = None,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Observed (with confidence band) vs. fitted by level, with exposure
    as background bars on a secondary axis -- the standard one-way
    rating-factor chart. Takes ``one_way_curve``'s output directly.

    Returns ``(fig, ax)``. ``ax`` is the foreground (rate) axes; the
    exposure bars live on ``ax.figure.axes[-1]`` if you need to adjust
    them further.
    """
    with theme():
        fig, ax = new_axes(ax, FIGSIZE)
        x = _set_level_ticks(ax, curve["level"])

        ax2 = add_exposure_bars(ax, x, curve["exposure"])

        ax.fill_between(
            x,
            curve["observed_lower"],
            curve["observed_upper"],
            color=PALETTE["observed"],
            alpha=0.18,
            zorder=3,
            label="Observed 95% band",
        )
        ax.plot(x, curve["observed"], marker="o", color=PALETTE["observed"], zorder=4, label="Observed")
        ax.plot(
            x,
            curve["fitted"],
            marker="s",
            linestyle="--",
            color=PALETTE["fitted"],
            zorder=4,
            label="Fitted",
        )

        ax.set_ylabel("Rate")
        ax.set_xlabel(feature_name or "Level")
        ax.set_title(title or (f"One-way curve: {feature_name}" if feature_name else "One-way curve"))
        combined_legend(ax, ax2)
        finalize(fig, save_path)
        return fig, ax


def plot_two_way_curve(
    curve: pd.DataFrame,
    *,
    value: str = "fitted",
    x_label: str | None = None,
    y_label: str | None = None,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Heatmap of ``value`` (``"fitted"`` or ``"observed"``) across the
    two features' levels, cell values annotated directly. Takes
    ``two_way_curve``'s output directly.

    Row/column order follows each level's first appearance in ``curve``
    (i.e. whatever order ``two_way_curve`` produced) rather than being
    re-sorted alphabetically, so a continuous feature's bins stay in
    their natural ascending order.
    """
    if value not in ("fitted", "observed"):
        raise ValueError(f"value must be 'fitted' or 'observed', got {value!r}.")

    with theme():
        x_levels = list(dict.fromkeys(curve["level_x"]))
        y_levels = list(dict.fromkeys(curve["level_y"]))
        pivot = curve.pivot(index="level_y", columns="level_x", values=value)
        pivot = pivot.reindex(index=y_levels, columns=x_levels)

        fig, ax = new_axes(ax, FIGSIZE)
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlBu_r", aspect="auto")
        ax.set_xticks(range(len(x_levels)))
        ax.set_xticklabels(x_levels, rotation=30, ha="right")
        ax.set_yticks(range(len(y_levels)))
        ax.set_yticklabels(y_levels)
        ax.grid(False)

        for i in range(len(y_levels)):
            for j in range(len(x_levels)):
                v = pivot.iat[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black", fontsize=9)

        fig.colorbar(im, ax=ax, label=value.capitalize())
        ax.set_xlabel(x_label or "")
        ax.set_ylabel(y_label or "")
        ax.set_title(title or f"Two-way curve ({value})")
        finalize(fig, save_path)
        return fig, ax


def _bar_relativity(ax, sub: pd.DataFrame, title: str):
    x = _set_level_ticks(ax, sub["level"])
    colors = [PALETTE["fitted"] if v >= 1 else PALETTE["model_b"] for v in sub["relativity"]]
    ax.bar(x, sub["relativity"], color=colors, zorder=2)
    ax.axhline(1.0, color=PALETTE["reference"], linestyle="--", linewidth=1, zorder=1)
    ax.set_ylabel("Relativity")
    ax.set_title(title)


def plot_relativity_table(
    table: pd.DataFrame,
    *,
    feature: str | None = None,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Bar chart of relativity by level, with a reference line at 1.0 (the
    base level). Takes ``relativity_table``'s output directly.

    ``feature=None`` (the default) facets every feature in ``table`` into
    its own stacked panel within one figure and returns ``(fig, axes)``
    (an array of Axes, one per feature, in the order features first
    appear in ``table``). Pass ``feature=`` to plot just one feature onto
    a single ``ax`` instead, returning ``(fig, ax)``.
    """
    with theme():
        if feature is not None:
            sub = table[table["feature"] == feature]
            if sub.empty:
                raise ValueError(f"No rows for feature={feature!r} in this relativity table.")
            fig, ax = new_axes(ax, FIGSIZE)
            _bar_relativity(ax, sub, title or f"Relativity: {feature}")
            finalize(fig, save_path)
            return fig, ax

        features = list(dict.fromkeys(table["feature"]))
        if ax is not None:
            raise ValueError(
                "Faceting every feature (feature=None) creates its own multi-panel "
                "figure and can't be drawn onto a single passed-in ax -- pass "
                "feature=<name> to plot one feature onto ax, or ax=None to facet."
            )
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            len(features), 1, figsize=(FIGSIZE[0], max(FIGSIZE[1] * 0.75, 3.0) * len(features))
        )
        axes = np.atleast_1d(axes)
        for sub_ax, feat in zip(axes, features, strict=True):
            _bar_relativity(sub_ax, table[table["feature"] == feat], feat)
        fig.suptitle(title or "Relativity table", fontsize=14, fontweight="bold")
        finalize(fig, save_path)
        return fig, axes


def plot_partial_dependence(
    pdp: pd.DataFrame,
    *,
    feature_name: str | None = None,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """The model's own fitted relationship to one feature, isolated from
    every other feature's real correlation with it. Takes
    ``partial_dependence``'s output directly.

    Deliberately sparser than ``plot_one_way_curve``: there's no
    "observed" line, no confidence band, and no exposure bars here --
    ``partial_dependence`` re-scores the *entire* book at each grid
    value in turn rather than grouping actual rows by their own value of
    the feature, so there's no per-level actual outcome or per-level
    exposure to show alongside it. This is a statement about what the
    model does, not about the data -- a single fitted line.
    """
    with theme():
        fig, ax = new_axes(ax, FIGSIZE)
        x = _set_level_ticks(ax, pdp["level"])
        ax.plot(x, pdp["partial_dependence"], marker="o", color=PALETTE["fitted"], zorder=3)
        ax.set_ylabel("Partial dependence (rate)")
        ax.set_xlabel(feature_name or "Level")
        ax.set_title(
            title or (f"Partial dependence: {feature_name}" if feature_name else "Partial dependence")
        )
        finalize(fig, save_path)
        return fig, ax
