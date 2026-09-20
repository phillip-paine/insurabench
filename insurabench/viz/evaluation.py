"""Chart functions for ``insurabench.evaluation`` outputs (design brief
§8). Same contract as ``insurabench.viz.curves``: every function here
takes a plain DataFrame (or float) an ``evaluation`` function already
returned, never a model, and returns ``(fig, ax)``.
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


def plot_lift_chart(
    chart: pd.DataFrame,
    *,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Actual vs. predicted by decile, exposure as background bars. Takes
    ``lift_chart``'s output directly.
    """
    with theme():
        fig, ax = new_axes(ax, FIGSIZE)
        x = np.arange(len(chart))
        ax2 = add_exposure_bars(ax, x, chart["exposure"])

        ax.plot(x, chart["actual"], marker="o", color=PALETTE["observed"], zorder=4, label="Actual")
        ax.plot(
            x,
            chart["predicted"],
            marker="s",
            linestyle="--",
            color=PALETTE["fitted"],
            zorder=4,
            label="Predicted",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(chart["decile"])
        ax.set_xlabel("Decile (ascending predicted value)")
        ax.set_ylabel("Rate")
        ax.set_title(title or "Lift chart")
        combined_legend(ax, ax2)
        finalize(fig, save_path)
        return fig, ax


def plot_double_lift_chart(
    chart: pd.DataFrame,
    *,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Actual vs. both models' predictions by disagreement-ranked decile,
    exposure as background bars. Takes ``double_lift_chart``'s output
    directly -- ``model_a_name``/``model_b_name`` must match whatever was
    passed to ``double_lift_chart`` (both default to ``"model_a"``/
    ``"model_b"``, so if you didn't rename them there, leave these as-is).
    """
    for name in (model_a_name, model_b_name):
        if name not in chart.columns:
            raise ValueError(
                f"{name!r} is not a column in this chart ({list(chart.columns)}) -- "
                f"pass the same model_a_name/model_b_name used when building it "
                f"with double_lift_chart."
            )

    with theme():
        fig, ax = new_axes(ax, FIGSIZE)
        x = np.arange(len(chart))
        ax2 = add_exposure_bars(ax, x, chart["exposure"])

        ax.plot(x, chart["actual"], marker="o", color=PALETTE["observed"], zorder=4, label="Actual")
        ax.plot(
            x,
            chart[model_a_name],
            marker="s",
            linestyle="--",
            color=PALETTE["fitted"],
            zorder=4,
            label=model_a_name,
        )
        ax.plot(
            x,
            chart[model_b_name],
            marker="^",
            linestyle=":",
            color=PALETTE["model_b"],
            zorder=4,
            label=model_b_name,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(chart["decile"])
        ax.set_xlabel("Decile (ascending model_a / model_b ratio)")
        ax.set_ylabel("Rate")
        ax.set_title(title or "Double lift chart")
        combined_legend(ax, ax2)
        finalize(fig, save_path)
        return fig, ax


def plot_gini_curve(
    curve: pd.DataFrame,
    *,
    gini_value: float | None = None,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Ordered Lorenz curve against the random-ranking diagonal, with an
    optional Gini annotation. Takes ``gini_curve``'s output directly;
    pass ``gini_value=gini_index(...)`` to annotate the scalar alongside
    the curve.
    """
    with theme():
        fig, ax = new_axes(ax, FIGSIZE)
        ax.plot(curve["cum_exposure"], curve["cum_loss"], color=PALETTE["fitted"], zorder=3, label="Model")
        ax.plot([0, 1], [0, 1], color=PALETTE["reference"], linestyle="--", zorder=2, label="Random")
        ax.fill_between(
            curve["cum_exposure"], curve["cum_loss"], curve["cum_exposure"], color=PALETTE["fitted"], alpha=0.12
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Cumulative exposure share")
        ax.set_ylabel("Cumulative loss share")
        ax.set_title(title or "Lorenz curve")
        if gini_value is not None:
            ax.text(
                0.05,
                0.92,
                f"Gini = {gini_value:.3f}",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
            )
        ax.legend(loc="lower right")
        finalize(fig, save_path)
        return fig, ax


def plot_calibration_table(
    table: pd.DataFrame,
    *,
    confidence: float = 0.95,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Observed (with an approximate confidence band from
    ``standard_error``) vs. fitted by decile, exposure as background
    bars, flagged (``significant``) buckets marked. Takes
    ``calibration_table``'s output directly.

    ``confidence`` only controls the width of the *displayed* error bars
    (``z_crit * standard_error``, ``z_crit`` from ``confidence``) -- it
    does not re-run the significance test itself, which
    ``calibration_table`` already fixed via its own ``confidence``
    argument (the ``significant`` column reflects that, not this one).
    """
    from scipy.stats import norm

    z_crit = float(norm.ppf(0.5 + confidence / 2))

    with theme():
        fig, ax = new_axes(ax, FIGSIZE)
        x = np.arange(len(table))
        ax2 = add_exposure_bars(ax, x, table["exposure"])

        yerr = z_crit * table["standard_error"].fillna(0.0)
        ax.errorbar(
            x,
            table["observed"],
            yerr=yerr,
            fmt="o",
            color=PALETTE["observed"],
            capsize=3,
            zorder=4,
            label="Observed (approx. band)",
        )
        ax.plot(
            x, table["fitted"], marker="s", linestyle="--", color=PALETTE["fitted"], zorder=4, label="Fitted"
        )

        sig = table["significant"].to_numpy(dtype=bool)
        if sig.any():
            ax.scatter(
                x[sig],
                table["observed"].to_numpy()[sig],
                marker="*",
                s=150,
                color=PALETTE["model_b"],
                zorder=5,
                label="Flagged",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(table["decile"])
        ax.set_xlabel("Decile (ascending predicted value)")
        ax.set_ylabel("Rate")
        ax.set_title(title or "Calibration")
        combined_legend(ax, ax2)
        finalize(fig, save_path)
        return fig, ax


def plot_stability_summary(
    summary: pd.DataFrame,
    *,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Mean relativity per level with a min-max range bar, reference line
    at 1.0. Takes ``stability_summary``'s output directly (from either
    ``bootstrap_relativities`` or ``split_relativities``).
    """
    with theme():
        fig, ax = new_axes(ax, FIGSIZE)
        x = np.arange(len(summary))
        yerr = np.vstack(
            [
                summary["relativity_mean"] - summary["relativity_min"],
                summary["relativity_max"] - summary["relativity_mean"],
            ]
        )
        ax.errorbar(
            x,
            summary["relativity_mean"],
            yerr=yerr,
            fmt="o",
            color=PALETTE["fitted"],
            capsize=4,
            zorder=3,
            label="Mean (min\u2013max range)",
        )
        ax.axhline(1.0, color=PALETTE["reference"], linestyle="--", linewidth=1, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["level"], rotation=30, ha="right")
        ax.set_ylabel("Relativity")
        ax.set_title(title or "Relativity stability")
        ax.legend(loc="upper left")
        finalize(fig, save_path)
        return fig, ax


def plot_rate_change_by_level(
    table: pd.DataFrame,
    *,
    feature_name: str | None = None,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Aggregate rate change per level (bars, colored by direction) with
    the split of each level's exposure that moves up vs. down annotated
    on top -- the segment-level view of a repricing impact. Takes
    ``rate_change_by_level``'s output directly.
    """
    with theme():
        fig, ax = new_axes(ax, FIGSIZE)
        x = np.arange(len(table))
        pct = table["pct_change"].to_numpy() * 100
        colors = [PALETTE["model_b"] if v >= 0 else PALETTE["fitted"] for v in pct]
        ax.bar(x, pct, color=colors, zorder=2)
        ax.axhline(0.0, color=PALETTE["reference"], linestyle="-", linewidth=1, zorder=1)

        # Fix ylim to a data-range-proportional padding *before* placing
        # the annotation text, and keep the text offset within that same
        # padding -- otherwise a fixed absolute offset either sits on top
        # of the bar (when the data range is tiny) or barely clears it
        # (when huge), and in both cases can end up outside whatever
        # ylim autoscale settles on, which then blows out the saved
        # figure's canvas via savefig's bbox_inches="tight" instead of
        # just clipping the text.
        y_min, y_max = min(0.0, float(np.nanmin(pct))), max(0.0, float(np.nanmax(pct)))
        pad = max((y_max - y_min) * 0.18, 0.05)
        ax.set_ylim(y_min - pad * 1.6, y_max + pad * 1.6)

        for xi, row in zip(x, table.itertuples(), strict=True):
            up = row.pct_exposure_increasing
            down = row.pct_exposure_decreasing
            if np.isnan(up) or np.isnan(down):
                continue
            offset = pad if row.pct_change >= 0 else -pad
            va = "bottom" if row.pct_change >= 0 else "top"
            ax.text(
                xi,
                row.pct_change * 100 + offset,
                f"{up:.0%} up / {down:.0%} down",
                ha="center",
                va=va,
                fontsize=8,
                color="#4D4D4D",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(table["level"], rotation=30, ha="right")
        ax.set_ylabel("Aggregate rate change (%)")
        ax.set_xlabel(feature_name or "Level")
        ax.set_title(title or (f"Rate change: {feature_name}" if feature_name else "Rate change by level"))
        finalize(fig, save_path)
        return fig, ax


def plot_rate_change_distribution(
    table: pd.DataFrame,
    *,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Exposure share by rate-change band, ascending -- the whole-book
    disruption summary ("what % of the book sees more than a 10%
    increase?"). Takes ``rate_change_distribution``'s output directly.
    Bars for a decreasing-rate band are shaded blue, increasing-rate red,
    and the (rare, usually empty) "undefined" band gray -- inferred from
    each band's label, since the table itself only carries the label
    string.
    """
    with theme():
        fig, ax = new_axes(ax, FIGSIZE)
        x = np.arange(len(table))

        def _color(label: str) -> str:
            if label.startswith("undefined"):
                return PALETTE["exposure"]
            # label looks like "(-10%, -5%]" or "(+5%, +10%]" -- sign of
            # the upper (second) edge decides which side of 0% it's on.
            upper = label.split(",")[1].strip(" ]%")
            return PALETTE["fitted"] if upper.startswith("-") else PALETTE["model_b"]

        colors = [_color(label) for label in table["band"]]
        bars = ax.bar(x, table["exposure_share"] * 100, color=colors, zorder=2)
        for bar, share in zip(bars, table["exposure_share"], strict=True):
            if share > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{share:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(table["band"], rotation=30, ha="right")
        ax.set_ylabel("Share of exposure (%)")
        ax.set_xlabel("Rate change band")
        ax.set_title(title or "Rate change distribution")
        finalize(fig, save_path)
        return fig, ax
