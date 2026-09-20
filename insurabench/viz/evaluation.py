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
