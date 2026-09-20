"""Shared plotting defaults (design brief §8: ``viz/theme.py``).

One consistent look across every chart in ``insurabench.viz`` -- the
point of a shared theme module is that a one-way curve, a lift chart, and
a calibration plot dropped into the same slide deck read as one family of
charts, not three different libraries' defaults stitched together.

Every public plotting function wraps its drawing code in ``with
theme():`` (a ``matplotlib.rc_context``), so the style never leaks into
the caller's own matplotlib state -- safe to call from a notebook that
has its own rcParams, or repeatedly inside a dashboard process.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

#: Semantic color roles, not raw hex codes, so a palette change is a
#: one-line edit here rather than a find-and-replace across every plot
#: function. Chosen for reasonable colorblind-safety (blue/red-orange
#: rather than red/green) and to read cleanly in both a slide projector
#: and a dark-mode dashboard.
PALETTE = {
    "observed": "#1B2A4A",  # actual outcome -- dark navy
    "fitted": "#2E86C1",  # model prediction -- blue
    "model_b": "#C0392B",  # a second model / flagged points -- red-orange
    "exposure": "#D5D8DC",  # exposure bars -- light neutral gray
    "reference": "#7F8C8D",  # diagonal / relativity=1.0 reference lines
}

#: (width, height) in inches at the default rcParams["figure.dpi"] below.
FIGSIZE = (7.5, 4.5)
FIGSIZE_WIDE = (9.5, 5.0)

_RC = {
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#4D4D4D",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#B0B0B0",
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#D0D0D0",
    "legend.fancybox": False,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def theme():
    """Context manager applying the shared house style. Use as
    ``with theme(): ...`` around any matplotlib drawing code.
    """
    return plt.rc_context(rc=_RC)


def new_axes(ax, figsize: tuple[float, float] = FIGSIZE):
    """Return ``(fig, ax)``: a fresh figure/axes if ``ax`` is ``None``,
    otherwise ``ax`` and its existing figure. Every ``insurabench.viz``
    plotting function accepts an optional ``ax=`` this way, so a caller
    building a multi-panel dashboard can hand in a subplot axes instead
    of always getting a new standalone figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def add_exposure_bars(ax, x, exposure, *, label: str = "Exposure"):
    """Draw ``exposure`` as bars on a secondary y-axis behind ``ax``'s own
    (foreground) content, and return that secondary axes.

    This is the standard actuarial "bars = volume, lines = rate" combo
    chart layout -- shared by ``one_way``/``lift``/``double_lift``/
    ``calibration`` plots so exposure always reads the same way across
    every chart. The z-order/patch-visibility dance here is what makes
    the bars sit strictly behind ``ax``'s lines/markers rather than
    painting over them.
    """
    ax2 = ax.twinx()
    ax2.bar(x, exposure, color=PALETTE["exposure"], width=0.6, zorder=1, label=label)
    ax2.set_ylabel("Exposure")
    ax2.grid(False)
    ax2.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    return ax2


def combined_legend(ax, ax2, *, loc: str = "upper left"):
    """Merge ``ax``'s and ``ax2``'s legend handles into one legend on
    ``ax`` -- needed whenever ``add_exposure_bars`` was used, since the
    bars' label otherwise lives on a separate (invisible-background)
    axes and would be silently dropped by a plain ``ax.legend()``.
    """
    lines, labels = ax.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax.legend(lines + bars, labels + bar_labels, loc=loc)


def finalize(fig, save_path: str | None):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
