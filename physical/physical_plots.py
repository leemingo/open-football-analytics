"""Plotting helpers for physical metrics on SkillCorner Open Data.

Colour policy. Bars and dots that name *players* are nominal: one hue for the
whole series, with length or position carrying the value, and selective direct
labels instead of a number on every mark. Colour is spent only where it encodes a
real second dimension (speed band, phase, estimator). The categorical slots below
are validated for the six data-viz checks -- lightness band, chroma floor,
protanopia/deuteranopia separation, and the normal-vision floor -- so slots must
be taken in order and never cycled.

Only the first three slots clear the checks for *all* pairs, which is what a
scatter needs (any two marks can sit side by side). Anything needing more
categories than that is faceted into small multiples rather than overlaid, so a
five-way position comparison never depends on telling yellow from orange.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical slots, in fixed order. Do not reorder: the order is the CVD-safety
# mechanism, not cosmetic.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3de"

# Speed bands read as an ordered progression, so they take the first three slots
# in band order rather than arbitrary hues.
BAND_COLORS = {"running": SERIES[0], "hsr": SERIES[1], "sprint": SERIES[2]}

__all__ = [
    "BAND_COLORS",
    "SERIES",
    "plot_band_composition",
    "plot_detection_share",
    "plot_estimator_sensitivity",
    "plot_leaderboard",
    "plot_percentile_profile",
    "plot_phase_share",
    "plot_speed_trace",
    "plot_tip_otip_facets",
    "plot_volume_vs_intensity",
    "save",
    "style_axes",
]


def style_axes(ax: plt.Axes, *, xlabel: str = "", ylabel: str = "", title: str = "",
               grid_axis: str = "x") -> plt.Axes:
    """Apply the recessive grid/axis treatment shared by every figure here."""

    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    return ax


def save(fig: plt.Figure, path: str | Path) -> Path:
    """Write a figure next to the repo's other tutorial outputs."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    return target


def plot_leaderboard(
    frame: pd.DataFrame,
    value_column: str,
    *,
    label_column: str = "player_name",
    top_n: int = 12,
    xlabel: str = "",
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal ranking bars, one hue, value labelled at each bar end.

    Player names are nominal, so colouring by rank would re-encode what bar
    length already shows.
    """

    top = frame.nlargest(top_n, value_column).iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 0.36 * len(top) + 1.2))
    positions = np.arange(len(top))
    ax.barh(positions, top[value_column], color=SERIES[0], height=0.62, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(top[label_column], fontsize=9, color=INK)
    span = float(top[value_column].max())
    for y, value in zip(positions, top[value_column]):
        ax.text(value + span * 0.012, y, f"{value:,.0f}" if span > 50 else f"{value:,.1f}",
                va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlim(0, span * 1.12)
    return style_axes(ax, xlabel=xlabel, title=title, grid_axis="x")


def plot_volume_vs_intensity(
    frame: pd.DataFrame,
    *,
    x: str = "total_distance_m_p60_bip",
    y: str = "high_intensity_distance_m_p60_bip",
    label_column: str = "player_name",
    n_labels: int = 6,
    xlabel: str = "Total distance (m per 60 BIP min)",
    ylabel: str = "High-intensity distance (m per 60 BIP min)",
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Volume against intensity, with only the extremes labelled.

    One hue: the question is where players sit relative to the two medians, not
    which group they belong to. Median guides split the plot into the four
    profiles (far and fast, far and steady, and so on).
    """

    data = frame[[x, y, label_column]].dropna()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.axvline(data[x].median(), color=GRID, linewidth=1.0, zorder=1)
    ax.axhline(data[y].median(), color=GRID, linewidth=1.0, zorder=1)
    ax.scatter(data[x], data[y], s=46, color=SERIES[0], alpha=0.85,
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
    # Label the marks furthest from the centre so the callouts describe the
    # profile extremes rather than crowding the middle.
    centre = np.hypot(
        (data[x] - data[x].median()) / max(data[x].std(), 1e-9),
        (data[y] - data[y].median()) / max(data[y].std(), 1e-9),
    )
    for _, row in data.loc[centre.nlargest(n_labels).index].iterrows():
        ax.annotate(row[label_column], (row[x], row[y]), textcoords="offset points",
                    xytext=(7, 3), fontsize=8.5, color=INK_SECONDARY)
    return style_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid_axis="both")


def plot_tip_otip_facets(
    frame: pd.DataFrame,
    *,
    x: str = "high_intensity_distance_tip_m_p30_tip",
    y: str = "high_intensity_distance_otip_m_p30_otip",
    facet_column: str = "position_group",
    order: tuple[str, ...] | None = None,
    title: str = "",
) -> plt.Figure:
    """One small multiple per position group: in-possession vs out-of-possession.

    Faceted rather than colour-coded because five categories cannot all be told
    apart under colour-vision deficiency in a single scatter. Each panel keeps the
    shared axes and a faint all-players backdrop so panels stay comparable, and
    the diagonal marks equal effort in both phases.
    """

    data = frame[[x, y, facet_column]].dropna()
    groups = list(order) if order else sorted(data[facet_column].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(2.5 * len(groups), 3.1),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    limit = float(max(data[x].max(), data[y].max())) * 1.08
    for ax, group in zip(axes, groups):
        subset = data.loc[data[facet_column] == group]
        ax.plot([0, limit], [0, limit], color=GRID, linewidth=1.0, zorder=1)
        ax.scatter(data[x], data[y], s=18, color=INK_MUTED, alpha=0.22,
                   edgecolor="none", zorder=2)
        ax.scatter(subset[x], subset[y], s=42, color=SERIES[0], alpha=0.9,
                   edgecolor=SURFACE, linewidth=1.1, zorder=3)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        style_axes(ax, title=f"{group}  (n={len(subset)})", grid_axis="both")
        ax.title.set_fontsize(9.5)
    axes[0].set_ylabel("Out of possession\n(m per 30 OTIP min)", color=INK_SECONDARY, fontsize=9)
    for ax in axes:
        ax.set_xlabel("In possession\n(m per 30 TIP min)", color=INK_SECONDARY, fontsize=9)
    if title:
        fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def plot_band_composition(
    frame: pd.DataFrame,
    *,
    group_column: str = "position_group",
    bands: tuple[str, ...] = ("running", "hsr", "sprint"),
    label_column: str | None = None,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Stacked speed-band distance per group, in band order.

    Walking/jogging is left out on purpose: it dominates total distance and
    flattens the differences the bands exist to show.
    """

    columns = {b: f"{b}_distance_m_p60_bip" for b in bands}
    grouped = (
        frame.groupby(group_column, observed=True)[list(columns.values())]
        .mean()
        .sort_values(columns[bands[-1]])
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 0.45 * len(grouped) + 1.4))
    positions = np.arange(len(grouped))
    left = np.zeros(len(grouped))
    for band in bands:
        values = grouped[columns[band]].to_numpy()
        # A 2px surface gap keeps adjacent segments from reading as one block.
        ax.barh(positions, values, left=left, height=0.62, color=BAND_COLORS[band],
                label=band.upper() if band != "running" else "Running 15-20",
                edgecolor=SURFACE, linewidth=1.5, zorder=2)
        left += values
    ax.set_yticks(positions)
    ax.set_yticklabels(grouped.index, fontsize=9, color=INK)
    legend = ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncols=len(bands))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return style_axes(ax, xlabel="Distance (m per 60 BIP min)", title=title, grid_axis="x")


def plot_phase_share(
    shares: pd.DataFrame,
    *,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Stacked phase composition, one row per team.

    ``shares`` is teams x phases and must already be normalised to fractions with
    phases beyond the fifth folded into "other" -- a ninth generated hue is never
    an option, so rare phases collapse rather than each taking a new colour.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 0.42 * len(shares) + 1.5))
    positions = np.arange(len(shares))
    left = np.zeros(len(shares))
    for index, column in enumerate(shares.columns):
        values = shares[column].to_numpy() * 100
        color = INK_MUTED if str(column).lower() == "other" else SERIES[index % len(SERIES)]
        ax.barh(positions, values, left=left, height=0.6, color=color, label=str(column),
                edgecolor=SURFACE, linewidth=1.5, zorder=2)
        left += values
    ax.set_yticks(positions)
    ax.set_yticklabels(shares.index, fontsize=9, color=INK)
    ax.set_xlim(0, 100)
    legend = ax.legend(frameon=False, fontsize=8.5, ncols=min(len(shares.columns), 6),
                       loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return style_axes(ax, xlabel="Share of off-ball runs (%)", title=title, grid_axis="x")


def plot_percentile_profile(
    percentiles: pd.Series,
    *,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """A player's percentile bar profile within their comparison group.

    Percentiles are a magnitude on one shared 0-100 scale, so one hue with a 50th
    reference line reads faster than a colour ramp.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 0.4 * len(percentiles) + 1.2))
    positions = np.arange(len(percentiles))[::-1]
    ax.barh(positions, percentiles.to_numpy(), color=SERIES[0], height=0.6, zorder=2)
    ax.axvline(50, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(percentiles.index, fontsize=9, color=INK)
    for y, value in zip(positions, percentiles.to_numpy()):
        ax.text(value + 1.5, y, f"{value:.0f}", va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlim(0, 108)
    return style_axes(ax, xlabel="Percentile within group", title=title, grid_axis="x")


def plot_estimator_sensitivity(
    frame: pd.DataFrame,
    *,
    metric_labels: dict[str, str],
    estimator_column: str = "estimator",
    title: str = "",
) -> plt.Figure:
    """One panel per metric, bars per estimator, each panel on its own scale.

    Separate panels rather than a second y-axis: distance and event counts do not
    share a scale, and a dual axis would invent a relationship between them.
    """

    metrics = list(metric_labels)
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.7 * len(metrics), 3.2))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        positions = np.arange(len(frame))
        ax.bar(positions, frame[metric], color=[SERIES[i % len(SERIES)] for i in positions],
               width=0.6, zorder=2)
        ax.set_xticks(positions)
        ax.set_xticklabels(frame[estimator_column], fontsize=8.5, color=INK, rotation=20, ha="right")
        for x, value in zip(positions, frame[metric]):
            ax.text(x, value, f"{value:,.1f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK_SECONDARY)
        top = float(frame[metric].max())
        ax.set_ylim(0, top * 1.18)
        style_axes(ax, title=metric_labels[metric], grid_axis="y")
        ax.title.set_fontsize(9.5)
    if title:
        fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def plot_speed_trace(
    kinematics: pd.DataFrame,
    *,
    thresholds: tuple[float, float, float] = (15.0, 20.0, 25.0),
    time_column: str = "timestamp",
    speed_column: str = "speed_mps",
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """One player's speed trace with the band thresholds drawn in.

    This is the figure that makes the sustain rule visible: a threshold crossing
    is not an effort until it stays above the line long enough.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 3.4))
    speed_kmh = kinematics[speed_column] * 3.6
    ax.plot(kinematics[time_column], speed_kmh, color=SERIES[0], linewidth=2.0, zorder=3)
    for threshold, label in zip(thresholds, ("Running 15", "HSR 20", "Sprint 25")):
        ax.axhline(threshold, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        ax.text(kinematics[time_column].max(), threshold, f" {label}", va="center",
                fontsize=8.5, color=INK_SECONDARY)
    return style_axes(ax, xlabel="Time in period (s)", ylabel="Speed (km/h)",
                      title=title, grid_axis="y")


def plot_detection_share(
    coverage: pd.DataFrame,
    *,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Per-match tracked-player counts in live vs dead frames, plus detected share.

    The point of the figure is that live frames carry the full XI on both sides
    while dead frames do not, so a low overall "coverage" number is dead-ball
    time rather than lost live play.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 3.6))
    positions = np.arange(len(coverage))
    width = 0.38
    ax.bar(positions - width / 2, coverage["players_per_live_frame"], width=width,
           color=SERIES[0], label="Live frames", zorder=2)
    ax.bar(positions + width / 2, coverage["players_per_dead_frame"], width=width,
           color=SERIES[1], label="Dead frames", zorder=2)
    ax.axhline(22, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax.text(positions[-1] + 0.5, 22, " 22 on pitch", va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xticks(positions)
    ax.set_xticklabels(coverage["match_id"], fontsize=8.5, color=INK, rotation=45, ha="right")
    legend = ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return style_axes(ax, ylabel="Tracked players per frame", title=title, grid_axis="y")
