"""Plotting helpers for center-origin xT experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle
import numpy as np
import pandas as pd

try:
    from football_cdf.constants import PITCH_X, PITCH_Y
except ModuleNotFoundError:
    PITCH_X = 105.0
    PITCH_Y = 68.0


CHANNEL_COLORS = {
    "central": "#5B14C8",
    "half-space": "#F28E2B",
    "wide": "#19A0AA",
    "unknown": "#777777",
}
ACTION_COLORS = {
    "pass": "#4E79A7",
    "carry": "#F28E2B",
    "shot": "#D9354A",
}


def add_route_columns(actions: pd.DataFrame) -> pd.DataFrame:
    """Add route labels used by plotting helpers."""
    frame = actions.copy()
    if "route_channel" in frame.columns:
        frame["route_channel"] = frame["route_channel"].fillna("unknown").astype(str)
        return frame

    y_cols = [col for col in ("start_y", "end_y") if col in frame.columns]
    if not y_cols:
        frame["route_channel"] = "unknown"
        return frame

    route_y = pd.concat([pd.to_numeric(frame[col], errors="coerce") for col in y_cols], axis=1).mean(axis=1)
    abs_route_y = route_y.abs()
    frame["route_channel"] = np.select(
        [abs_route_y.le(PITCH_Y / 6), abs_route_y.le(PITCH_Y / 3)],
        ["central", "half-space"],
        default="wide",
    )
    frame.loc[route_y.isna(), "route_channel"] = "unknown"
    return frame


def draw_center_origin_pitch(ax: plt.Axes, *, line_color: str = "#8a8a8a", linewidth: float = 1.0) -> None:
    """Draw a simple 105x68 center-origin football pitch."""
    half_x = PITCH_X / 2
    half_y = PITCH_Y / 2
    ax.add_patch(Rectangle((-half_x, -half_y), PITCH_X, PITCH_Y, fill=False, edgecolor=line_color, linewidth=linewidth))
    ax.axvline(0, color=line_color, linewidth=linewidth, alpha=0.9)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, edgecolor=line_color, linewidth=linewidth))
    ax.scatter([0], [0], s=8, color=line_color, zorder=3)

    penalty_w = 40.32
    six_w = 18.32
    for side in [-1, 1]:
        goal_x = side * half_x
        box_x = goal_x - side * 16.5
        six_x = goal_x - side * 5.5
        ax.add_patch(Rectangle((min(goal_x, box_x), -penalty_w / 2), 16.5, penalty_w, fill=False, edgecolor=line_color, linewidth=linewidth))
        ax.add_patch(Rectangle((min(goal_x, six_x), -six_w / 2), 5.5, six_w, fill=False, edgecolor=line_color, linewidth=linewidth))
        ax.scatter([goal_x - side * 11.0], [0], s=8, color=line_color, zorder=3)
        ax.add_patch(Rectangle((goal_x, -7.32 / 2), side * 1.5, 7.32, fill=False, edgecolor=line_color, linewidth=linewidth))
        arc_center = (goal_x - side * 11.0, 0)
        theta1, theta2 = (310, 50) if side < 0 else (130, 230)
        ax.add_patch(Arc(arc_center, 18.3, 18.3, theta1=theta1, theta2=theta2, edgecolor=line_color, linewidth=linewidth))

    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_aspect("equal", adjustable="box")


def plot_xthreat_surface(surface: np.ndarray, *, title: str = "xT surface", ax: plt.Axes | None = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    im = ax.imshow(
        surface,
        extent=[-PITCH_X / 2, PITCH_X / 2, -PITCH_Y / 2, PITCH_Y / 2],
        origin="lower",
        cmap="magma",
        aspect="auto",
    )
    draw_center_origin_pitch(ax, line_color="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("x, attacking left to right")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    return fig


def plot_precomputed_xthreat_surface(
    precomputed_model_or_surface,
    *,
    title: str = "Precomputed xT surface | center-origin display",
    row_zero_top: bool | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a Karun/socceraction precomputed xT map in center-origin display coordinates."""
    surface = getattr(precomputed_model_or_surface, "xT", precomputed_model_or_surface)
    surface = np.asarray(surface, dtype="float64")
    if row_zero_top is None:
        row_zero_top = bool(getattr(precomputed_model_or_surface, "row_zero_top", True))
    display_surface = np.flipud(surface) if row_zero_top else surface
    return plot_xthreat_surface(display_surface, title=title, ax=ax)


def plot_skillcorner_comparison(
    scored: pd.DataFrame,
    *,
    source_col: str = "custom_xT_end",
    sample: int = 20000,
    random_state: int = 7,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    frame = scored[
        scored["action_type"].eq("pass")
        & scored["skillcorner_target_xthreat"].notna()
        & scored[source_col].notna()
    ].copy()
    if sample and len(frame) > sample:
        frame = frame.sample(sample, random_state=random_state)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
    else:
        fig = ax.figure
    ax.scatter(frame["skillcorner_target_xthreat"], frame[source_col], s=5, alpha=0.15)
    high = float(np.nanmax([frame["skillcorner_target_xthreat"].max(), frame[source_col].max()]))
    high = max(high, 0.01)
    ax.plot([0, high], [0, high], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("SkillCorner target xT")
    ax.set_ylabel(source_col)
    ax.set_title(f"{source_col} vs SkillCorner target xT")
    ax.grid(alpha=0.25)
    return fig


def plot_team_xthreat_bar(
    team_summary: pd.DataFrame,
    *,
    value_col: str = "custom_xT_added_per_100",
    season_name: str | None = "2025",
    top_n: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    frame = team_summary.copy()
    if season_name is not None and "season_name" in frame.columns:
        frame = frame[frame["season_name"].astype(str).eq(str(season_name))].copy()
    frame = frame.sort_values(value_col, ascending=True)
    if top_n is not None:
        frame = frame.tail(int(top_n))
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure
    colors = np.where(frame[value_col].ge(0), "#5B14C8", "#D9354A")
    ax.barh(frame["team_shortname"], frame[value_col], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(value_col)
    ax.set_title(f"Team xT | {season_name or 'all seasons'}")
    ax.grid(axis="x", alpha=0.25)
    return fig


def save_basic_xthreat_figures(
    scored: pd.DataFrame,
    team_summary: pd.DataFrame,
    surface: np.ndarray,
    *,
    out_dir: str | Path,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = {
        "custom_xthreat_surface": plot_xthreat_surface(surface, title="Custom learned xT surface"),
        "custom_vs_skillcorner_xthreat": plot_skillcorner_comparison(scored, source_col="custom_xT_end"),
        "team_custom_xthreat": plot_team_xthreat_bar(team_summary),
    }
    if "precomputed_xT_end" in scored.columns:
        figs["precomputed_vs_skillcorner_xthreat"] = plot_skillcorner_comparison(scored, source_col="precomputed_xT_end")
    paths: dict[str, Path] = {}
    for name, fig in figs.items():
        path = out / f"{name}.png"
        fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        paths[name] = path
    return paths


def plot_action_xthreat_heatmap(
    scored: pd.DataFrame,
    *,
    season_name: str | None = "2025",
    team_shortname: str | None = None,
    action_type: str | None = None,
    value_col: str = "custom_xT_added",
    bins: tuple[int, int] = (16, 12),
    positive_only: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot total xT-added by action destination cell."""
    frame = scored.copy()
    if season_name is not None and "season_name" in frame.columns:
        frame = frame[frame["season_name"].astype(str).eq(str(season_name))].copy()
    if team_shortname is not None:
        frame = frame[frame["team_shortname"].astype(str).eq(str(team_shortname))].copy()
    if action_type is not None:
        frame = frame[frame["action_type"].astype(str).eq(str(action_type))].copy()
    frame = frame.dropna(subset=["end_x", "end_y", value_col]).copy()
    weights = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    if positive_only:
        weights = weights.clip(lower=0.0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    title_bits = ["positive xT heatmap" if positive_only else "net xT heatmap"]
    if team_shortname:
        title_bits.append(str(team_shortname))
    if action_type:
        title_bits.append(str(action_type))
    if season_name:
        title_bits.append(str(season_name))
    title = " | ".join(title_bits)
    if frame.empty or float(weights.abs().sum()) == 0.0:
        draw_center_origin_pitch(ax, line_color="#9a9a9a", linewidth=1.0)
        ax.text(0.5, 0.5, "No xT data after filters", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("x, attacking left to right")
        ax.set_ylabel("y")
        return fig
    heat, xedges, yedges = np.histogram2d(
        frame["end_x"],
        frame["end_y"],
        bins=bins,
        range=[[-PITCH_X / 2, PITCH_X / 2], [-PITCH_Y / 2, PITCH_Y / 2]],
        weights=weights,
    )
    im = ax.imshow(
        heat.T,
        extent=[-PITCH_X / 2, PITCH_X / 2, -PITCH_Y / 2, PITCH_Y / 2],
        origin="lower",
        cmap="magma",
        aspect="auto",
    )
    draw_center_origin_pitch(ax, line_color="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("x, attacking left to right")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    return fig


def plot_route_xthreat_bars(
    route_summary: pd.DataFrame,
    *,
    team_shortname: str | None = None,
    season_name: str | None = "2025",
    value_col: str = "xT",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    frame = route_summary.copy()
    if season_name is not None and "season_name" in frame.columns:
        frame = frame[frame["season_name"].astype(str).eq(str(season_name))].copy()
    if team_shortname is not None and "team_shortname" in frame.columns:
        frame = frame[frame["team_shortname"].astype(str).eq(str(team_shortname))].copy()
    if value_col not in frame.columns:
        frame[value_col] = np.nan
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame[frame["action_type"].isin(["pass", "carry"])].copy()
    pivot = frame.pivot_table(index="route_channel", columns="action_type", values=value_col, aggfunc="sum", fill_value=0.0)
    order = ["central", "half-space", "wide"]
    pivot = pivot.reindex([c for c in order if c in pivot.index])
    pivot = pivot.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    title = "Pass xT vs carry xT by channel"
    if team_shortname:
        title += f" | {team_shortname}"
    if season_name:
        title += f" | {season_name}"
    if pivot.empty or pivot.shape[1] == 0:
        ax.text(0.5, 0.5, "No route xT data after filters", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Route channel")
        ax.set_ylabel(value_col)
        ax.grid(axis="y", alpha=0.25)
        return fig
    color_order = [ACTION_COLORS.get(c, "#777777") for c in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=color_order)
    ax.set_xlabel("Route channel")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_team_route_map(
    scored: pd.DataFrame,
    *,
    season_name: str | None = "2025",
    team_shortname: str | None = None,
    action_type: str | None = None,
    value_col: str = "custom_xT_added",
    top_n: int = 250,
    min_value: float = 0.0,
    color_by: str = "route_channel",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot the highest-xT pass/carry routes as arrows on a center-origin pitch."""
    frame = add_route_columns(scored)
    if season_name is not None and "season_name" in frame.columns:
        frame = frame[frame["season_name"].astype(str).eq(str(season_name))].copy()
    if team_shortname is not None:
        frame = frame[frame["team_shortname"].astype(str).eq(str(team_shortname))].copy()
    if action_type is not None:
        frame = frame[frame["action_type"].astype(str).eq(str(action_type))].copy()
    else:
        frame = frame[frame["action_type"].isin(["pass", "carry"])].copy()
    frame = frame.dropna(subset=["start_x", "start_y", "end_x", "end_y", value_col]).copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    frame = frame[frame[value_col].ge(float(min_value))].copy()
    if top_n and len(frame) > top_n:
        frame = frame.sort_values(value_col, ascending=False).head(int(top_n)).copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    draw_center_origin_pitch(ax, line_color="#9a9a9a", linewidth=1.0)
    if frame.empty:
        ax.set_title("No routes after filters")
        return fig

    palette = CHANNEL_COLORS if color_by == "route_channel" else ACTION_COLORS
    color_col = color_by if color_by in frame.columns else "route_channel"
    labels = frame[color_col].astype(str)
    colors = labels.map(palette).fillna("#777777").to_numpy()
    ax.quiver(
        frame["start_x"],
        frame["start_y"],
        frame["end_x"] - frame["start_x"],
        frame["end_y"] - frame["start_y"],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=colors,
        width=0.0028,
        alpha=0.72,
    )
    for label in sorted(labels.dropna().unique()):
        ax.plot([], [], color=palette.get(label, "#777777"), label=label, linewidth=3)
    title_bits = ["High xT routes"]
    if team_shortname:
        title_bits.append(str(team_shortname))
    if action_type:
        title_bits.append(str(action_type))
    if season_name:
        title_bits.append(str(season_name))
    ax.set_title(" | ".join(title_bits))
    ax.set_xlabel("x, attacking left to right")
    ax.set_ylabel("y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    return fig


def plot_team_route_change(
    change: pd.DataFrame,
    *,
    team_shortname: str,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    row = change[change["team_shortname"].astype(str).eq(str(team_shortname))]
    if row.empty:
        raise ValueError(f"No route change row for {team_shortname!r}")
    row = row.iloc[0]
    cols = ["central_xT_share_delta", "half_space_xT_share_delta", "wide_xT_share_delta", "pass_xT_share_delta", "carry_xT_share_delta"]
    labels = ["Central", "Half-space", "Wide", "Pass", "Carry"]
    values = [float(row[c]) if c in row and pd.notna(row[c]) else 0.0 for c in cols]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    colors = np.where(np.asarray(values) >= 0, "#5B14C8", "#D9354A")
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Share delta, 2025 - 2024")
    ax.set_title(f"Attack route mix change | {team_shortname}")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_xthreat_components(model, *, min_shots_for_goal_prob: int = 5, axarr=None) -> plt.Figure:
    """Plot the core Karun xT ingredients from a fitted model.

    `P(goal | shot, zone)` is an empirical goals/shots ratio. Cells with very
    few shots can otherwise show values like 1.0 after a single goal, so the
    display masks cells with fewer than `min_shots_for_goal_prob` shots.
    """
    shot = getattr(model, "shot_prob_matrix", None)
    move = getattr(model, "move_prob_matrix", None)
    scoring = getattr(model, "scoring_prob_matrix", None)
    shot_counts = getattr(model, "shot_count_matrix", None)

    scoring_display = None
    scoring_title = "P(goal | shot, zone)"
    if scoring is not None:
        scoring_display = np.asarray(scoring, dtype="float64").copy()
        if shot_counts is not None and min_shots_for_goal_prob > 1:
            scoring_display = scoring_display.copy()
            scoring_display[np.asarray(shot_counts) < int(min_shots_for_goal_prob)] = np.nan
            scoring_title += f" | shots >= {int(min_shots_for_goal_prob)}"

    components = [
        (shot, "P(shot | zone)"),
        (move, "P(move | zone)"),
        (scoring_display, scoring_title),
    ]
    if shot is not None and scoring is not None:
        payoff = np.asarray(shot) * np.asarray(scoring)
        if shot_counts is not None and min_shots_for_goal_prob > 1:
            payoff = payoff.copy()
            payoff[np.asarray(shot_counts) < int(min_shots_for_goal_prob)] = np.nan
        components.append((payoff, "Immediate shot payoff"))
    else:
        components.append((None, "Immediate shot payoff"))

    if axarr is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes = axes.ravel()
    else:
        axes = np.asarray(axarr).ravel()
        fig = axes[0].figure
    for ax, (surface, title) in zip(axes, components):
        if surface is None:
            ax.axis("off")
            ax.set_title(f"{title} unavailable")
            continue
        surface_arr = np.asarray(surface, dtype="float64")
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(color="#202020")
        im = ax.imshow(
            surface_arr,
            extent=[-PITCH_X / 2, PITCH_X / 2, -PITCH_Y / 2, PITCH_Y / 2],
            origin="lower",
            cmap=cmap,
            aspect="auto",
        )
        draw_center_origin_pitch(ax, line_color="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    return fig

def plot_xthreat_iterations(
    heatmaps,
    *,
    iterations: list[int] | None = None,
    axarr=None,
) -> plt.Figure:
    """Plot selected xT iteration surfaces from model.heatmaps."""
    heatmaps = list(heatmaps)
    if not heatmaps:
        raise ValueError("heatmaps is empty")
    if iterations is None:
        iterations = [0, 1, 2, 3, 5, len(heatmaps) - 1]
    resolved = []
    for idx in iterations:
        j = len(heatmaps) + idx if idx < 0 else idx
        j = min(max(j, 0), len(heatmaps) - 1)
        if j not in resolved:
            resolved.append(j)
    n = len(resolved)
    if axarr is None:
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows))
        axes = np.asarray(axes).ravel()
    else:
        axes = np.asarray(axarr).ravel()
        fig = axes[0].figure
    vmax = max(float(np.nanmax(np.asarray(heatmaps[j]))) for j in resolved)
    for ax, j in zip(axes, resolved):
        im = ax.imshow(
            np.asarray(heatmaps[j], dtype="float64"),
            extent=[-PITCH_X / 2, PITCH_X / 2, -PITCH_Y / 2, PITCH_Y / 2],
            origin="lower",
            cmap="magma",
            aspect="auto",
            vmin=0,
            vmax=vmax if vmax > 0 else None,
        )
        draw_center_origin_pitch(ax, line_color="white", linewidth=0.8)
        label = "final" if j == len(heatmaps) - 1 else str(j)
        ax.set_title(f"xT iteration {label}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_action_xthreat_start_heatmap(
    scored: pd.DataFrame,
    *,
    season_name: str | None = "2025",
    team_shortname: str | None = None,
    action_type: str | None = None,
    value_col: str = "custom_xT_added",
    bins: tuple[int, int] = (16, 12),
    positive_only: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot total xT-added by action start cell."""
    frame = scored.copy()
    if season_name is not None and "season_name" in frame.columns:
        frame = frame[frame["season_name"].astype(str).eq(str(season_name))].copy()
    if team_shortname is not None:
        frame = frame[frame["team_shortname"].astype(str).eq(str(team_shortname))].copy()
    if action_type is not None:
        frame = frame[frame["action_type"].astype(str).eq(str(action_type))].copy()
    frame = frame.dropna(subset=["start_x", "start_y", value_col]).copy()
    weights = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    if positive_only:
        weights = weights.clip(lower=0.0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    title_bits = ["positive xT start heatmap" if positive_only else "net xT start heatmap"]
    if team_shortname:
        title_bits.append(str(team_shortname))
    if action_type:
        title_bits.append(str(action_type))
    if season_name:
        title_bits.append(str(season_name))
    title = " | ".join(title_bits)
    if frame.empty or float(weights.abs().sum()) == 0.0:
        draw_center_origin_pitch(ax, line_color="#9a9a9a", linewidth=1.0)
        ax.text(0.5, 0.5, "No xT data after filters", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("x, attacking left to right")
        ax.set_ylabel("y")
        return fig
    heat, _, _ = np.histogram2d(
        frame["start_x"],
        frame["start_y"],
        bins=bins,
        range=[[-PITCH_X / 2, PITCH_X / 2], [-PITCH_Y / 2, PITCH_Y / 2]],
        weights=weights,
    )
    im = ax.imshow(
        heat.T,
        extent=[-PITCH_X / 2, PITCH_X / 2, -PITCH_Y / 2, PITCH_Y / 2],
        origin="lower",
        cmap="magma",
        aspect="auto",
    )
    draw_center_origin_pitch(ax, line_color="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("x, attacking left to right")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    return fig


def plot_sequence_xthreat_credit(
    sequence_actions: pd.DataFrame,
    *,
    value_col: str = "custom_xT_added",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Draw one possession sequence with action arrows colored by xT-added."""
    frame = sequence_actions.copy()
    frame = frame[frame["action_type"].isin(["pass", "carry"])].dropna(subset=["start_x", "start_y", "end_x", "end_y"]).copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    draw_center_origin_pitch(ax, line_color="#9a9a9a", linewidth=1.0)
    if frame.empty:
        ax.set_title("No move actions in selected sequence")
        return fig
    values = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    colors = np.where(values >= 0, "#5B14C8", "#D9354A")
    ax.quiver(
        frame["start_x"], frame["start_y"],
        frame["end_x"] - frame["start_x"], frame["end_y"] - frame["start_y"],
        angles="xy", scale_units="xy", scale=1,
        color=colors, width=0.003, alpha=0.78,
    )
    for _, row in frame.iterrows():
        label = int(row.get("sequence_action_number", 0))
        ax.annotate(str(label), (row["start_x"], row["start_y"]), fontsize=8, color="black")
    title_bits = ["Sequence xT credit"]
    for col in ["team_shortname", "match_id", "phase_index"]:
        if col in sequence_actions.columns and sequence_actions[col].notna().any():
            title_bits.append(f"{col}={sequence_actions[col].dropna().astype(str).iloc[0]}")
    ax.set_title(" | ".join(title_bits))
    ax.set_xlabel("x, attacking left to right")
    ax.set_ylabel("y")
    ax.plot([], [], color="#5B14C8", label="xT gain", linewidth=3)
    ax.plot([], [], color="#D9354A", label="xT loss", linewidth=3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    return fig


def plot_sequence_xthreat_waterfall(
    sequence_actions: pd.DataFrame,
    *,
    value_col: str = "custom_xT_added",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot xT-added per action in one possession sequence."""
    frame = sequence_actions.copy()
    if "sequence_action_number" not in frame.columns:
        frame["sequence_action_number"] = np.arange(1, len(frame) + 1)
    values = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    labels = frame["sequence_action_number"].astype(str) + " " + frame["player_name"].fillna("unknown").astype(str)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(4, len(frame) * 0.32)))
    else:
        fig = ax.figure
    colors = np.where(values >= 0, "#5B14C8", "#D9354A")
    y = np.arange(len(frame))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel(value_col)
    ax.set_title("Action-by-action xT credit")
    ax.grid(axis="x", alpha=0.25)
    return fig


def plot_team_xthreat_surface_comparison(
    league_surface: np.ndarray,
    team_surface: np.ndarray,
    *,
    team_name: str,
    axarr=None,
) -> plt.Figure:
    """Compare team-specific xT surface with the league-wide surface."""
    if axarr is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    else:
        axes = np.asarray(axarr).ravel()
        fig = axes[0].figure
    surfaces = [
        (league_surface, "League xT"),
        (team_surface, f"{team_name} xT"),
        (np.asarray(team_surface) - np.asarray(league_surface), f"{team_name} - league"),
    ]
    vmax = max(float(np.nanmax(np.asarray(league_surface))), float(np.nanmax(np.asarray(team_surface))))
    diff_abs = float(np.nanmax(np.abs(surfaces[2][0])))
    for ax, (surface, title) in zip(axes, surfaces):
        kwargs = {"cmap": "magma", "vmin": 0, "vmax": vmax} if " - league" not in title else {"cmap": "coolwarm", "vmin": -diff_abs, "vmax": diff_abs}
        im = ax.imshow(
            np.asarray(surface, dtype="float64"),
            extent=[-PITCH_X / 2, PITCH_X / 2, -PITCH_Y / 2, PITCH_Y / 2],
            origin="lower",
            aspect="auto",
            **kwargs,
        )
        draw_center_origin_pitch(ax, line_color="white" if " - league" not in title else "#444444", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    return fig


def plot_sequence_xthreat_trajectory(
    sequence_actions: pd.DataFrame,
    *,
    value_col: str = "custom_xT_added",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot cumulative xT over the course of one possession sequence."""
    frame = sequence_actions.copy()
    if "sequence_action_number" not in frame.columns:
        frame["sequence_action_number"] = np.arange(1, len(frame) + 1)
    values = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    cumulative = values.cumsum()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
    else:
        fig = ax.figure
    ax.plot(frame["sequence_action_number"], cumulative, marker="o", color="#5B14C8", linewidth=2)
    ax.bar(frame["sequence_action_number"], values, color=np.where(values >= 0, "#5B14C8", "#D9354A"), alpha=0.25)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Action number")
    ax.set_ylabel("Cumulative xT")
    title_bits = ["Possession xT trajectory"]
    for col in ["team_shortname", "match_id", "phase_index"]:
        if col in frame.columns and frame[col].notna().any():
            title_bits.append(f"{col}={frame[col].dropna().astype(str).iloc[0]}")
    ax.set_title(" | ".join(title_bits))
    ax.grid(alpha=0.25)
    return fig


def plot_top_sequence_trajectories(
    scored: pd.DataFrame,
    sequence_summary: pd.DataFrame,
    *,
    top_n: int = 5,
    value_col: str = "custom_xT_added",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot cumulative xT trajectories for top possession sequences."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5))
    else:
        fig = ax.figure
    ordered = sequence_summary.sort_values("xT", ascending=False).head(int(top_n)).copy()
    for _, row in ordered.iterrows():
        frame = scored[
            scored["match_id"].astype(str).eq(str(row["match_id"]))
            & scored["phase_index"].astype(str).eq(str(row["phase_index"]))
            & scored["team_shortname"].astype(str).eq(str(row["team_shortname"]))
        ].copy()
        sort_cols = [c for c in ["period", "frame_event", "action_idx", "event_id"] if c in frame.columns]
        if sort_cols:
            frame = frame.sort_values(sort_cols)
        values = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
        cumulative = values.cumsum()
        x = np.arange(1, len(frame) + 1)
        label = f"{row['team_shortname']} | {row['match_id']} | phase {row['phase_index']} | xT {row['xT']:.3f}"
        ax.plot(x, cumulative, marker="o", linewidth=1.8, alpha=0.8, label=label)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Action number")
    ax.set_ylabel("Cumulative xT")
    ax.set_title(f"Top {top_n} possession xT trajectories")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    return fig


def plot_shoot_vs_move_decision_grid(
    decision_grid: pd.DataFrame,
    *,
    axarr=None,
) -> plt.Figure:
    """Visualize shot value, move value, and their difference by zone."""
    if decision_grid.empty:
        raise ValueError("decision_grid is empty")
    l = int(decision_grid["x_cell"].max() + 1)
    w = int(decision_grid["y_cell"].max() + 1)

    def mat(col: str) -> np.ndarray:
        arr = np.full((w, l), np.nan, dtype="float64")
        for _, row in decision_grid.iterrows():
            arr[int(row["y_cell"]), int(row["x_cell"])] = row[col]
        return arr

    surfaces = [
        (mat("shot_value_masked"), "Shot value | P(goal | shot)"),
        (mat("move_attempt_value"), "Move attempt continuation value"),
        (mat("shoot_minus_move"), "Shot value - move value"),
    ]
    if axarr is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    else:
        axes = np.asarray(axarr).ravel()
        fig = axes[0].figure
    for ax, (surface, title) in zip(axes, surfaces):
        cmap_name = "coolwarm" if " - " in title else "magma"
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="#202020")
        kwargs = {}
        if " - " in title:
            m = float(np.nanmax(np.abs(surface))) if np.isfinite(surface).any() else 0.01
            kwargs = {"vmin": -m, "vmax": m}
        im = ax.imshow(
            surface,
            extent=[-PITCH_X / 2, PITCH_X / 2, -PITCH_Y / 2, PITCH_Y / 2],
            origin="lower",
            cmap=cmap,
            aspect="auto",
            **kwargs,
        )
        draw_center_origin_pitch(ax, line_color="white" if cmap_name == "magma" else "#444444", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    return fig


def plot_player_decision_vs_team_profile(
    player_profile: pd.DataFrame,
    *,
    season_name: str | None = "2025",
    team_shortname: str | None = None,
    value_col: str = "xT_above_team_profile_per_100",
    top_n: int = 20,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar chart for player action value above/below team zone-action baseline."""
    frame = player_profile.copy()
    if season_name is not None and "season_name" in frame.columns:
        frame = frame[frame["season_name"].astype(str).eq(str(season_name))].copy()
    if team_shortname is not None:
        frame = frame[frame["team_shortname"].astype(str).eq(str(team_shortname))].copy()
    frame = frame.sort_values(value_col, ascending=True).tail(int(top_n)).copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(5, len(frame) * 0.32)))
    else:
        fig = ax.figure
    labels = frame["player_name"].fillna("unknown").astype(str) + " | " + frame["team_shortname"].fillna("unknown").astype(str)
    colors = np.where(frame[value_col] >= 0, "#5B14C8", "#D9354A")
    ax.barh(labels, frame[value_col], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(value_col)
    title = "Player value above team zone-action profile"
    if team_shortname:
        title += f" | {team_shortname}"
    if season_name:
        title += f" | {season_name}"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    return fig
