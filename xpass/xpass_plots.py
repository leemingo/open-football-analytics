"""Plotting helpers for SkillCorner xPass / PAx reports."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from football_cdf.constants import PITCH_X, PITCH_Y
from xpass.skillcorner_passes import DEFAULT_OUTPUT_DIR
from xpass.pass_clusters import PASS_CLUSTER_LABELS_EN, add_pass_cluster_features


DEFAULT_PLAYER_PAX_PATH = DEFAULT_OUTPUT_DIR / "player_pax.csv"
DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_DIR / "figures"

PURPLE = "#5B14C8"
RED = "#D9354A"
DARK = "#12081F"
GRID = "#B8B8B8"
POINT_RED = "#E7343F"

SOURCE_LABELS = {
    "skillcorner": "xPass",
    "custom": "Custom xPass",
}


def _source_label(source: str) -> str:
    return SOURCE_LABELS.get(str(source).lower(), str(source))


def _source_columns(source: str) -> tuple[str, str]:
    """Return average-xPass and PAx-per-100 column names for one source."""
    source_key = str(source).lower().strip()
    if source_key not in {"skillcorner", "custom"}:
        raise ValueError("source must be either 'skillcorner' or 'custom'")
    return f"avg_{source_key}_xpass", f"{source_key}_pax_per_100"


def _pass_xpass_column(source: str, *, custom_col: str = "model_xpass") -> str:
    """Return the per-pass xPass column for one source."""
    source_key = str(source).lower().strip()
    if source_key == "skillcorner":
        return "skillcorner_xpass"
    if source_key == "custom":
        return custom_col
    raise ValueError("source must be either 'skillcorner' or 'custom'")


def _default_subtitle(season_name: str | None, source: str) -> str:
    season = "All seasons" if season_name is None else str(season_name)
    return f"SkillCorner Open Data | {season} | {_source_label(source)}"


def _draw_pitch(ax: plt.Axes) -> None:
    """Draw a simple 105x68 football pitch."""
    from matplotlib.patches import Arc, Circle, Rectangle

    line = "#888888"
    lw = 1.0
    ax.add_patch(Rectangle((0, 0), PITCH_X, PITCH_Y, fill=False, ec=line, lw=lw))
    ax.plot([PITCH_X / 2, PITCH_X / 2], [0, PITCH_Y], color=line, lw=lw)
    ax.add_patch(Circle((PITCH_X / 2, PITCH_Y / 2), 9.15, fill=False, ec=line, lw=lw))
    ax.add_patch(Circle((PITCH_X / 2, PITCH_Y / 2), 0.25, color=line, lw=0))

    penalty_w = 40.32
    six_w = 18.32
    for x0, sign in [(0, 1), (PITCH_X, -1)]:
        ax.add_patch(
            Rectangle(
                (x0 if sign > 0 else x0 - 16.5, PITCH_Y / 2 - penalty_w / 2),
                16.5,
                penalty_w,
                fill=False,
                ec=line,
                lw=lw,
            )
        )
        ax.add_patch(
            Rectangle(
                (x0 if sign > 0 else x0 - 5.5, PITCH_Y / 2 - six_w / 2),
                5.5,
                six_w,
                fill=False,
                ec=line,
                lw=lw,
            )
        )
        ax.add_patch(
            Rectangle(
                (x0 - 2.0 if sign > 0 else x0, PITCH_Y / 2 - 3.66),
                2.0,
                7.32,
                fill=False,
                ec=line,
                lw=lw,
            )
        )
        ax.add_patch(Circle((x0 + sign * 11.0, PITCH_Y / 2), 0.25, color=line, lw=0))
        theta1, theta2 = (310, 50) if sign > 0 else (130, 230)
        ax.add_patch(
            Arc(
                (x0 + sign * 11.0, PITCH_Y / 2),
                18.3,
                18.3,
                theta1=theta1,
                theta2=theta2,
                ec=line,
                lw=lw,
            )
        )

    ax.set_xlim(-3, PITCH_X + 3)
    ax.set_ylim(-9, PITCH_Y + 5)
    ax.set_aspect("equal")
    ax.axis("off")


def _filter_player_frame(
    player_pax: pd.DataFrame,
    *,
    min_passes: int = 300,
    season_name: str | None = None,
    exclude_goalkeepers: bool = False,
) -> pd.DataFrame:
    frame = player_pax.copy()
    if season_name is not None and "season_name" in frame.columns:
        frame = frame[frame["season_name"].astype(str).eq(str(season_name))].copy()
    if min_passes:
        frame = frame[frame["passes"].ge(int(min_passes))].copy()
    if exclude_goalkeepers and "player_position" in frame.columns:
        frame = frame[~frame["player_position"].astype("string").eq("GK")].copy()
    return frame.reset_index(drop=True)


def _player_label(row: pd.Series) -> str:
    player = row.get("player_name", "Unknown")
    team = row.get("team_shortname", "")
    if pd.isna(team) or not str(team):
        return str(player)
    return f"{player} ({team})"


def _top_bottom(
    frame: pd.DataFrame,
    *,
    value_col: str,
    n: int,
) -> pd.DataFrame:
    if value_col not in frame.columns:
        raise ValueError(f"Missing required value column: {value_col}")
    top = frame.nlargest(n, value_col).copy()
    bottom = frame.nsmallest(n, value_col).sort_values(value_col, ascending=False).copy()
    top["plot_group"] = "top"
    bottom["plot_group"] = "bottom"
    return pd.concat([top, bottom], ignore_index=True)


def _style_bar_axis(ax: plt.Axes, *, xlabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=14, color=DARK, labelpad=12)
    ax.tick_params(axis="both", labelsize=12, colors=DARK)
    ax.grid(axis="x", color=GRID, linewidth=1.1, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)


def plot_xp_top_bottom(
    player_pax: pd.DataFrame,
    *,
    n: int = 5,
    min_passes: int = 300,
    season_name: str | None = "2025",
    exclude_goalkeepers: bool = False,
    source: str = "skillcorner",
    value_col: str | None = None,
    title: str = "xP | Top 5 vs. Bottom 5",
    subtitle: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot top/bottom players by average expected completion."""
    source_avg_col, _ = _source_columns(source)
    value_col = value_col or source_avg_col
    if subtitle is None:
        subtitle = _default_subtitle(season_name, source)
    frame = _filter_player_frame(
        player_pax,
        min_passes=min_passes,
        season_name=season_name,
        exclude_goalkeepers=exclude_goalkeepers,
    )
    rows = _top_bottom(frame, value_col=value_col, n=n)
    labels = rows.apply(_player_label, axis=1)
    values = rows[value_col].astype(float)
    colors = np.where(rows["plot_group"].eq("top"), PURPLE, RED)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 7.0))
    else:
        fig = ax.figure

    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13, color=DARK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.linspace(0, 1.0, 5))
    _style_bar_axis(ax, xlabel="Average Expected Pass Completion (xP)")
    for yi, value in zip(y, values):
        ax.text(
            min(float(value) + 0.015, 0.995),
            yi,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=13,
            fontweight="bold",
            color=DARK,
        )

    fig.text(0.28, 0.94, title, fontsize=24, fontweight="bold", color=DARK)
    fig.text(0.28, 0.89, subtitle, fontsize=16, fontweight="bold", color=DARK)
    fig.subplots_adjust(left=0.28, right=0.96, top=0.84, bottom=0.13)
    return fig


def plot_pax_top_bottom(
    player_pax: pd.DataFrame,
    *,
    n: int = 5,
    min_passes: int = 300,
    season_name: str | None = "2025",
    exclude_goalkeepers: bool = False,
    source: str = "skillcorner",
    value_col: str | None = None,
    title: str = "PAx | Top 5 vs. Bottom 5",
    subtitle: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot top/bottom players by PAx per 100 passes."""
    _, source_pax_col = _source_columns(source)
    value_col = value_col or source_pax_col
    if subtitle is None:
        subtitle = _default_subtitle(season_name, source)
    frame = _filter_player_frame(
        player_pax,
        min_passes=min_passes,
        season_name=season_name,
        exclude_goalkeepers=exclude_goalkeepers,
    )
    rows = _top_bottom(frame, value_col=value_col, n=n)
    labels = rows.apply(_player_label, axis=1)
    values = rows[value_col].astype(float)
    colors = np.where(rows["plot_group"].eq("top"), PURPLE, RED)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 7.0))
    else:
        fig = ax.figure

    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13, color=DARK)
    ax.invert_yaxis()
    lim = max(abs(values.min()), abs(values.max()), 1.0)
    pad = max(0.5, lim * 0.18)
    ax.set_xlim(-lim - pad, lim + pad)
    ax.axvline(0, color=DARK, linewidth=1.3)
    _style_bar_axis(ax, xlabel="Passes Above Expected (PAx) per 100 passes")

    offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
    for yi, value in zip(y, values):
        ha = "left" if value >= 0 else "right"
        x = float(value) + offset if value >= 0 else float(value) - offset
        ax.text(
            x,
            yi,
            f"{value:.2f}",
            va="center",
            ha=ha,
            fontsize=13,
            fontweight="bold",
            color=DARK,
        )

    fig.text(0.32, 0.94, title, fontsize=24, fontweight="bold", color=DARK)
    fig.text(0.32, 0.89, subtitle, fontsize=16, fontweight="bold", color=DARK)
    fig.subplots_adjust(left=0.32, right=0.96, top=0.84, bottom=0.13)
    return fig


def plot_passing_risk_profile(
    player_pax: pd.DataFrame,
    *,
    min_passes: int = 300,
    season_name: str | None = "2025",
    source: str = "skillcorner",
    expected_col: str | None = None,
    actual_col: str = "completion_rate",
    pax_col: str | None = None,
    title: str = "Profiling Passing Risk",
    subtitle: str | None = None,
    label_n: int = 6,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Scatter actual completion against expected completion."""
    source_avg_col, source_pax_col = _source_columns(source)
    expected_col = expected_col or source_avg_col
    pax_col = pax_col or source_pax_col
    subtitle = subtitle or _default_subtitle(season_name, source).replace(
        "SkillCorner Open Data", "SkillCorner Open Data players"
    )
    frame = _filter_player_frame(player_pax, min_passes=min_passes, season_name=season_name)
    for col in [expected_col, actual_col, pax_col]:
        if col not in frame.columns:
            raise ValueError(f"Missing required column: {col}")
    frame = frame.dropna(subset=[expected_col, actual_col, pax_col]).copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 8))
    else:
        fig = ax.figure

    x = frame[expected_col].astype(float)
    y = frame[actual_col].astype(float)
    ax.scatter(
        x,
        y,
        s=70,
        color=POINT_RED,
        edgecolor=DARK,
        linewidth=0.5,
        alpha=0.9,
    )

    low = float(np.floor(min(x.min(), y.min()) * 20) / 20)
    high = float(np.ceil(max(x.max(), y.max()) * 20) / 20)
    low = max(0.0, low - 0.02)
    high = min(1.0, high + 0.02)
    ax.plot([low, high], [low, high], color="#C6C6C6", linewidth=1.5)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Expected Completion %", fontsize=13, fontweight="bold", color=DARK)
    ax.set_ylabel("Pass Completion %", fontsize=13, fontweight="bold", color=DARK)
    ax.grid(color="#E2E2E2", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#BDBDBD")

    label_rows = pd.concat(
        [
            frame.nlargest(label_n, pax_col),
            frame.nsmallest(label_n, pax_col),
            frame.nsmallest(max(2, label_n // 2), expected_col),
            frame.nlargest(max(2, label_n // 2), expected_col),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["season_name", "team_shortname", "player_name"])
    for _, row in label_rows.iterrows():
        ax.annotate(
            str(row["player_name"]),
            (float(row[expected_col]), float(row[actual_col])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color=DARK,
        )

    ax.annotate(
        "Harder passes",
        xy=(low + (high - low) * 0.12, low + (high - low) * 0.05),
        xytext=(low + (high - low) * 0.27, low + (high - low) * 0.05),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.5},
        ha="center",
        va="center",
        fontsize=10,
        color=DARK,
    )
    ax.annotate(
        "Easier passes",
        xy=(high - (high - low) * 0.10, low + (high - low) * 0.05),
        xytext=(high - (high - low) * 0.26, low + (high - low) * 0.05),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.5},
        ha="center",
        va="center",
        fontsize=10,
        color=DARK,
    )
    ax.text(
        high - (high - low) * 0.02,
        high - (high - low) * 0.04,
        "Executed better\nthan expected",
        ha="right",
        va="top",
        fontsize=10,
        color=DARK,
    )
    ax.text(
        high - (high - low) * 0.02,
        high - (high - low) * 0.16,
        "Executed worse\nthan expected",
        ha="right",
        va="top",
        fontsize=10,
        color=DARK,
    )

    fig.text(0.08, 0.95, title, fontsize=22, fontweight="bold", color=DARK)
    fig.text(0.08, 0.92, subtitle, fontsize=13, color=DARK)
    fig.text(0.78, 0.06, f"Minimum {min_passes} passes", fontsize=10, color=DARK)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.12)
    return fig


def plot_player_pax_pass_map(
    scored_passes: pd.DataFrame,
    *,
    season_name: str | int,
    team_shortname: str,
    player_name: str,
    source: str = "skillcorner",
    custom_col: str = "model_xpass",
    positive_threshold: float = 0.5,
    negative_threshold: float = -0.8,
    title: str | None = None,
    subtitle: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot one player's high-positive and high-negative per-pass PAx actions.

    Purple passes are completed passes whose per-pass PAx is above
    ``positive_threshold``. Red passes are missed passes whose per-pass PAx is
    below ``negative_threshold``. The default thresholds match the Analyst-style
    example figure.
    """
    xpass_col = _pass_xpass_column(source, custom_col=custom_col)
    required = [
        "season_name",
        "team_shortname",
        "player_name",
        "pass_completed",
        "passer_x",
        "passer_y",
        "target_x",
        "target_y",
        xpass_col,
    ]
    missing = [col for col in required if col not in scored_passes.columns]
    if missing:
        raise ValueError(f"Missing required columns for player pass map: {missing}")

    frame = scored_passes[
        scored_passes["season_name"].astype(str).eq(str(season_name))
        & scored_passes["team_shortname"].astype(str).eq(str(team_shortname))
        & scored_passes["player_name"].astype(str).eq(str(player_name))
    ].copy()
    frame = frame.dropna(subset=["passer_x", "passer_y", "target_x", "target_y", xpass_col])
    if frame.empty:
        raise ValueError(
            f"No matching scored passes for {season_name=} {team_shortname=} {player_name=}"
        )

    frame["pass_completed"] = frame["pass_completed"].astype(bool)
    frame["pax_pass"] = frame["pass_completed"].astype(float) - pd.to_numeric(frame[xpass_col], errors="coerce")
    positive = frame[frame["pax_pass"].gt(positive_threshold)].copy()
    negative = frame[frame["pax_pass"].lt(negative_threshold)].copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 7.5))
    else:
        fig = ax.figure

    _draw_pitch(ax)
    for rows, color, zorder in [(negative, RED, 2), (positive, PURPLE, 3)]:
        for _, row in rows.iterrows():
            x0 = float(row["passer_x"]) + PITCH_X / 2.0
            y0 = float(row["passer_y"]) + PITCH_Y / 2.0
            x1 = float(row["target_x"]) + PITCH_X / 2.0
            y1 = float(row["target_y"]) + PITCH_Y / 2.0
            ax.plot([x0, x1], [y0, y1], color=color, lw=2.1, alpha=0.78, zorder=zorder)
            ax.scatter(
                [x1],
                [y1],
                s=28,
                facecolor="white",
                edgecolor=color,
                linewidth=1.1,
                zorder=zorder + 1,
            )

    title = title or str(player_name)
    subtitle = subtitle or f"{team_shortname} | {season_name} | {_source_label(source)}"
    fig.text(0.17, 0.93, title, fontsize=22, fontweight="bold", color=DARK)
    fig.text(0.17, 0.895, subtitle, fontsize=16, color=DARK)

    summary = f"completion {frame['pass_completed'].mean():.1%} | PAx/100 {frame['pax_pass'].mean() * 100:.2f}"
    fig.text(0.17, 0.865, summary, fontsize=10.5, color=DARK)

    # Direction legend
    y_legend = -4.3
    ax.annotate(
        "",
        xy=(18, y_legend),
        xytext=(0, y_legend),
        arrowprops={"arrowstyle": "-|>", "lw": 2.0, "color": "#888888"},
        annotation_clip=False,
    )
    ax.text(20.5, y_legend, "pass start", va="center", fontsize=11, color=DARK)
    ax.plot([38, 52], [y_legend, y_legend], color="#888888", lw=1.4, clip_on=False)
    ax.scatter([52], [y_legend], s=22, facecolor="white", edgecolor="#888888", linewidth=1.2, clip_on=False)
    ax.text(54, y_legend, "pass end", va="center", fontsize=11, color=DARK)

    ax.scatter([78], [y_legend], marker="s", s=110, color=PURPLE, clip_on=False)
    ax.text(80.5, y_legend, f"PAx > {positive_threshold:g}", va="center", fontsize=11, color=DARK)
    ax.scatter([78], [y_legend - 4.0], marker="s", s=110, color=RED, clip_on=False)
    ax.text(80.5, y_legend - 4.0, f"PAx < {negative_threshold:g}", va="center", fontsize=11, color=DARK)

    fig.subplots_adjust(left=0.06, right=0.96, top=0.84, bottom=0.08)
    return fig


def save_xpass_report_figures(
    player_pax: pd.DataFrame,
    *,
    out_dir: str | Path = DEFAULT_FIGURE_DIR,
    min_passes: int = 300,
    season_name: str | None = "2025",
    source: str = "skillcorner",
    exclude_goalkeepers_from_xp: bool = False,
    exclude_goalkeepers_from_pax: bool = False,
    dpi: int = 180,
) -> dict[str, Path]:
    """Create the three xP/PAx report figures and save them."""
    source_key = str(source).lower().strip()
    _source_columns(source_key)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        f"{source_key}_xp_top_bottom": plot_xp_top_bottom(
            player_pax,
            min_passes=min_passes,
            season_name=season_name,
            source=source_key,
            exclude_goalkeepers=exclude_goalkeepers_from_xp,
        ),
        f"{source_key}_pax_top_bottom": plot_pax_top_bottom(
            player_pax,
            min_passes=min_passes,
            season_name=season_name,
            source=source_key,
            exclude_goalkeepers=exclude_goalkeepers_from_pax,
        ),
        f"{source_key}_passing_risk_profile": plot_passing_risk_profile(
            player_pax,
            min_passes=min_passes,
            season_name=season_name,
            source=source_key,
        ),
    }
    paths: dict[str, Path] = {}
    for name, fig in figures.items():
        path = out / f"{name}.png"
        fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(fig)
        paths[name] = path
    return paths


def save_skillcorner_report_figures(
    player_pax: pd.DataFrame,
    *,
    out_dir: str | Path = DEFAULT_FIGURE_DIR,
    min_passes: int = 300,
    season_name: str | None = "2025",
    dpi: int = 180,
) -> dict[str, Path]:
    """Backward-compatible wrapper for SkillCorner report figures."""
    return save_xpass_report_figures(
        player_pax,
        out_dir=out_dir,
        min_passes=min_passes,
        season_name=season_name,
        source="skillcorner",
        dpi=dpi,
    )


RANK_COLORS = ["#08BFC6", "#F28E17", "#B635D0", "#E6C400", "#4BD316", "#EC159E"]


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _cluster_label(value: object) -> str:
    raw = str(value)
    return PASS_CLUSTER_LABELS_EN.get(raw, raw)


def _cluster_plot_frame(
    scored_passes: pd.DataFrame,
    *,
    season_name: str | int | None = "2025",
    cluster_col: str = "pass_cluster",
    open_play_only: bool = True,
    final_third_entries_only: bool = False,
) -> pd.DataFrame:
    frame = scored_passes.copy()
    if cluster_col not in frame.columns:
        frame = add_pass_cluster_features(frame)
    if season_name is not None and "season_name" in frame.columns:
        frame = frame[frame["season_name"].astype(str).eq(str(season_name))].copy()

    if open_play_only:
        start_type = frame.get("start_type", pd.Series("", index=frame.index)).astype("string").str.lower()
        phase = frame.get("team_in_possession_phase_type", pd.Series("", index=frame.index)).astype("string").str.lower()
        set_piece = start_type.str.contains("free_kick|corner|throw_in|goal_kick", regex=True, na=False) | phase.eq("set_play")
        frame = frame[~set_piece].copy()

    if final_third_entries_only:
        target_third = frame.get("player_targeted_third_pass", frame.get("third_end", pd.Series("", index=frame.index)))
        start_third = frame.get("third_start", pd.Series("", index=frame.index))
        target_third = target_third.astype("string").str.lower()
        start_third = start_third.astype("string").str.lower()
        final_third_by_label = target_third.eq("attacking_third") & ~start_third.eq("attacking_third")
        final_third_x = frame["target_x"].ge(PITCH_X / 6.0) & frame["passer_x"].lt(PITCH_X / 6.0)
        frame = frame[final_third_by_label | final_third_x].copy()

    required = ["team_shortname", "passer_x", "passer_y", "target_x", "target_y", cluster_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for cluster map: {missing}")
    return frame.dropna(subset=required).copy()


def rank_team_pass_clusters(
    scored_passes: pd.DataFrame,
    *,
    season_name: str | int | None = "2025",
    cluster_col: str = "pass_cluster",
    top_n: int = 5,
    min_cluster_passes: int = 20,
    open_play_only: bool = True,
    final_third_entries_only: bool = False,
    rank_by: str = "overrepresentation",
) -> pd.DataFrame:
    """Rank each team's pass clusters by overrepresentation or volume.

    ``overrepresentation`` is team cluster share minus league cluster share on
    the same filtered pass set. This mirrors the StatsBomb-style question:
    which pass patterns does this team use more than the league baseline?
    """
    frame = _cluster_plot_frame(
        scored_passes,
        season_name=season_name,
        cluster_col=cluster_col,
        open_play_only=open_play_only,
        final_third_entries_only=final_third_entries_only,
    )
    cluster_values = frame[cluster_col].astype(str)
    league_counts = cluster_values.value_counts().rename_axis(cluster_col).reset_index(name="league_cluster_passes")
    league_counts["league_share"] = league_counts["league_cluster_passes"] / league_counts["league_cluster_passes"].sum()

    team_counts = (
        frame.assign(_cluster=cluster_values)
        .groupby(["team_shortname", "_cluster"], dropna=False)
        .size()
        .rename("team_cluster_passes")
        .reset_index()
        .rename(columns={"_cluster": cluster_col})
    )
    team_totals = frame.groupby("team_shortname").size().rename("team_passes").reset_index()
    ranked = team_counts.merge(team_totals, on="team_shortname", how="left")
    ranked = ranked.merge(league_counts, on=cluster_col, how="left")
    ranked["team_share"] = ranked["team_cluster_passes"] / ranked["team_passes"]
    ranked["overrepresentation"] = ranked["team_share"] - ranked["league_share"]
    ranked["overrepresentation_pp"] = ranked["overrepresentation"] * 100.0
    ranked["overrepresentation_ratio"] = ranked["team_share"] / ranked["league_share"].replace(0, np.nan)
    ranked["pass_cluster_label"] = ranked[cluster_col].map(_cluster_label)
    if season_name is not None:
        ranked["season_name"] = str(season_name)

    ranked = ranked[ranked["team_cluster_passes"].ge(int(min_cluster_passes))].copy()
    metric_map = {
        "overrepresentation": "overrepresentation",
        "share": "team_share",
        "passes": "team_cluster_passes",
        "frequency": "team_cluster_passes",
    }
    metric_col = metric_map.get(rank_by)
    if metric_col is None:
        raise ValueError("rank_by must be 'overrepresentation', 'share', 'passes', or 'frequency'")
    ranked = ranked.sort_values(["team_shortname", metric_col, "team_cluster_passes"], ascending=[True, False, False])
    ranked["cluster_rank"] = ranked.groupby("team_shortname").cumcount() + 1
    ranked = ranked[ranked["cluster_rank"].le(int(top_n))].copy()
    return ranked.sort_values(["team_shortname", "cluster_rank"], ignore_index=True)


def _plot_pass_arrows(ax: plt.Axes, rows: pd.DataFrame, *, color: str, lw: float = 0.9, alpha: float = 0.78) -> None:
    from matplotlib.patches import FancyArrowPatch

    for _, row in rows.iterrows():
        x0 = float(row["passer_x"]) + PITCH_X / 2.0
        y0 = float(row["passer_y"]) + PITCH_Y / 2.0
        x1 = float(row["target_x"]) + PITCH_X / 2.0
        y1 = float(row["target_y"]) + PITCH_Y / 2.0
        if abs(x1 - x0) + abs(y1 - y0) < 0.5:
            continue
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=6.5,
                linewidth=lw,
                color=color,
                alpha=alpha,
                shrinkA=0,
                shrinkB=0,
                zorder=4,
            )
        )


def _rank_legend_handles(top_n: int, *, compact: bool = False) -> list[Line2D]:
    handles: list[Line2D] = []
    for rank in range(1, top_n + 1):
        if compact:
            label = _ordinal(rank)
        else:
            label = "Most overrepresented" if rank == 1 else f"{_ordinal(rank)} most overrepresented"
        handles.append(
            Line2D(
                [0],
                [0],
                color=RANK_COLORS[(rank - 1) % len(RANK_COLORS)],
                lw=2.0,
                marker=">",
                markersize=7,
                label=label,
            )
        )
    return handles


def plot_team_overrepresented_pass_clusters(
    scored_passes: pd.DataFrame,
    *,
    season_name: str | int | None = "2025",
    top_n: int = 5,
    passes_per_cluster: int = 18,
    min_cluster_passes: int = 20,
    open_play_only: bool = True,
    final_third_entries_only: bool = False,
    rank_by: str = "overrepresentation",
    team_order: list[str] | None = None,
    n_cols: int = 4,
    random_state: int = 7,
    title: str = "Open Play Buildup",
    subtitle: str | None = None,
) -> plt.Figure:
    """Plot each team's top overrepresented pass clusters on mini pitches."""
    frame = _cluster_plot_frame(
        scored_passes,
        season_name=season_name,
        open_play_only=open_play_only,
        final_third_entries_only=final_third_entries_only,
    )
    rankings = rank_team_pass_clusters(
        frame,
        season_name=None,
        top_n=top_n,
        min_cluster_passes=min_cluster_passes,
        open_play_only=False,
        final_third_entries_only=False,
        rank_by=rank_by,
    )
    if team_order is None:
        team_order = (
            frame.groupby("team_shortname")
            .size()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
    teams = [team for team in team_order if team in set(frame["team_shortname"].astype(str))]
    if not teams:
        raise ValueError("No teams available for pass cluster map")

    n_rows = int(np.ceil(len(teams) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.25, n_rows * 2.55 + 1.35))
    axes_arr = np.asarray(axes).reshape(-1)
    cluster_values = frame["pass_cluster"].astype(str)

    for team_idx, (team, ax) in enumerate(zip(teams, axes_arr)):
        _draw_pitch(ax)
        ax.set_title(str(team), fontsize=10, fontweight="bold", color=DARK, pad=3)
        team_rankings = rankings[rankings["team_shortname"].astype(str).eq(str(team))].sort_values("cluster_rank")
        for _, rank_row in team_rankings.iterrows():
            rank = int(rank_row["cluster_rank"])
            cluster = str(rank_row["pass_cluster"])
            rows = frame[frame["team_shortname"].astype(str).eq(str(team)) & cluster_values.eq(cluster)]
            if rows.empty:
                continue
            sample_n = min(int(passes_per_cluster), len(rows))
            rows = rows.sample(sample_n, random_state=random_state + team_idx * 100 + rank)
            _plot_pass_arrows(ax, rows, color=RANK_COLORS[(rank - 1) % len(RANK_COLORS)], lw=0.75, alpha=0.78)

    for ax in axes_arr[len(teams):]:
        ax.axis("off")

    season_text = "All seasons" if season_name is None else str(season_name)
    if subtitle is None:
        mode = "Final-third entries" if final_third_entries_only else "Open-play passes"
        subtitle = f"Top {top_n} overrepresented pass clusters for each team | SkillCorner Open Data {season_text} | {mode}"
    fig.text(0.055, 0.975, title, fontsize=24, fontweight="bold", color="black", ha="left", va="top")
    fig.text(0.055, 0.94, subtitle, fontsize=13, color="black", ha="left", va="top")
    fig.legend(
        handles=_rank_legend_handles(top_n),
        loc="lower center",
        ncol=min(top_n, 5),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(left=0.045, right=0.98, top=0.86, bottom=0.09, wspace=0.12, hspace=0.22)
    return fig


def plot_team_pass_cluster_panels(
    scored_passes: pd.DataFrame,
    *,
    team_shortname: str,
    season_name: str | int | None = "2025",
    top_n: int = 6,
    passes_per_cluster: int = 160,
    min_cluster_passes: int = 10,
    open_play_only: bool = True,
    final_third_entries_only: bool = True,
    rank_by: str = "passes",
    n_cols: int = 3,
    random_state: int = 11,
    title: str | None = None,
    subtitle: str | None = None,
) -> plt.Figure:
    """Plot one team's ranked pass clusters in separate pitch panels."""
    frame = _cluster_plot_frame(
        scored_passes,
        season_name=season_name,
        open_play_only=open_play_only,
        final_third_entries_only=final_third_entries_only,
    )
    frame = frame[frame["team_shortname"].astype(str).eq(str(team_shortname))].copy()
    if frame.empty:
        raise ValueError(f"No passes available for team_shortname={team_shortname!r}")

    rankings = rank_team_pass_clusters(
        frame,
        season_name=None,
        top_n=top_n,
        min_cluster_passes=min_cluster_passes,
        open_play_only=False,
        final_third_entries_only=False,
        rank_by=rank_by,
    )
    rankings = rankings[rankings["team_shortname"].astype(str).eq(str(team_shortname))].sort_values("cluster_rank")
    if rankings.empty:
        raise ValueError(f"No ranked clusters available for team_shortname={team_shortname!r}")

    n_panels = min(top_n, len(rankings))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.1, n_rows * 4.55 + 1.35))
    axes_arr = np.asarray(axes).reshape(-1)
    cluster_values = frame["pass_cluster"].astype(str)

    for panel_idx, (_, rank_row) in enumerate(rankings.head(n_panels).iterrows()):
        ax = axes_arr[panel_idx]
        _draw_pitch(ax)
        rank = int(rank_row["cluster_rank"])
        cluster = str(rank_row["pass_cluster"])
        rows = frame[cluster_values.eq(cluster)]
        sample_n = min(int(passes_per_cluster), len(rows))
        rows = rows.sample(sample_n, random_state=random_state + rank)
        _plot_pass_arrows(ax, rows, color=RANK_COLORS[(rank - 1) % len(RANK_COLORS)], lw=1.0, alpha=0.82)
        label = _cluster_label(cluster)
        ax.set_title(
            f"{_ordinal(rank)} | {label}\n"
            f"share {rank_row['team_share']:.1%}, overrep {rank_row['overrepresentation_pp']:+.1f} pp",
            fontsize=10,
            fontweight="bold",
            color=DARK,
            pad=4,
        )
        ax.annotate(
            "",
            xy=(PITCH_X + 1.0, PITCH_Y / 2.0),
            xytext=(PITCH_X - 9.0, PITCH_Y / 2.0),
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "black"},
            annotation_clip=False,
        )

    for ax in axes_arr[n_panels:]:
        ax.axis("off")

    season_text = "All seasons" if season_name is None else str(season_name)
    if title is None:
        title = f"How do {team_shortname} pass into the final third?" if final_third_entries_only else f"{team_shortname} pass cluster profile"
    if subtitle is None:
        mode = "final-third entries" if final_third_entries_only else "open-play passes"
        subtitle = f"SkillCorner Open Data {season_text} | Ranked by {rank_by} | {mode}"
    fig.text(0.055, 0.975, title, fontsize=22, fontweight="bold", color="black", ha="left", va="top")
    fig.text(0.055, 0.94, subtitle, fontsize=14, color="black", ha="left", va="top")
    fig.legend(
        handles=_rank_legend_handles(n_panels, compact=True),
        loc="lower center",
        ncol=min(n_panels, 6),
        frameon=False,
        fontsize=10,
        title="Pass cluster rank",
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(left=0.055, right=0.98, top=0.86, bottom=0.10, wspace=0.16, hspace=0.26)
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SkillCorner xPass / PAx report figures.")
    parser.add_argument("--player-pax", type=Path, default=DEFAULT_PLAYER_PAX_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--min-passes", type=int, default=300)
    parser.add_argument("--season-name", default="2025")
    parser.add_argument("--source", choices=["skillcorner", "custom"], default="skillcorner")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    player_pax = pd.read_csv(args.player_pax)
    paths = save_xpass_report_figures(
        player_pax,
        out_dir=args.out_dir,
        min_passes=args.min_passes,
        season_name=args.season_name,
        source=args.source,
        dpi=args.dpi,
    )
    for name, path in paths.items():
        print(f"[figure] {name}: {path}")


if __name__ == "__main__":
    main()
