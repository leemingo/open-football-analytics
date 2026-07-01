"""Rule-based pass clustering and xPass / PAx summaries.

The clusters in this module are deliberately interpretable. They use the
geometric start/end pass features already built for the xPass model and, where
available, SkillCorner Dynamic Events context such as pass range, direction,
third/channel, penalty-area target, line-break flags, and possession phase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xpass.xpass_features import SKILLCORNER_XPASS_COLUMN, TARGET_COLUMN


PASS_CLUSTER_ORDER = [
    "Short lateral pass",
    "Build-up pass",
    "Progressive pass",
    "Wide progression pass",
    "Switch of play",
    "Box entry pass",
    "Penetrative pass",
    "Long pass",
]

PASS_CLUSTER_LABELS_EN = {
    label: label for label in PASS_CLUSTER_ORDER
}

SOURCE_XPASS_COLUMNS = {
    "skillcorner": SKILLCORNER_XPASS_COLUMN,
    "custom": "model_xpass",
}

BUILD_UP_POSITIONS = {"gk", "cb", "lcb", "rcb", "lb", "rb", "lwb", "rwb", "dm", "ldm", "rdm"}
WIDE_CHANNELS = {"wide_left", "wide_right", "left", "right"}


def _series(df: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index)
    return df[column]


def _num(df: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    return pd.to_numeric(_series(df, column, default), errors="coerce")


def _str(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    return _series(df, column, default).astype("string").fillna(default).str.lower()


def _bool(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    source = f"{column}_bool" if f"{column}_bool" in df.columns else column
    if source not in df.columns:
        return pd.Series(default, index=df.index, dtype="bool")
    s = df[source]
    if s.dtype == bool:
        return s.fillna(default).astype(bool)
    mapped = (
        s.astype("string")
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    )
    return mapped.fillna(default).astype(bool)


def _xpass_column(source: str, custom_col: str = "model_xpass") -> str:
    key = str(source).lower().strip()
    if key == "custom":
        return custom_col
    if key in SOURCE_XPASS_COLUMNS:
        return SOURCE_XPASS_COLUMNS[key]
    if source in SOURCE_XPASS_COLUMNS.values():
        return source
    raise ValueError("source must be 'skillcorner', 'custom', or a known xPass column")


def add_pass_cluster_features(passes: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable pass-type labels and diagnostic rule flags.

    Priority matters. Box entries and line-breaking/penetrative passes are
    labelled before broad categories such as long passes, so high-value tactical
    actions remain visible instead of being swallowed by generic distance rules.
    """
    out = passes.copy()

    passer_x = _num(out, "passer_x")
    passer_y = _num(out, "passer_y")
    target_x = _num(out, "target_x")
    target_y = _num(out, "target_y")
    distance = _num(out, "pass_distance_feature")
    if distance.isna().all():
        distance = np.hypot(target_x - passer_x, target_y - passer_y)
    progression = _num(out, "pass_progression")
    if progression.isna().all():
        progression = target_x - passer_x
    lateral = _num(out, "pass_lateral_distance")
    if lateral.isna().all():
        lateral = (target_y - passer_y).abs()
    goal_gain = _num(out, "target_goal_distance_gain")

    pass_range = _str(out, "pass_range")
    direction = _str(out, "pass_direction")
    start_type = _str(out, "start_type")
    phase = _str(out, "team_in_possession_phase_type")
    player_position = _str(out, "player_position")
    channel_start = _str(out, "channel_start")
    channel_end = _str(out, "channel_end")
    target_channel = _str(out, "player_targeted_channel_pass")
    third_start = _str(out, "third_start")
    third_end = _str(out, "third_end")
    target_third = _str(out, "player_targeted_third_pass")
    furthest_line_break = _str(out, "furthest_line_break")

    high_pass = _bool(out, "high_pass")
    target_penalty_area = _bool(out, "player_targeted_penalty_area_pass") | _bool(out, "penalty_area_end")
    start_penalty_area = _bool(out, "penalty_area_start")
    first_line_break = _bool(out, "first_line_break")
    second_last_line_break = _bool(out, "second_last_line_break")
    last_line_break = _bool(out, "last_line_break")

    set_piece_start = start_type.str.contains("free_kick|corner|throw_in|goal_kick", regex=True)
    wide_start = channel_start.isin(WIDE_CHANNELS) | passer_y.abs().ge(23.0)
    wide_target = target_channel.isin(WIDE_CHANNELS) | channel_end.isin(WIDE_CHANNELS) | target_y.abs().ge(23.0)
    opposite_half_switch = (passer_y * target_y).lt(-80.0)

    box_entry = (
        target_penalty_area
        | (target_x.ge(36.0) & target_y.abs().le(20.16))
    ) & ~start_penalty_area

    penetration = (
        last_line_break
        | furthest_line_break.eq("last")
        | (
            second_last_line_break
            & target_x.ge(20.0)
            & (progression.ge(8.0) | goal_gain.ge(8.0))
        )
        | (
            target_third.eq("attacking_third")
            & target_x.ge(28.0)
            & progression.ge(10.0)
            & distance.le(35.0)
        )
    )

    switch = (
        distance.ge(28.0)
        & lateral.ge(22.0)
        & (opposite_half_switch | (wide_start & wide_target))
    )

    wide_progression = (
        wide_target
        & lateral.ge(8.0)
        & progression.ge(-5.0)
        & ~switch
    )

    long_pass = distance.ge(30.0) | pass_range.eq("long") | (high_pass & distance.ge(25.0))
    forward_pass = (
        direction.eq("forward")
        | progression.ge(8.0)
        | goal_gain.ge(8.0)
        | first_line_break
        | second_last_line_break
    )
    build_up = (
        phase.eq("build_up")
        | (third_start.eq("defensive_third") & target_x.le(5.0))
        | (passer_x.le(-10.0) & distance.le(30.0))
        | (player_position.isin(BUILD_UP_POSITIONS) & passer_x.le(5.0) & distance.le(35.0))
    )
    short_lateral = (
        (distance.le(15.0) | pass_range.eq("short"))
        & (progression.abs().le(7.0) | direction.str.contains("sideway", regex=False))
    )

    labels = np.full(len(out), "Short lateral pass", dtype=object)
    rules = [
        ("Box entry pass", box_entry),
        ("Penetrative pass", penetration),
        ("Switch of play", switch),
        ("Long pass", long_pass),
        ("Wide progression pass", wide_progression),
        ("Progressive pass", forward_pass),
        ("Build-up pass", build_up),
        ("Short lateral pass", short_lateral),
    ]
    assigned = pd.Series(False, index=out.index)
    for label, mask in rules:
        mask = mask.fillna(False) & ~assigned
        labels[np.asarray(mask)] = label
        assigned = assigned | mask

    fallback_long = ~assigned & long_pass.fillna(False)
    fallback_forward = ~assigned & ~fallback_long & progression.ge(8.0).fillna(False)
    fallback_build = ~assigned & ~fallback_long & ~fallback_forward & passer_x.le(-10.0).fillna(False)
    labels[np.asarray(fallback_long)] = "Long pass"
    labels[np.asarray(fallback_forward)] = "Progressive pass"
    labels[np.asarray(fallback_build)] = "Build-up pass"

    out["pass_cluster"] = pd.Categorical(labels, categories=PASS_CLUSTER_ORDER, ordered=True)
    out["pass_cluster_reason"] = out["pass_cluster"].astype("string")
    out["cluster_is_box_entry"] = box_entry.fillna(False)
    out["cluster_is_penetration"] = penetration.fillna(False)
    out["cluster_is_switch"] = switch.fillna(False)
    out["cluster_is_wide_progression"] = wide_progression.fillna(False)
    out["cluster_is_long_pass"] = long_pass.fillna(False)
    out["cluster_is_forward_pass"] = forward_pass.fillna(False)
    out["cluster_is_build_up"] = build_up.fillna(False)
    out["cluster_is_short_lateral"] = short_lateral.fillna(False)
    out["cluster_is_set_piece_start"] = set_piece_start.fillna(False)
    out["cluster_any_line_break"] = (first_line_break | second_last_line_break | last_line_break).fillna(False)
    return out


def _base_frame(scored: pd.DataFrame, *, source: str, custom_col: str = "model_xpass") -> tuple[pd.DataFrame, str, str]:
    xpass_col = _xpass_column(source, custom_col=custom_col)
    if TARGET_COLUMN not in scored.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    if xpass_col not in scored.columns:
        raise ValueError(f"Missing xPass column: {xpass_col}")
    prefix = "custom" if xpass_col == custom_col else "skillcorner" if xpass_col == SKILLCORNER_XPASS_COLUMN else xpass_col
    frame = scored.copy()
    if "pass_cluster" not in frame.columns:
        frame = add_pass_cluster_features(frame)
    frame[xpass_col] = pd.to_numeric(frame[xpass_col], errors="coerce")
    frame = frame[frame[xpass_col].notna()].copy()
    frame["_completed"] = frame[TARGET_COLUMN].astype(float)
    frame["_xpass"] = frame[xpass_col].astype(float)
    frame["_pax"] = frame["_completed"] - frame["_xpass"]
    return frame, xpass_col, prefix


def summarize_pass_clusters(
    scored: pd.DataFrame,
    *,
    source: str = "skillcorner",
    custom_col: str = "model_xpass",
    min_passes: int = 0,
) -> pd.DataFrame:
    """Aggregate completion, xPass, and PAx by pass cluster."""
    frame, _, prefix = _base_frame(scored, source=source, custom_col=custom_col)
    agg = {
        "passes": ("event_id", "size") if "event_id" in frame.columns else ("_completed", "size"),
        "completed": ("_completed", "sum"),
        "expected_completed": ("_xpass", "sum"),
        "avg_xpass": ("_xpass", "mean"),
        "pax": ("_pax", "sum"),
    }
    optional = {
        "avg_pass_distance": "pass_distance_feature",
        "avg_progression": "pass_progression",
        "avg_goal_distance_gain": "target_goal_distance_gain",
        "avg_lateral_distance": "pass_lateral_distance",
        "high_pass_share": "high_pass",
        "set_piece_share": "cluster_is_set_piece_start",
        "avg_passing_options": "n_passing_options",
        "line_break_share": "cluster_any_line_break",
    }
    for out_col, source_col in optional.items():
        if source_col in frame.columns:
            agg[out_col] = (source_col, "mean")
    summary = (
        frame.groupby("pass_cluster", observed=False, dropna=False)
        .agg(**agg)
        .reindex(PASS_CLUSTER_ORDER)
        .reset_index()
    )
    summary["source"] = prefix
    summary["share"] = summary["passes"] / summary["passes"].sum()
    summary["completion_rate"] = summary["completed"] / summary["passes"]
    summary["pax_per_pass"] = summary["pax"] / summary["passes"]
    summary["pax_per_100"] = summary["pax_per_pass"] * 100.0
    if min_passes:
        summary = summary[summary["passes"].ge(int(min_passes))].copy()
    return summary


def summarize_player_cluster_profile(
    scored: pd.DataFrame,
    *,
    source: str = "skillcorner",
    custom_col: str = "model_xpass",
    min_passes: int = 300,
) -> pd.DataFrame:
    """Return each player's xPass / PAx and pass-type distribution."""
    frame, _, prefix = _base_frame(scored, source=source, custom_col=custom_col)
    group_cols = [col for col in ["season_name", "team_shortname", "player_name"] if col in frame.columns]
    if not group_cols:
        group_cols = ["player_name"]

    base_agg = {
        "passes": ("_completed", "size"),
        "completed": ("_completed", "sum"),
        "expected_completed": ("_xpass", "sum"),
        "avg_xpass": ("_xpass", "mean"),
        "pax": ("_pax", "sum"),
        "avg_pass_distance": ("pass_distance_feature", "mean"),
        "avg_progression": ("pass_progression", "mean"),
    }
    if "player_position" in frame.columns:
        base_agg["player_position"] = (
            "player_position",
            lambda s: s.astype("string").dropna().mode().iloc[0]
            if not s.astype("string").dropna().empty
            else pd.NA,
        )

    summary = frame.groupby(group_cols, dropna=False).agg(**base_agg).reset_index()
    summary["completion_rate"] = summary["completed"] / summary["passes"]
    summary["pax_per_100"] = summary["pax"] / summary["passes"] * 100.0
    summary = summary[summary["passes"].ge(int(min_passes))].copy()

    counts = (
        frame.groupby(group_cols + ["pass_cluster"], observed=False, dropna=False)
        .size()
        .rename("cluster_passes")
        .reset_index()
    )
    wide = counts.pivot_table(
        index=group_cols,
        columns="pass_cluster",
        values="cluster_passes",
        fill_value=0,
        observed=False,
    )
    for cluster in PASS_CLUSTER_ORDER:
        if cluster not in wide.columns:
            wide[cluster] = 0
    wide = wide[PASS_CLUSTER_ORDER].reset_index()
    profile = summary.merge(wide, on=group_cols, how="left")
    for cluster in PASS_CLUSTER_ORDER:
        profile[f"share_{cluster}"] = profile[cluster] / profile["passes"]
    profile["primary_cluster"] = profile[PASS_CLUSTER_ORDER].idxmax(axis=1)
    profile["source"] = prefix
    return profile.sort_values("pax_per_100", ascending=False, ignore_index=True)


def summarize_team_xpass_pax_comparison(
    scored: pd.DataFrame,
    *,
    custom_col: str = "model_xpass",
    min_passes: int = 0,
) -> pd.DataFrame:
    """Compare team-level xPass and PAx from SkillCorner and custom estimates."""
    required = [TARGET_COLUMN, SKILLCORNER_XPASS_COLUMN, custom_col]
    missing = [col for col in required if col not in scored.columns]
    if missing:
        raise ValueError(f"Missing required columns for team comparison: {missing}")
    group_cols = [col for col in ["season_name", "team_shortname"] if col in scored.columns]
    if not group_cols:
        group_cols = ["team_shortname"]

    frame = scored.copy()
    frame[SKILLCORNER_XPASS_COLUMN] = pd.to_numeric(frame[SKILLCORNER_XPASS_COLUMN], errors="coerce")
    frame[custom_col] = pd.to_numeric(frame[custom_col], errors="coerce")
    frame = frame.dropna(subset=[SKILLCORNER_XPASS_COLUMN, custom_col]).copy()
    frame["_completed"] = frame[TARGET_COLUMN].astype(float)
    frame["_skillcorner_pax"] = frame["_completed"] - frame[SKILLCORNER_XPASS_COLUMN]
    frame["_custom_pax"] = frame["_completed"] - frame[custom_col]

    summary = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            passes=("_completed", "size"),
            completed=("_completed", "sum"),
            skillcorner_expected_completed=(SKILLCORNER_XPASS_COLUMN, "sum"),
            custom_expected_completed=(custom_col, "sum"),
            avg_skillcorner_xpass=(SKILLCORNER_XPASS_COLUMN, "mean"),
            avg_custom_xpass=(custom_col, "mean"),
            skillcorner_pax=("_skillcorner_pax", "sum"),
            custom_pax=("_custom_pax", "sum"),
            avg_pass_distance=("pass_distance_feature", "mean"),
            avg_progression=("pass_progression", "mean"),
        )
        .reset_index()
    )
    summary["completion_rate"] = summary["completed"] / summary["passes"]
    summary["skillcorner_pax_per_100"] = summary["skillcorner_pax"] / summary["passes"] * 100.0
    summary["custom_pax_per_100"] = summary["custom_pax"] / summary["passes"] * 100.0
    summary["skillcorner_minus_custom_xpass"] = summary["avg_skillcorner_xpass"] - summary["avg_custom_xpass"]
    summary["skillcorner_minus_custom_expected"] = (
        summary["skillcorner_expected_completed"] - summary["custom_expected_completed"]
    )
    summary["skillcorner_minus_custom_pax_per_100"] = (
        summary["skillcorner_pax_per_100"] - summary["custom_pax_per_100"]
    )
    if min_passes:
        summary = summary[summary["passes"].ge(int(min_passes))].copy()
    return summary.sort_values("skillcorner_pax_per_100", ascending=False, ignore_index=True)
