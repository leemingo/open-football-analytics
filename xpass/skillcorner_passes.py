"""Build pass-level tables from SkillCorner Dynamic Events.

This module mirrors the structure of :mod:`xg.skillcorner_shots`, but treats
``player_possession`` rows ending in a pass as the modelling unit. The table is
designed for two related tasks:

* training a project-owned xPass completion model from SkillCorner event
  context; and
* comparing that model with SkillCorner's supplied
  ``player_targeted_xpass_completion`` value.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from football_cdf.constants import CDF_PERIOD_MAP, PITCH_X, PITCH_Y
from football_cdf.skillcorner_api import META_FILENAMES, find_match_dir
from football_cdf.skillcorner_preprocessing import SkillcornerDataPreprocessor


DEFAULT_SKILLCORNER_ROOT = Path("/data2/MHL/data/skillcorner/kleague")
DEFAULT_OUTPUT_DIR = Path("tmp/data/skillcorner_xpass")
DEFAULT_PASSES_PATH = DEFAULT_OUTPUT_DIR / "passes.parquet"


PASS_SOURCE_COLUMNS = [
    "event_id",
    "index",
    "match_id",
    "frame_start",
    "frame_end",
    "frame_physical_start",
    "time_start",
    "time_end",
    "minute_start",
    "second_start",
    "duration",
    "period",
    "attacking_side_id",
    "attacking_side",
    "event_type_id",
    "event_type",
    "event_subtype_id",
    "event_subtype",
    "player_id",
    "player_name",
    "player_position_id",
    "player_position",
    "team_id",
    "team_shortname",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "channel_id_start",
    "channel_start",
    "third_id_start",
    "third_start",
    "penalty_area_start",
    "channel_id_end",
    "channel_end",
    "third_id_end",
    "third_end",
    "penalty_area_end",
    "game_state_id",
    "game_state",
    "team_score",
    "opponent_team_score",
    "phase_index",
    "player_possession_phase_index",
    "first_player_possession_in_team_possession",
    "last_player_possession_in_team_possession",
    "team_possession_loss_in_phase",
    "team_in_possession_phase_type_id",
    "team_in_possession_phase_type",
    "team_out_of_possession_phase_type_id",
    "team_out_of_possession_phase_type",
    "game_interruption_before_id",
    "game_interruption_before",
    "game_interruption_after_id",
    "game_interruption_after",
    "lead_to_shot",
    "lead_to_goal",
    "distance_covered",
    "trajectory_angle",
    "trajectory_direction_id",
    "trajectory_direction",
    "in_to_out",
    "out_to_in",
    "speed_avg",
    "speed_avg_band_id",
    "speed_avg_band",
    "last_defensive_line_x_start",
    "last_defensive_line_x_end",
    "delta_to_last_defensive_line_start",
    "delta_to_last_defensive_line_end",
    "delta_to_last_defensive_line_gain",
    "last_defensive_line_height_start",
    "last_defensive_line_height_end",
    "last_defensive_line_height_gain",
    "inside_defensive_shape_start",
    "inside_defensive_shape_end",
    "start_type_id",
    "start_type",
    "end_type_id",
    "end_type",
    "consecutive_on_ball_engagements",
    "one_touch",
    "quick_pass",
    "carry",
    "forward_momentum",
    "is_header",
    "hand_pass",
    "pass_angle_received",
    "pass_direction_received_id",
    "pass_direction_received",
    "pass_distance_received",
    "pass_range_received_id",
    "pass_range_received",
    "pass_outcome_id",
    "pass_outcome",
    "pass_distance",
    "pass_range_id",
    "pass_range",
    "pass_angle",
    "pass_direction_id",
    "pass_direction",
    "pass_ahead",
    "high_pass",
    "targeted_passing_option_event_id",
    "n_passing_options",
    "n_off_ball_runs",
    "n_passing_options_line_break",
    "n_passing_options_first_line_break",
    "n_passing_options_second_last_line_break",
    "n_passing_options_last_line_break",
    "n_passing_options_ahead",
    "n_passing_options_dangerous_difficult",
    "n_passing_options_dangerous_not_difficult",
    "n_passing_options_not_dangerous_not_difficult",
    "n_passing_options_not_dangerous_difficult",
    "n_passing_options_at_start",
    "n_passing_options_at_end",
    "n_passing_options_ahead_at_start",
    "n_passing_options_ahead_at_end",
    "n_teammates_ahead_end",
    "n_teammates_ahead_start",
    "n_opponents_ahead_player_in_possession_pass_moment",
    "n_opponents_ahead_pass_reception",
    "n_opponents_bypassed",
    "n_opponents_ahead_end",
    "n_opponents_ahead_start",
    "n_opponents_overtaken",
    "player_targeted_id",
    "player_targeted_name",
    "player_targeted_position_id",
    "player_targeted_position",
    "player_targeted_x_pass",
    "player_targeted_y_pass",
    "player_targeted_channel_pass_id",
    "player_targeted_channel_pass",
    "player_targeted_third_pass_id",
    "player_targeted_third_pass",
    "player_targeted_penalty_area_pass",
    "player_targeted_x_reception",
    "player_targeted_y_reception",
    "player_targeted_channel_reception_id",
    "player_targeted_channel_reception",
    "player_targeted_third_reception_id",
    "player_targeted_third_reception",
    "player_targeted_penalty_area_reception",
    "player_targeted_distance_to_goal_start",
    "player_targeted_distance_to_goal_end",
    "player_targeted_angle_to_goal_start",
    "player_targeted_angle_to_goal_end",
    "player_targeted_average_speed",
    "player_targeted_speed_avg_band_id",
    "player_targeted_speed_avg_band",
    "player_targeted_xpass_completion",
    "player_targeted_difficult_pass_target",
    "player_targeted_xthreat",
    "player_targeted_dangerous",
    "n_player_targeted_opponents_ahead_start",
    "n_player_targeted_opponents_ahead_end",
    "n_player_targeted_teammates_ahead_start",
    "n_player_targeted_teammates_ahead_end",
    "n_player_targeted_teammates_within_5m_start",
    "n_player_targeted_teammates_within_5m_end",
    "n_player_targeted_opponents_within_5m_start",
    "n_player_targeted_opponents_within_5m_end",
    "first_line_break",
    "first_line_break_type_id",
    "first_line_break_type",
    "second_last_line_break",
    "second_last_line_break_type_id",
    "second_last_line_break_type",
    "last_line_break",
    "last_line_break_type_id",
    "last_line_break_type",
    "furthest_line_break_id",
    "furthest_line_break",
    "furthest_line_break_type_id",
    "furthest_line_break_type",
    "organised_defense",
    "defensive_structure",
    "n_defensive_lines",
    "goal_side_start",
    "goal_side_end",
    "xloss_player_possession_start",
    "xloss_player_possession_end",
    "xloss_player_possession_max",
    "xshot_player_possession_start",
    "xshot_player_possession_end",
    "xshot_player_possession_max",
    "is_player_possession_start_matched",
    "is_player_possession_end_matched",
    "is_previous_pass_matched",
    "is_pass_reception_matched",
    "fully_extrapolated",
]


SET_PIECE_START_TYPES = {
    "corner_reception",
    "corner_interception",
    "free_kick_reception",
    "free_kick_interception",
    "goal_kick_reception",
    "goal_kick_interception",
    "throw_in_reception",
    "throw_in_interception",
}


@dataclass(frozen=True)
class MatchContext:
    """Metadata needed to normalize one SkillCorner match."""

    match_id: str
    match_path: Path
    event_path: Path
    meta_path: Path
    raw_metadata: dict
    match_metadata: dict


def _first_existing_file(match_path: Path, candidates: Iterable[str]) -> Path:
    for name in candidates:
        path = match_path / name
        if path.exists():
            return path
    tried = ", ".join(str(match_path / name) for name in candidates)
    raise FileNotFoundError(f"Could not find metadata file. Tried: {tried}")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_match_context(
    skillcorner_root: str | Path,
    match_id: str | int,
    *,
    match_dir: str | Path | None = None,
) -> MatchContext:
    """Resolve match paths and metadata using the project SkillCorner parser."""
    root = Path(skillcorner_root)
    match_id_str = str(match_id)

    if match_dir is not None and Path(match_dir).exists():
        match_path = Path(match_dir)
    else:
        located = find_match_dir(root, match_id_str)
        if located is None:
            raise FileNotFoundError(
                f"Could not find SkillCorner match_id={match_id_str!r} under {root}"
            )
        match_path = located

    meta_path = _first_existing_file(match_path, META_FILENAMES)
    event_path = match_path / SkillcornerDataPreprocessor.EVENT_FILENAME
    if not event_path.exists():
        raise FileNotFoundError(f"Missing dynamic events file: {event_path}")

    raw_metadata = _load_json(meta_path)
    match_metadata = SkillcornerDataPreprocessor.extract_match_metadata(raw_metadata)

    return MatchContext(
        match_id=match_id_str,
        match_path=match_path,
        event_path=event_path,
        meta_path=meta_path,
        raw_metadata=raw_metadata,
        match_metadata=match_metadata,
    )


def load_match_index(
    skillcorner_root: str | Path = DEFAULT_SKILLCORNER_ROOT,
    *,
    season_names: Iterable[str] | None = None,
    include_non_closed: bool = False,
) -> pd.DataFrame:
    """Load the SkillCorner ``matches_index.csv`` with optional filtering."""
    root = Path(skillcorner_root)
    index_path = root / "matches_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing SkillCorner matches index: {index_path}")

    index = pd.read_csv(index_path)
    index["match_id"] = index["match_id"].astype(str)
    index["season_name"] = index["season_name"].astype(str)

    if season_names is not None:
        wanted = {str(value) for value in season_names}
        index = index[index["season_name"].isin(wanted)].copy()
    if not include_non_closed and "status" in index.columns:
        index = index[index["status"].eq("closed")].copy()

    sort_cols = [c for c in ["season_name", "date_time", "match_id"] if c in index.columns]
    return index.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def _home_away(team_id: pd.Series, match_metadata: dict) -> pd.Series:
    team = team_id.astype("string")
    home_team_id = str(match_metadata.get("home_team_id"))
    away_team_id = str(match_metadata.get("away_team_id"))
    return pd.Series(
        np.select([team.eq(home_team_id), team.eq(away_team_id)], ["home", "away"], default=pd.NA),
        index=team_id.index,
        dtype="string",
    )


def _rescale_center_origin(
    series: pd.Series,
    *,
    source_size: float,
    target_size: float,
) -> pd.Series:
    values = _safe_numeric(series)
    if source_size is None or pd.isna(source_size) or float(source_size) == 0.0:
        return values
    return values * (float(target_size) / float(source_size))


def _add_set_piece_flags(passes: pd.DataFrame) -> None:
    if "start_type" in passes.columns:
        start_type = passes["start_type"].astype("string")
        passes["is_set_piece_start"] = start_type.isin(SET_PIECE_START_TYPES)
        passes["is_corner_start"] = start_type.str.startswith("corner", na=False)
        passes["is_free_kick_start"] = start_type.str.startswith("free_kick", na=False)
        passes["is_throw_in_start"] = start_type.str.startswith("throw_in", na=False)
        passes["is_goal_kick_start"] = start_type.str.startswith("goal_kick", na=False)
    else:
        passes["is_set_piece_start"] = False
        passes["is_corner_start"] = False
        passes["is_free_kick_start"] = False
        passes["is_throw_in_start"] = False
        passes["is_goal_kick_start"] = False


def _prepare_match_passes(
    events: pd.DataFrame,
    context: MatchContext,
    *,
    index_row: pd.Series | None = None,
) -> pd.DataFrame:
    passes = events[
        events["event_type"].eq("player_possession")
        & events["end_type"].eq("pass")
    ].copy()
    if passes.empty:
        return passes

    metadata = context.match_metadata
    raw_pitch_length = pd.to_numeric(metadata.get("pitch_length"), errors="coerce")
    raw_pitch_width = pd.to_numeric(metadata.get("pitch_width"), errors="coerce")
    if pd.isna(raw_pitch_length):
        raw_pitch_length = PITCH_X
    if pd.isna(raw_pitch_width):
        raw_pitch_width = PITCH_Y

    x_cols = [
        "x_start",
        "x_end",
        "player_targeted_x_pass",
        "player_targeted_x_reception",
    ]
    y_cols = [
        "y_start",
        "y_end",
        "player_targeted_y_pass",
        "player_targeted_y_reception",
    ]
    for col in x_cols:
        if col in passes.columns:
            passes[f"source_{col}"] = _safe_numeric(passes[col])
            passes[col] = _rescale_center_origin(
                passes[col],
                source_size=float(raw_pitch_length),
                target_size=PITCH_X,
            )
    for col in y_cols:
        if col in passes.columns:
            passes[f"source_{col}"] = _safe_numeric(passes[col])
            passes[col] = _rescale_center_origin(
                passes[col],
                source_size=float(raw_pitch_width),
                target_size=PITCH_Y,
            )

    passes["possession_start_x"] = _safe_numeric(passes.get("x_start", pd.Series(index=passes.index)))
    passes["possession_start_y"] = _safe_numeric(passes.get("y_start", pd.Series(index=passes.index)))
    passes["passer_x"] = _safe_numeric(passes.get("x_end", pd.Series(index=passes.index)))
    passes["passer_y"] = _safe_numeric(passes.get("y_end", pd.Series(index=passes.index)))
    passes["target_x"] = _safe_numeric(passes.get("player_targeted_x_pass", pd.Series(index=passes.index)))
    passes["target_y"] = _safe_numeric(passes.get("player_targeted_y_pass", pd.Series(index=passes.index)))
    passes["target_reception_x"] = _safe_numeric(
        passes.get("player_targeted_x_reception", pd.Series(index=passes.index))
    )
    passes["target_reception_y"] = _safe_numeric(
        passes.get("player_targeted_y_reception", pd.Series(index=passes.index))
    )
    passes["skillcorner_xpass"] = _safe_numeric(
        passes.get("player_targeted_xpass_completion", pd.Series(index=passes.index))
    )
    passes["has_skillcorner_xpass"] = passes["skillcorner_xpass"].notna()

    passes["match_id"] = context.match_id
    passes["period_name"] = passes["period"].map(CDF_PERIOD_MAP) if "period" in passes.columns else pd.NA
    passes["home_away"] = _home_away(passes["team_id"], metadata) if "team_id" in passes.columns else pd.NA
    passes["opponent_home_away"] = np.where(passes["home_away"].eq("home"), "away", "home")

    passes["competition_id"] = metadata.get("competition_id", pd.NA)
    passes["competition_name"] = metadata.get("competition_name", pd.NA)
    passes["competition_round_id"] = metadata.get("competition_round_id", pd.NA)
    passes["competition_round_name"] = metadata.get("competition_round_name", pd.NA)
    passes["competition_round_number"] = metadata.get("competition_round_number", pd.NA)
    passes["season_id"] = metadata.get("season_id", pd.NA)
    passes["season_name"] = metadata.get("season_name", pd.NA)
    passes["kickoff_time"] = metadata.get("kickoff_time", pd.NA)
    passes["home_team_id"] = metadata.get("home_team_id", pd.NA)
    passes["home_team_name"] = metadata.get("home_team_name", pd.NA)
    passes["away_team_id"] = metadata.get("away_team_id", pd.NA)
    passes["away_team_name"] = metadata.get("away_team_name", pd.NA)
    passes["final_home_score"] = metadata.get("final_home_score", pd.NA)
    passes["final_away_score"] = metadata.get("final_away_score", pd.NA)
    passes["final_score"] = metadata.get("final_score", pd.NA)
    passes["source_fps"] = metadata.get("source_fps", pd.NA)
    passes["raw_pitch_length"] = raw_pitch_length
    passes["raw_pitch_width"] = raw_pitch_width
    passes["match_path"] = str(context.match_path)
    passes["event_path"] = str(context.event_path)

    if index_row is not None:
        for col in ("date_time", "status", "match_dir"):
            if col in index_row.index:
                passes[f"index_{col}"] = index_row[col]

    passes["pass_completed"] = passes["pass_outcome"].eq("successful") if "pass_outcome" in passes.columns else False
    passes["pass_unsuccessful"] = passes["pass_outcome"].eq("unsuccessful") if "pass_outcome" in passes.columns else False
    passes["pass_offside"] = passes["pass_outcome"].eq("offside") if "pass_outcome" in passes.columns else False

    _add_set_piece_flags(passes)

    for col in [
        "lead_to_goal",
        "lead_to_shot",
        "one_touch",
        "quick_pass",
        "carry",
        "forward_momentum",
        "pass_ahead",
        "high_pass",
        "inside_defensive_shape_start",
        "inside_defensive_shape_end",
        "organised_defense",
        "is_player_possession_start_matched",
        "is_player_possession_end_matched",
        "is_previous_pass_matched",
        "is_pass_reception_matched",
        "fully_extrapolated",
    ]:
        if col in passes.columns:
            passes[f"{col}_bool"] = _safe_bool(passes[col])

    id_cols = [
        "match_id",
        "event_id",
        "team_id",
        "player_id",
        "player_targeted_id",
        "home_team_id",
        "away_team_id",
        "season_id",
        "competition_id",
        "competition_round_id",
        "targeted_passing_option_event_id",
    ]
    for col in id_cols:
        if col in passes.columns:
            passes[col] = passes[col].astype("string")

    return passes.reset_index(drop=True)


def build_skillcorner_pass_table(
    skillcorner_root: str | Path = DEFAULT_SKILLCORNER_ROOT,
    *,
    season_names: Iterable[str] | None = None,
    include_non_closed: bool = False,
    limit_matches: int | None = None,
    limit_matches_per_season: int | None = None,
) -> pd.DataFrame:
    """Build a pass-level table across SkillCorner matches."""
    index = load_match_index(
        skillcorner_root,
        season_names=season_names,
        include_non_closed=include_non_closed,
    )
    if limit_matches_per_season is not None:
        index = (
            index.groupby("season_name", group_keys=False)
            .head(int(limit_matches_per_season))
            .reset_index(drop=True)
        )
    if limit_matches is not None:
        index = index.head(int(limit_matches)).copy()

    tables: list[pd.DataFrame] = []
    for i, row in index.iterrows():
        context = resolve_match_context(
            skillcorner_root,
            row["match_id"],
            match_dir=row.get("match_dir"),
        )
        events = pd.read_csv(
            context.event_path,
            usecols=lambda col: col in PASS_SOURCE_COLUMNS,
            low_memory=False,
        )
        match_passes = _prepare_match_passes(events, context, index_row=row)
        if not match_passes.empty:
            tables.append(match_passes)

        if (i + 1) % 50 == 0 or i + 1 == len(index):
            print(f"[passes] processed {i + 1:,}/{len(index):,} matches; passes={sum(len(t) for t in tables):,}")

    if not tables:
        return pd.DataFrame()

    passes = pd.concat(tables, ignore_index=True, sort=False)
    sort_cols = [
        col
        for col in ["season_name", "kickoff_time", "match_id", "period", "frame_start", "event_id"]
        if col in passes.columns
    ]
    return passes.sort_values(sort_cols, kind="mergesort", ignore_index=True)


def write_pass_table(passes: pd.DataFrame, out_path: str | Path = DEFAULT_PASSES_PATH) -> Path:
    """Write the pass table and a compact JSON summary."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    passes.to_parquet(out, index=False)
    summary_path = out.with_suffix(".summary.json")
    summary = {
        "rows": int(len(passes)),
        "matches": int(passes["match_id"].nunique()) if "match_id" in passes else 0,
        "seasons": sorted(passes["season_name"].dropna().astype(str).unique().tolist())
        if "season_name" in passes
        else [],
        "completed_passes": int(passes["pass_completed"].sum()) if "pass_completed" in passes else 0,
        "skillcorner_xpass_coverage": float(passes["skillcorner_xpass"].notna().mean())
        if "skillcorner_xpass" in passes and len(passes)
        else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def summarize_pass_table(passes: pd.DataFrame) -> pd.DataFrame:
    """Return a season-level pass/xPass coverage summary."""
    if passes.empty:
        return pd.DataFrame()
    out = (
        passes.groupby("season_name", dropna=False)
        .agg(
            matches=("match_id", "nunique"),
            passes=("event_id", "size"),
            completed=("pass_completed", "sum"),
            skillcorner_xpass_rows=("skillcorner_xpass", lambda s: int(s.notna().sum())),
            avg_skillcorner_xpass=("skillcorner_xpass", "mean"),
        )
        .reset_index()
    )
    out["completion_rate"] = out["completed"] / out["passes"]
    out["skillcorner_xpass_coverage"] = out["skillcorner_xpass_rows"] / out["passes"]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SkillCorner pass table for xPass modelling.")
    parser.add_argument("--skillcorner-root", type=Path, default=DEFAULT_SKILLCORNER_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_PASSES_PATH)
    parser.add_argument("--season-names", nargs="*", default=None, help="Optional season filters, e.g. 2023 2024 2025")
    parser.add_argument("--include-non-closed", action="store_true")
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--limit-matches-per-season", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    passes = build_skillcorner_pass_table(
        args.skillcorner_root,
        season_names=args.season_names,
        include_non_closed=args.include_non_closed,
        limit_matches=args.limit_matches,
        limit_matches_per_season=args.limit_matches_per_season,
    )
    out = write_pass_table(passes, args.out)
    print(f"[passes] wrote {len(passes):,} rows to {out}")
    if not passes.empty:
        print(summarize_pass_table(passes).to_string(index=False))


if __name__ == "__main__":
    main()
