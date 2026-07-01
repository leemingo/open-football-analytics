"""Build a shot-level table from SkillCorner Dynamic Events.

The implementation follows the preprocessing choices explored in
``notebooks/validate_skillcorner.ipynb``:

* use ``SkillcornerDataPreprocessor`` metadata parsing/path conventions;
* keep SkillCorner Dynamic Event coordinates in their native attacker-left-to-
  right, centre-origin convention;
* rescale coordinates to the common 105 x 68 metre pitch used elsewhere in the
  project;
* treat ``player_possession`` rows whose ``end_type`` is ``shot`` as shots.

The resulting table is the canonical input for feature engineering and xG
model training.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from football_cdf.constants import CDF_PERIOD_MAP, PITCH_X, PITCH_Y
from football_cdf.skillcorner_paths import META_FILENAMES, find_match_dir
from football_cdf.skillcorner_preprocessing import SkillcornerDataPreprocessor


SKILLCORNER_ROOT_ENV = "SKILLCORNER_ROOT"
DEFAULT_SKILLCORNER_ROOT = os.environ.get(SKILLCORNER_ROOT_ENV)
DEFAULT_OUTPUT_DIR = Path("tmp/data/skillcorner_xg")
DEFAULT_SHOTS_PATH = DEFAULT_OUTPUT_DIR / "shots.parquet"


SHOT_SOURCE_COLUMNS = [
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
    "n_player_targeted_opponents_ahead_start",
    "n_player_targeted_opponents_ahead_end",
    "n_player_targeted_teammates_ahead_start",
    "n_player_targeted_teammates_ahead_end",
    "n_player_targeted_teammates_within_5m_start",
    "n_player_targeted_teammates_within_5m_end",
    "n_player_targeted_opponents_within_5m_start",
    "n_player_targeted_opponents_within_5m_end",
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


def _resolve_skillcorner_root(skillcorner_root: str | Path | None) -> Path:
    if skillcorner_root is None:
        raise ValueError(
            f"SkillCorner data root is required. Pass --skillcorner-root or set {SKILLCORNER_ROOT_ENV}."
        )
    return Path(skillcorner_root)


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
    skillcorner_root: str | Path | None,
    match_id: str | int,
    *,
    match_dir: str | Path | None = None,
) -> MatchContext:
    """Resolve match paths and metadata using the project SkillCorner parser."""
    root = _resolve_skillcorner_root(skillcorner_root)
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
    skillcorner_root: str | Path | None = DEFAULT_SKILLCORNER_ROOT,
    *,
    season_names: Iterable[str] | None = None,
    include_non_closed: bool = False,
) -> pd.DataFrame:
    """Load the SkillCorner ``matches_index.csv`` with optional filtering."""
    root = _resolve_skillcorner_root(skillcorner_root)
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

    return index.sort_values(["season_name", "date_time", "match_id"], kind="mergesort").reset_index(drop=True)


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


def _prepare_match_shots(
    events: pd.DataFrame,
    context: MatchContext,
    *,
    index_row: pd.Series | None = None,
) -> pd.DataFrame:
    shots = events[
        events["event_type"].eq("player_possession")
        & events["end_type"].eq("shot")
    ].copy()
    if shots.empty:
        return shots

    metadata = context.match_metadata
    raw_pitch_length = pd.to_numeric(metadata.get("pitch_length"), errors="coerce")
    raw_pitch_width = pd.to_numeric(metadata.get("pitch_width"), errors="coerce")
    if pd.isna(raw_pitch_length):
        raw_pitch_length = PITCH_X
    if pd.isna(raw_pitch_width):
        raw_pitch_width = PITCH_Y

    for col in ("x_start", "x_end"):
        if col in shots.columns:
            shots[f"source_{col}"] = _safe_numeric(shots[col])
            shots[col] = _rescale_center_origin(
                shots[col],
                source_size=float(raw_pitch_length),
                target_size=PITCH_X,
            )
    for col in ("y_start", "y_end"):
        if col in shots.columns:
            shots[f"source_{col}"] = _safe_numeric(shots[col])
            shots[col] = _rescale_center_origin(
                shots[col],
                source_size=float(raw_pitch_width),
                target_size=PITCH_Y,
            )

    shots["shot_x"] = _safe_numeric(shots["x_end"])
    shots["shot_y"] = _safe_numeric(shots["y_end"])
    shots["possession_start_x"] = _safe_numeric(shots["x_start"])
    shots["possession_start_y"] = _safe_numeric(shots["y_start"])
    shots["shot_display_x"] = shots["shot_x"] + PITCH_X / 2.0
    shots["shot_display_y"] = shots["shot_y"] + PITCH_Y / 2.0

    shots["match_id"] = context.match_id
    shots["period_name"] = shots["period"].map(CDF_PERIOD_MAP)
    shots["home_away"] = _home_away(shots["team_id"], metadata)
    shots["opponent_home_away"] = np.where(shots["home_away"].eq("home"), "away", "home")

    shots["competition_id"] = metadata.get("competition_id", pd.NA)
    shots["competition_name"] = metadata.get("competition_name", pd.NA)
    shots["competition_round_id"] = metadata.get("competition_round_id", pd.NA)
    shots["competition_round_name"] = metadata.get("competition_round_name", pd.NA)
    shots["competition_round_number"] = metadata.get("competition_round_number", pd.NA)
    shots["season_id"] = metadata.get("season_id", pd.NA)
    shots["season_name"] = metadata.get("season_name", pd.NA)
    shots["kickoff_time"] = metadata.get("kickoff_time", pd.NA)
    shots["home_team_id"] = metadata.get("home_team_id", pd.NA)
    shots["home_team_name"] = metadata.get("home_team_name", pd.NA)
    shots["away_team_id"] = metadata.get("away_team_id", pd.NA)
    shots["away_team_name"] = metadata.get("away_team_name", pd.NA)
    shots["final_home_score"] = metadata.get("final_home_score", pd.NA)
    shots["final_away_score"] = metadata.get("final_away_score", pd.NA)
    shots["final_score"] = metadata.get("final_score", pd.NA)
    shots["source_fps"] = metadata.get("source_fps", pd.NA)
    shots["raw_pitch_length"] = raw_pitch_length
    shots["raw_pitch_width"] = raw_pitch_width
    shots["match_path"] = str(context.match_path)
    shots["event_path"] = str(context.event_path)

    if index_row is not None:
        for col in ("date_time", "status", "match_dir"):
            if col in index_row.index:
                shots[f"index_{col}"] = index_row[col]

    if "game_interruption_after" in shots.columns:
        shots["goal"] = shots["game_interruption_after"].eq("goal_for")
    else:
        shots["goal"] = False

    if "lead_to_goal" in shots.columns:
        shots["lead_to_goal_bool"] = _safe_bool(shots["lead_to_goal"])
    if "lead_to_shot" in shots.columns:
        shots["lead_to_shot_bool"] = _safe_bool(shots["lead_to_shot"])

    if "start_type" in shots.columns:
        start_type = shots["start_type"].astype("string")
        shots["is_set_piece_start"] = start_type.isin(SET_PIECE_START_TYPES)
        shots["is_corner_start"] = start_type.str.startswith("corner", na=False)
        shots["is_free_kick_start"] = start_type.str.startswith("free_kick", na=False)
        shots["is_throw_in_start"] = start_type.str.startswith("throw_in", na=False)
        shots["is_goal_kick_start"] = start_type.str.startswith("goal_kick", na=False)
    else:
        shots["is_set_piece_start"] = False
        shots["is_corner_start"] = False
        shots["is_free_kick_start"] = False
        shots["is_throw_in_start"] = False
        shots["is_goal_kick_start"] = False

    id_cols = [
        "match_id",
        "event_id",
        "team_id",
        "player_id",
        "home_team_id",
        "away_team_id",
        "season_id",
        "competition_id",
        "competition_round_id",
    ]
    for col in id_cols:
        if col in shots.columns:
            shots[col] = shots[col].astype("string")

    return shots.reset_index(drop=True)


def build_skillcorner_shot_table(
    skillcorner_root: str | Path | None = DEFAULT_SKILLCORNER_ROOT,
    *,
    season_names: Iterable[str] | None = None,
    include_non_closed: bool = False,
    limit_matches: int | None = None,
    limit_matches_per_season: int | None = None,
) -> pd.DataFrame:
    """Build a shot-level table across SkillCorner matches."""
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
            usecols=lambda col: col in SHOT_SOURCE_COLUMNS,
            low_memory=False,
        )
        match_shots = _prepare_match_shots(events, context, index_row=row)
        if not match_shots.empty:
            tables.append(match_shots)

        if (i + 1) % 50 == 0 or i + 1 == len(index):
            print(f"[shots] processed {i + 1:,}/{len(index):,} matches; shots={sum(len(t) for t in tables):,}")

    if not tables:
        return pd.DataFrame()

    shots = pd.concat(tables, ignore_index=True, sort=False)
    shots = shots.sort_values(
        ["season_name", "kickoff_time", "match_id", "period", "frame_start", "event_id"],
        kind="mergesort",
        ignore_index=True,
    )
    return shots


def write_shot_table(shots: pd.DataFrame, out_path: str | Path = DEFAULT_SHOTS_PATH) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(out, index=False)
    summary_path = out.with_suffix(".summary.json")
    summary = {
        "rows": int(len(shots)),
        "matches": int(shots["match_id"].nunique()) if "match_id" in shots else 0,
        "seasons": sorted(shots["season_name"].dropna().astype(str).unique().tolist())
        if "season_name" in shots
        else [],
        "goals": int(shots["goal"].sum()) if "goal" in shots else 0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SkillCorner shot table for xG modelling.")
    parser.add_argument(
        "--skillcorner-root",
        type=Path,
        default=DEFAULT_SKILLCORNER_ROOT,
        required=DEFAULT_SKILLCORNER_ROOT is None,
        help=f"Directory containing SkillCorner match bundles. Can also be set via {SKILLCORNER_ROOT_ENV}.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_SHOTS_PATH)
    parser.add_argument("--season-names", nargs="*", default=None, help="Optional season filters, e.g. 2023 2024 2025")
    parser.add_argument("--include-non-closed", action="store_true")
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--limit-matches-per-season", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    shots = build_skillcorner_shot_table(
        args.skillcorner_root,
        season_names=args.season_names,
        include_non_closed=args.include_non_closed,
        limit_matches=args.limit_matches,
        limit_matches_per_season=args.limit_matches_per_season,
    )
    out = write_shot_table(shots, args.out)
    print(f"[shots] wrote {len(shots):,} rows to {out}")
    if not shots.empty:
        by_season = shots.groupby("season_name").agg(
            matches=("match_id", "nunique"),
            shots=("event_id", "size"),
            goals=("goal", "sum"),
        )
        print(by_season.to_string())


if __name__ == "__main__":
    main()
