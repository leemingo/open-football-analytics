"""Build SkillCorner action tables for center-origin xT experiments."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from football_cdf.skillcorner_paths import META_FILENAMES, find_match_dir
from football_cdf.skillcorner_preprocessing import SkillcornerDataPreprocessor


SKILLCORNER_ROOT_ENV = "SKILLCORNER_ROOT"
DEFAULT_SKILLCORNER_ROOT = os.environ.get(SKILLCORNER_ROOT_ENV)
DEFAULT_CACHE_DIR = Path("tmp/data/cache")
DEFAULT_XPASS_PASSES_PATH = Path("tmp/data/skillcorner_xpass/passes.parquet")
DEFAULT_XG_SHOTS_PATH = Path("tmp/data/skillcorner_xg/shots.parquet")
DEFAULT_OUTPUT_DIR = Path("tmp/data/skillcorner_xthreat")
DEFAULT_ACTIONS_PATH = DEFAULT_OUTPUT_DIR / "actions.parquet"
DEFAULT_OPENDATA_ROOT = Path("tmp/data/skillcorner_opendata")
SKILLCORNER_OPENDATA_API_URL = "https://api.github.com/repos/SkillCorner/opendata/contents/data/matches?ref=master"
SKILLCORNER_OPENDATA_RAW_BASE_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches"

PITCH_X = 105.0
PITCH_Y = 68.0
CARRY_MIN_DISTANCE = 1.0

OPENDATA_SOURCE_COLUMNS = {
    "event_id",
    "index",
    "match_id",
    "frame_start",
    "frame_end",
    "time_start",
    "time_end",
    "minute_start",
    "second_start",
    "duration",
    "period",
    "attacking_side",
    "event_type",
    "event_subtype",
    "player_id",
    "player_name",
    "player_position",
    "team_id",
    "team_shortname",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "channel_start",
    "channel_end",
    "third_start",
    "third_end",
    "penalty_area_start",
    "penalty_area_end",
    "game_state",
    "team_score",
    "opponent_team_score",
    "phase_index",
    "player_possession_phase_index",
    "first_player_possession_in_team_possession",
    "last_player_possession_in_team_possession",
    "team_possession_loss_in_phase",
    "team_in_possession_phase_type",
    "team_out_of_possession_phase_type",
    "game_interruption_before",
    "game_interruption_after",
    "lead_to_shot",
    "lead_to_goal",
    "distance_covered",
    "trajectory_angle",
    "trajectory_direction",
    "speed_avg",
    "speed_avg_band",
    "start_type",
    "end_type",
    "one_touch",
    "quick_pass",
    "carry",
    "forward_momentum",
    "is_header",
    "pass_outcome",
    "targeted_passing_option_event_id",
    "high_pass",
    "player_targeted_id",
    "player_targeted_name",
    "player_targeted_position",
    "player_targeted_x_pass",
    "player_targeted_y_pass",
    "player_targeted_x_reception",
    "player_targeted_y_reception",
    "player_targeted_xpass_completion",
    "player_targeted_xthreat",
    "player_targeted_dangerous",
    "pass_distance",
    "pass_range",
    "pass_angle",
    "pass_direction",
    "n_passing_options",
    "n_off_ball_runs",
    "n_passing_options_ahead",
    "n_opponents_ahead_player_in_possession_pass_moment",
    "n_opponents_ahead_pass_reception",
    "n_opponents_bypassed",
    "organised_defense",
    "defensive_structure",
    "inside_defensive_shape_start",
    "inside_defensive_shape_end",
    "fully_extrapolated",
}

SET_PIECE_START_TYPES = {
    "corner",
    "corner_crossed",
    "corner_short",
    "free_kick",
    "free_kick_crossed",
    "free_kick_short",
    "goal_kick",
    "throw_in",
    "kick_off",
    "penalty",
}


@dataclass(frozen=True)
class ActionTableStats:
    n_rows: int
    n_matches: int
    n_pass: int
    n_carry: int
    n_shot: int
    n_successful_moves: int
    n_goals: int
    n_passes_with_skillcorner_xthreat: int
    n_shots_with_actual_goal: int


@dataclass(frozen=True)
class SkillCornerMatch:
    match_id: str
    match_dir: Path
    event_path: Path
    match_path: Path


def _event_files(cache_dir: str | Path) -> list[Path]:
    return sorted(Path(cache_dir).glob("*/events_aug.parquet"))


def _as_str_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_bool(series: pd.Series | None, *, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype="bool")
    if series.dtype == bool:
        return series.fillna(default).astype(bool)
    mapped = (
        series.astype("string")
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    return mapped.fillna(default).astype(bool)


def _resolve_skillcorner_matches_root(skillcorner_root: str | Path | None) -> Path:
    if skillcorner_root is None:
        raise ValueError(
            f"SkillCorner Dynamic Events root is required. Pass --skillcorner-root, "
            f"use --download-opendata, or set {SKILLCORNER_ROOT_ENV}."
        )
    root = Path(skillcorner_root)
    candidates = [root, root / "data" / "matches", root / "matches"]
    for candidate in candidates:
        if candidate.exists() and (
            any(candidate.glob("*/*_dynamic_events.csv"))
            or any(candidate.glob("*/dynamic_events.csv"))
            or (candidate / "matches_index.csv").exists()
        ):
            return candidate
    if root.exists():
        return root
    raise FileNotFoundError(f"SkillCorner Dynamic Events root does not exist: {root}")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _url_text(url: str) -> str:
    try:
        with urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8")
    except URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Could not download {url}: {exc}") from exc


def _url_bytes(url: str) -> bytes:
    try:
        with urlopen(url, timeout=60) as response:
            return response.read()
    except URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Could not download {url}: {exc}") from exc


def list_skillcorner_opendata_match_ids() -> list[str]:
    """Return match ids available in the public SkillCorner Open Data repo."""
    payload = json.loads(_url_text(SKILLCORNER_OPENDATA_API_URL))
    match_ids = [item["name"] for item in payload if item.get("type") == "dir"]
    return sorted(match_ids)


def download_skillcorner_opendata(
    out_dir: str | Path = DEFAULT_OPENDATA_ROOT,
    *,
    match_ids: Iterable[str | int] | None = None,
    force: bool = False,
) -> Path:
    """Download the SkillCorner Open Data files needed for xT.

    Only match metadata and Dynamic Events CSV files are required for this
    xThreat workflow, so tracking JSONL and phases files are intentionally not
    downloaded.
    """
    out = Path(out_dir)
    matches_root = out / "data" / "matches"
    matches_root.mkdir(parents=True, exist_ok=True)
    ids = [str(match_id) for match_id in match_ids] if match_ids is not None else list_skillcorner_opendata_match_ids()
    for match_id in ids:
        match_dir = matches_root / match_id
        match_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("dynamic_events.csv", "match.json"):
            filename = f"{match_id}_{suffix}"
            target = match_dir / filename
            if target.exists() and not force:
                continue
            url = f"{SKILLCORNER_OPENDATA_RAW_BASE_URL}/{match_id}/{filename}"
            target.write_bytes(_url_bytes(url))
    return out


def _first_existing_file(root: Path, names: Iterable[str]) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    joined = ", ".join(str(root / name) for name in names)
    raise FileNotFoundError(f"Missing required file. Tried: {joined}")


def _match_id_from_metadata(path: Path) -> str:
    try:
        metadata = _read_json(path)
    except json.JSONDecodeError:
        return path.parent.name
    return str(metadata.get("id") or path.parent.name.split("_")[-1] or path.parent.name)


def _discover_indexed_matches(root: Path, *, limit_matches: int | None = None) -> list[SkillCornerMatch]:
    index_path = root / "matches_index.csv"
    if not index_path.exists():
        return []
    index = pd.read_csv(index_path)
    if "match_id" not in index.columns:
        return []
    if "status" in index.columns:
        index = index[index["status"].astype("string").eq("closed")].copy()
    if limit_matches is not None:
        index = index.head(int(limit_matches)).copy()

    matches: list[SkillCornerMatch] = []
    for _, row in index.iterrows():
        match_id = str(row["match_id"])
        match_dir = Path(row["match_dir"]) if "match_dir" in row and pd.notna(row["match_dir"]) else None
        if match_dir is None or not match_dir.exists():
            match_dir = find_match_dir(root, match_id)
        if match_dir is None:
            continue
        event_path = match_dir / SkillcornerDataPreprocessor.EVENT_FILENAME
        if not event_path.exists():
            continue
        match_path = _first_existing_file(match_dir, META_FILENAMES)
        matches.append(
            SkillCornerMatch(
                match_id=match_id,
                match_dir=match_dir,
                event_path=event_path,
                match_path=match_path,
            )
        )
    return matches


def discover_opendata_matches(
    skillcorner_root: str | Path | None = DEFAULT_SKILLCORNER_ROOT,
    *,
    limit_matches: int | None = None,
) -> list[SkillCornerMatch]:
    """Find SkillCorner Dynamic Events match folders under a local root.

    Supports the public Open Data layout
    ``data/matches/{match_id}/{match_id}_dynamic_events.csv`` and the regular
    SkillCorner bundle layout ``{root}/{match_id}/dynamic_events.csv``. If a
    ``matches_index.csv`` exists, it is used as the primary match list.
    """
    root = _resolve_skillcorner_matches_root(skillcorner_root)
    matches = _discover_indexed_matches(root, limit_matches=limit_matches)
    if matches:
        return matches

    matches: list[SkillCornerMatch] = []
    for event_path in sorted(root.glob("*/*_dynamic_events.csv")):
        match_id = event_path.stem.replace("_dynamic_events", "")
        match_path = event_path.with_name(f"{match_id}_match.json")
        if not match_path.exists():
            continue
        matches.append(
            SkillCornerMatch(
                match_id=match_id,
                match_dir=event_path.parent,
                event_path=event_path,
                match_path=match_path,
            )
        )
    for event_path in sorted(root.glob("*/dynamic_events.csv")):
        match_dir = event_path.parent
        try:
            match_path = _first_existing_file(match_dir, META_FILENAMES)
        except FileNotFoundError:
            continue
        match_id = _match_id_from_metadata(match_path)
        matches.append(
            SkillCornerMatch(
                match_id=match_id,
                match_dir=match_dir,
                event_path=event_path,
                match_path=match_path,
            )
        )
    if limit_matches is not None:
        matches = matches[: int(limit_matches)]
    if not matches:
        raise FileNotFoundError(f"No SkillCorner Open Data matches found under {root}")
    return matches


def _metadata_from_match_json(match_id: str, match_metadata: dict) -> dict:
    try:
        extracted = SkillcornerDataPreprocessor.extract_match_metadata(match_metadata)
    except Exception:
        extracted = {}
    if extracted:
        return {
            "match_id": str(match_id),
            "competition_id": extracted.get("competition_id", pd.NA),
            "competition_name": extracted.get("competition_name", pd.NA),
            "competition_round_id": extracted.get("competition_round_id", pd.NA),
            "competition_round_name": extracted.get("competition_round_name", pd.NA),
            "competition_round_number": extracted.get("competition_round_number", pd.NA),
            "season_id": extracted.get("season_id", pd.NA),
            "season_name": extracted.get("season_name", pd.NA),
            "kickoff_time": extracted.get("kickoff_time", pd.NA),
            "home_team_id": extracted.get("home_team_id", pd.NA),
            "home_team_name": extracted.get("home_team_name", pd.NA),
            "home_team_shortname": extracted.get("home_team_shortname", extracted.get("home_team_name", pd.NA)),
            "away_team_id": extracted.get("away_team_id", pd.NA),
            "away_team_name": extracted.get("away_team_name", pd.NA),
            "away_team_shortname": extracted.get("away_team_shortname", extracted.get("away_team_name", pd.NA)),
            "final_home_score": extracted.get("final_home_score", pd.NA),
            "final_away_score": extracted.get("final_away_score", pd.NA),
            "final_score": extracted.get("final_score", pd.NA),
            "raw_pitch_length": float(extracted.get("pitch_length", PITCH_X) or PITCH_X),
            "raw_pitch_width": float(extracted.get("pitch_width", PITCH_Y) or PITCH_Y),
        }

    competition_edition = match_metadata.get("competition_edition") or {}
    competition = competition_edition.get("competition") or {}
    season = competition_edition.get("season") or {}
    round_info = match_metadata.get("competition_round") or {}
    home_team = match_metadata.get("home_team") or {}
    away_team = match_metadata.get("away_team") or {}
    final_home = match_metadata.get("home_team_score", pd.NA)
    final_away = match_metadata.get("away_team_score", pd.NA)
    final_score = f"{final_home}-{final_away}" if pd.notna(final_home) and pd.notna(final_away) else pd.NA
    return {
        "match_id": str(match_id),
        "competition_id": competition.get("id", pd.NA),
        "competition_name": competition.get("name", pd.NA),
        "competition_round_id": round_info.get("id", pd.NA),
        "competition_round_name": round_info.get("name", pd.NA),
        "competition_round_number": round_info.get("round_number", pd.NA),
        "season_id": season.get("id", pd.NA),
        "season_name": season.get("name", pd.NA),
        "kickoff_time": match_metadata.get("date_time", pd.NA),
        "home_team_id": home_team.get("id", pd.NA),
        "home_team_name": home_team.get("name", pd.NA),
        "home_team_shortname": home_team.get("short_name", pd.NA),
        "away_team_id": away_team.get("id", pd.NA),
        "away_team_name": away_team.get("name", pd.NA),
        "away_team_shortname": away_team.get("short_name", pd.NA),
        "final_home_score": final_home,
        "final_away_score": final_away,
        "final_score": final_score,
        "raw_pitch_length": float(match_metadata.get("pitch_length", PITCH_X) or PITCH_X),
        "raw_pitch_width": float(match_metadata.get("pitch_width", PITCH_Y) or PITCH_Y),
    }


def _home_away(team_id: pd.Series, metadata: dict) -> pd.Series:
    team = team_id.astype("string")
    home_team_id = str(metadata.get("home_team_id"))
    away_team_id = str(metadata.get("away_team_id"))
    return pd.Series(
        np.select([team.eq(home_team_id), team.eq(away_team_id)], ["home", "away"], default=pd.NA),
        index=team_id.index,
        dtype="string",
    )


def _rescale_center_origin(series: pd.Series, *, source_size: float, target_size: float) -> pd.Series:
    values = _safe_numeric(series)
    if pd.isna(source_size) or float(source_size) == 0.0:
        return values
    return values * (float(target_size) / float(source_size))


def _normalize_open_data_coordinates(events: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    out = events.copy()
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
    raw_pitch_length = float(metadata.get("raw_pitch_length", PITCH_X) or PITCH_X)
    raw_pitch_width = float(metadata.get("raw_pitch_width", PITCH_Y) or PITCH_Y)
    for col in x_cols:
        if col in out.columns:
            out[f"source_{col}"] = _safe_numeric(out[col])
            out[col] = _rescale_center_origin(out[col], source_size=raw_pitch_length, target_size=PITCH_X)
    for col in y_cols:
        if col in out.columns:
            out[f"source_{col}"] = _safe_numeric(out[col])
            out[col] = _rescale_center_origin(out[col], source_size=raw_pitch_width, target_size=PITCH_Y)

    return out


def _base_action_columns(events: pd.DataFrame, metadata: dict, match: SkillCornerMatch) -> pd.DataFrame:
    out = events.copy()
    out["match_id"] = str(match.match_id)
    out["home_away"] = _home_away(out["team_id"], metadata) if "team_id" in out.columns else pd.NA
    out["opponent_home_away"] = np.where(out["home_away"].eq("home"), "away", "home") if "home_away" in out.columns else pd.NA
    for key, value in metadata.items():
        if key not in out.columns:
            out[key] = value
    out["match_path"] = str(match.match_path)
    out["event_path"] = str(match.event_path)
    if "start_type" in out.columns:
        start_type = out["start_type"].astype("string")
        out["is_open_play"] = ~start_type.isin(SET_PIECE_START_TYPES)
    else:
        out["is_open_play"] = True
    return out


def _coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for col in columns:
        if col in frame.columns:
            values = _safe_numeric(frame[col])
            result = result.where(result.notna(), values)
    return result


def _build_pass_actions(possessions: pd.DataFrame) -> pd.DataFrame:
    passes = possessions[possessions["end_type"].astype("string").eq("pass")].copy()
    if passes.empty:
        return passes
    passes["parent_pp_id"] = passes["event_id"].astype("string")
    passes["event_id"] = passes["event_id"].astype("string") + ":pass"
    passes["action_type"] = "pass"
    passes["source_action_type"] = "pass"
    passes["start_x"] = _safe_numeric(passes["x_end"])
    passes["start_y"] = _safe_numeric(passes["y_end"])
    passes["end_x"] = _coalesce_numeric(passes, ["player_targeted_x_reception", "player_targeted_x_pass", "x_end"])
    passes["end_y"] = _coalesce_numeric(passes, ["player_targeted_y_reception", "player_targeted_y_pass", "y_end"])
    passes["outcome_flag"] = passes["pass_outcome"].astype("string").eq("successful") if "pass_outcome" in passes.columns else False
    passes["move_success"] = passes["outcome_flag"].fillna(False).astype(bool)
    passes["goal"] = False
    passes["skillcorner_target_xthreat"] = _coalesce_numeric(passes, ["player_targeted_xthreat"])
    passes["skillcorner_target_dangerous"] = passes.get("player_targeted_dangerous", pd.Series(pd.NA, index=passes.index))
    passes["skillcorner_xpass_completion"] = _coalesce_numeric(passes, ["player_targeted_xpass_completion"])
    passes["skillcorner_pass_outcome"] = passes.get("pass_outcome", pd.Series(pd.NA, index=passes.index))
    return passes


def _build_carry_actions(possessions: pd.DataFrame, *, min_distance: float = CARRY_MIN_DISTANCE) -> pd.DataFrame:
    start_x = _safe_numeric(possessions["x_start"])
    start_y = _safe_numeric(possessions["y_start"])
    end_x = _safe_numeric(possessions["x_end"])
    end_y = _safe_numeric(possessions["y_end"])
    displacement = np.hypot(end_x - start_x, end_y - start_y)
    if "carry" in possessions.columns:
        carry_flag = _safe_bool(possessions["carry"]).reindex(possessions.index, fill_value=False)
    else:
        carry_flag = pd.Series(False, index=possessions.index, dtype="bool")
    moved = (
        start_x.notna()
        & start_y.notna()
        & end_x.notna()
        & end_y.notna()
        & displacement.ge(float(min_distance))
    )
    carries = possessions[carry_flag | moved].copy()
    if carries.empty:
        return carries
    carries["parent_pp_id"] = carries["event_id"].astype("string")
    carries["event_id"] = carries["event_id"].astype("string") + ":carry"
    carries["action_type"] = "carry"
    carries["source_action_type"] = "carry"
    carries["start_x"] = start_x.reindex(carries.index)
    carries["start_y"] = start_y.reindex(carries.index)
    carries["end_x"] = end_x.reindex(carries.index)
    carries["end_y"] = end_y.reindex(carries.index)
    carries["outcome_flag"] = True
    carries["move_success"] = True
    carries["goal"] = False
    carries["skillcorner_target_xthreat"] = np.nan
    carries["skillcorner_target_dangerous"] = pd.NA
    carries["skillcorner_xpass_completion"] = np.nan
    carries["skillcorner_pass_outcome"] = pd.NA
    return carries


def _build_shot_actions(possessions: pd.DataFrame) -> pd.DataFrame:
    shots = possessions[possessions["end_type"].astype("string").eq("shot")].copy()
    if shots.empty:
        return shots
    shots["parent_pp_id"] = shots["event_id"].astype("string")
    shots["event_id"] = shots["event_id"].astype("string") + ":shot"
    shots["action_type"] = "shot"
    shots["source_action_type"] = "shot"
    shots["start_x"] = _safe_numeric(shots["x_end"])
    shots["start_y"] = _safe_numeric(shots["y_end"])
    shots["end_x"] = shots["start_x"]
    shots["end_y"] = shots["start_y"]
    shots["outcome_flag"] = False
    shots["move_success"] = False
    if "game_interruption_after" in shots.columns:
        shots["goal"] = shots["game_interruption_after"].astype("string").eq("goal_for")
    else:
        shots["goal"] = _safe_bool(shots.get("lead_to_goal", pd.Series(False, index=shots.index)))
    shots["shot_goal_actual"] = shots["goal"]
    shots["skillcorner_target_xthreat"] = np.nan
    shots["skillcorner_target_dangerous"] = pd.NA
    shots["skillcorner_xpass_completion"] = np.nan
    shots["skillcorner_pass_outcome"] = pd.NA
    return shots


def _prepare_opendata_match_actions(
    events: pd.DataFrame,
    match: SkillCornerMatch,
    metadata: dict,
    *,
    keep_only_open_play: bool = True,
    carry_min_distance: float = CARRY_MIN_DISTANCE,
) -> pd.DataFrame:
    possessions = events[events["event_type"].astype("string").eq("player_possession")].copy()
    if possessions.empty:
        return possessions
    possessions = _normalize_open_data_coordinates(possessions, metadata)
    possessions = _base_action_columns(possessions, metadata, match)
    if keep_only_open_play and "is_open_play" in possessions.columns:
        possessions = possessions[possessions["is_open_play"].fillna(False).astype(bool)].copy()
    parts = [
        _build_carry_actions(possessions, min_distance=carry_min_distance),
        _build_pass_actions(possessions),
        _build_shot_actions(possessions),
    ]
    actions = [part for part in parts if not part.empty]
    if not actions:
        return pd.DataFrame()
    return pd.concat(actions, ignore_index=True, sort=False)


def build_skillcorner_opendata_actions(
    skillcorner_root: str | Path | None = DEFAULT_SKILLCORNER_ROOT,
    *,
    limit_matches: int | None = None,
    keep_only_open_play: bool = True,
    keep_only_in_pitch: bool = True,
    carry_min_distance: float = CARRY_MIN_DISTANCE,
) -> tuple[pd.DataFrame, ActionTableStats]:
    """Build xT actions directly from SkillCorner Open Data Dynamic Events."""
    matches = discover_opendata_matches(skillcorner_root, limit_matches=limit_matches)
    tables: list[pd.DataFrame] = []
    for i, match in enumerate(matches):
        match_metadata = _read_json(match.match_path)
        metadata = _metadata_from_match_json(match.match_id, match_metadata)
        events = pd.read_csv(
            match.event_path,
            usecols=lambda col: col in OPENDATA_SOURCE_COLUMNS,
            low_memory=False,
        )
        match_actions = _prepare_opendata_match_actions(
            events,
            match,
            metadata,
            keep_only_open_play=keep_only_open_play,
            carry_min_distance=carry_min_distance,
        )
        if not match_actions.empty:
            tables.append(match_actions)
        if (i + 1) % 10 == 0 or i + 1 == len(matches):
            print(f"[xT/actions] processed {i + 1:,}/{len(matches):,} matches; actions={sum(len(t) for t in tables):,}")

    if not tables:
        return pd.DataFrame(), summarize_action_table(pd.DataFrame())

    actions = pd.concat(tables, ignore_index=True, sort=False)
    actions = prepare_xthreat_actions(
        actions,
        keep_only_open_play=False,
        keep_only_in_pitch=keep_only_in_pitch,
        map_drive_to_carry=True,
    )
    sort_cols = [
        col
        for col in ["season_name", "kickoff_time", "match_id", "period", "frame_start", "event_id"]
        if col in actions.columns
    ]
    if sort_cols:
        actions = actions.sort_values(sort_cols, kind="mergesort", ignore_index=True)
    return actions, summarize_action_table(actions)


def _read_event_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["cache_match_dir"] = path.parent.name
    return frame


def load_cached_epv_actions(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    limit_matches: int | None = None,
) -> pd.DataFrame:
    """Load existing SkillCorner pass/drive/shot action rows from cache."""
    paths = _event_files(cache_dir)
    if limit_matches is not None:
        paths = paths[: int(limit_matches)]
    if not paths:
        raise FileNotFoundError(f"No events_aug.parquet files found under {cache_dir}")
    frames = [_read_event_cache(path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def _metadata_from_passes(passes_path: str | Path) -> pd.DataFrame:
    path = Path(passes_path)
    if not path.exists():
        return pd.DataFrame(columns=["match_id"])
    passes = pd.read_parquet(path)
    cols = [c for c in ["match_id", "season_name", "competition_round_name", "home_team_shortname", "away_team_shortname"] if c in passes.columns]
    if not cols:
        return pd.DataFrame(columns=["match_id"])
    meta = passes[cols].drop_duplicates(subset=["match_id"]).copy()
    meta["match_id"] = _as_str_id(meta["match_id"])
    return meta


def _metadata_from_shots(shots_path: str | Path) -> pd.DataFrame:
    path = Path(shots_path)
    if not path.exists():
        return pd.DataFrame(columns=["match_id"])
    shots = pd.read_parquet(path)
    cols = [c for c in ["match_id", "season_name", "competition_round_name", "home_team_shortname", "away_team_shortname"] if c in shots.columns]
    if not cols:
        return pd.DataFrame(columns=["match_id"])
    meta = shots[cols].drop_duplicates(subset=["match_id"]).copy()
    meta["match_id"] = _as_str_id(meta["match_id"])
    return meta


def attach_match_metadata(
    actions: pd.DataFrame,
    *,
    passes_path: str | Path = DEFAULT_XPASS_PASSES_PATH,
    shots_path: str | Path = DEFAULT_XG_SHOTS_PATH,
) -> pd.DataFrame:
    """Attach season metadata from existing xPass/xG tables when available."""
    out = actions.copy()
    out["match_id"] = _as_str_id(out["match_id"])
    pieces = []
    for meta in [_metadata_from_passes(passes_path), _metadata_from_shots(shots_path)]:
        if not meta.empty:
            pieces.append(meta)
    if not pieces:
        return out
    meta = pd.concat(pieces, ignore_index=True, sort=False)
    meta = meta.drop_duplicates(subset=["match_id"], keep="first")
    overlap = [c for c in meta.columns if c in out.columns and c != "match_id"]
    if overlap:
        out = out.drop(columns=overlap)
    return out.merge(meta, on="match_id", how="left")


def attach_skillcorner_pass_xthreat(
    actions: pd.DataFrame,
    *,
    passes_path: str | Path = DEFAULT_XPASS_PASSES_PATH,
) -> pd.DataFrame:
    """Attach SkillCorner pass target xT to pass rows by match_id/PP event_id."""
    path = Path(passes_path)
    out = actions.copy()
    if not path.exists():
        out["skillcorner_target_xthreat"] = np.nan
        out["skillcorner_target_dangerous"] = pd.NA
        return out

    passes = pd.read_parquet(path)
    keep = [
        "match_id",
        "event_id",
        "player_targeted_xthreat",
        "player_targeted_dangerous",
        "player_targeted_xpass_completion",
        "pass_outcome",
    ]
    keep = [c for c in keep if c in passes.columns]
    pass_ref = passes[keep].copy()
    pass_ref["match_id"] = _as_str_id(pass_ref["match_id"])
    pass_ref["parent_pp_id"] = _as_str_id(pass_ref["event_id"])
    rename = {
        "player_targeted_xthreat": "skillcorner_target_xthreat",
        "player_targeted_dangerous": "skillcorner_target_dangerous",
        "player_targeted_xpass_completion": "skillcorner_xpass_completion",
        "pass_outcome": "skillcorner_pass_outcome",
    }
    pass_ref = pass_ref.rename(columns=rename).drop(columns=["event_id"], errors="ignore")
    out["match_id"] = _as_str_id(out["match_id"])
    out["parent_pp_id"] = _as_str_id(out["parent_pp_id"])
    return out.merge(pass_ref.drop_duplicates(["match_id", "parent_pp_id"]), on=["match_id", "parent_pp_id"], how="left")


def attach_actual_shot_goals(
    actions: pd.DataFrame,
    *,
    shots_path: str | Path = DEFAULT_XG_SHOTS_PATH,
) -> pd.DataFrame:
    """Attach actual shot goal labels from the xG shot table when available."""
    path = Path(shots_path)
    out = actions.copy()
    out["shot_goal_actual"] = pd.NA
    if not path.exists():
        return out
    shots = pd.read_parquet(path)
    keep = [c for c in ["match_id", "event_id", "goal"] if c in shots.columns]
    if len(keep) < 3:
        return out
    shot_ref = shots[keep].copy()
    shot_ref["match_id"] = _as_str_id(shot_ref["match_id"])
    shot_ref["parent_pp_id"] = _as_str_id(shot_ref["event_id"])
    shot_ref = shot_ref.rename(columns={"goal": "shot_goal_actual"}).drop(columns=["event_id"])
    out["match_id"] = _as_str_id(out["match_id"])
    out["parent_pp_id"] = _as_str_id(out["parent_pp_id"])
    out = out.drop(columns=["shot_goal_actual"], errors="ignore")
    return out.merge(shot_ref.drop_duplicates(["match_id", "parent_pp_id"]), on=["match_id", "parent_pp_id"], how="left")


def _within_pitch(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    x = pd.to_numeric(frame[x_col], errors="coerce")
    y = pd.to_numeric(frame[y_col], errors="coerce")
    return x.between(-PITCH_X / 2, PITCH_X / 2) & y.between(-PITCH_Y / 2, PITCH_Y / 2)


def prepare_xthreat_actions(
    actions: pd.DataFrame,
    *,
    keep_only_open_play: bool = True,
    keep_only_in_pitch: bool = True,
    map_drive_to_carry: bool = True,
) -> pd.DataFrame:
    """Normalize cached EPV actions into the xT action schema."""
    frame = actions.copy()
    if keep_only_open_play and "is_open_play" in frame.columns:
        frame = frame[frame["is_open_play"].fillna(False).astype(bool)].copy()
    frame = frame[frame["action_type"].isin(["pass", "drive", "carry", "shot"])].copy()

    frame["source_action_type"] = frame["action_type"].astype("string")
    if map_drive_to_carry:
        frame["action_type"] = frame["action_type"].replace({"drive": "carry"})

    rename = {
        "x_origin": "start_x",
        "y_origin": "start_y",
        "x_dest": "end_x",
        "y_dest": "end_y",
    }
    frame = frame.rename(columns=rename)
    for col in ["start_x", "start_y", "end_x", "end_y"]:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    is_pass = frame["action_type"].eq("pass")
    is_shot = frame["action_type"].eq("shot")
    for col in [
        "skillcorner_target_xthreat",
        "skillcorner_target_dangerous",
        "skillcorner_xpass_completion",
        "skillcorner_pass_outcome",
    ]:
        if col in frame.columns:
            frame.loc[~is_pass, col] = pd.NA
    if "shot_goal_actual" in frame.columns:
        frame.loc[~is_shot, "shot_goal_actual"] = pd.NA

    frame["move_success"] = np.where(frame["action_type"].isin(["pass", "carry"]), frame["outcome_flag"].fillna(False).astype(bool), False)
    if "shot_goal_actual" in frame.columns:
        actual = frame["shot_goal_actual"]
        fallback = frame.get("shot_goal", pd.Series(False, index=frame.index))
        frame["goal"] = np.where(is_shot, actual.where(actual.notna(), fallback).fillna(False).astype(bool), False)
    else:
        frame["goal"] = np.where(is_shot, frame.get("shot_goal", pd.Series(False, index=frame.index)).fillna(False).astype(bool), False)

    move_mask = frame["action_type"].isin(["pass", "carry"])
    shot_mask = is_shot
    required_xy = _within_pitch(frame, "start_x", "start_y")
    if keep_only_in_pitch:
        move_dest_ok = _within_pitch(frame, "end_x", "end_y")
        frame = frame[(shot_mask & required_xy) | (move_mask & required_xy & move_dest_ok)].copy()
    else:
        frame = frame[(shot_mask & required_xy) | move_mask].copy()

    preferred = [
        "event_id",
        "parent_pp_id",
        "match_id",
        "season_name",
        "period",
        "time_start",
        "time_end",
        "frame_event",
        "frame_start",
        "frame_end",
        "team_id",
        "team_shortname",
        "player_id",
        "player_name",
        "player_position",
        "action_type",
        "source_action_type",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "move_success",
        "goal",
        "outcome_flag",
        "skillcorner_target_xthreat",
        "skillcorner_target_dangerous",
        "skillcorner_xpass_completion",
        "skillcorner_pass_outcome",
        "team_in_possession_phase_type",
        "start_type",
        "end_type",
        "is_open_play",
        "distance_covered",
        "trajectory_angle",
        "is_header",
    ]
    rest = [c for c in frame.columns if c not in preferred]
    return frame[[c for c in preferred if c in frame.columns] + rest].reset_index(drop=True)


def build_skillcorner_xthreat_actions(
    skillcorner_root: str | Path | None = DEFAULT_SKILLCORNER_ROOT,
    *,
    source: str = "opendata",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    passes_path: str | Path = DEFAULT_XPASS_PASSES_PATH,
    shots_path: str | Path = DEFAULT_XG_SHOTS_PATH,
    limit_matches: int | None = None,
    keep_only_open_play: bool = True,
    keep_only_in_pitch: bool = True,
    carry_min_distance: float = CARRY_MIN_DISTANCE,
) -> tuple[pd.DataFrame, ActionTableStats]:
    """Build the canonical center-origin xT action table."""
    if source == "opendata":
        return build_skillcorner_opendata_actions(
            skillcorner_root,
            limit_matches=limit_matches,
            keep_only_open_play=keep_only_open_play,
            keep_only_in_pitch=keep_only_in_pitch,
            carry_min_distance=carry_min_distance,
        )
    if source != "cache":
        raise ValueError("source must be 'opendata' or 'cache'")
    actions = load_cached_epv_actions(cache_dir, limit_matches=limit_matches)
    actions = attach_match_metadata(actions, passes_path=passes_path, shots_path=shots_path)
    actions = attach_skillcorner_pass_xthreat(actions, passes_path=passes_path)
    actions = attach_actual_shot_goals(actions, shots_path=shots_path)
    actions = prepare_xthreat_actions(
        actions,
        keep_only_open_play=keep_only_open_play,
        keep_only_in_pitch=keep_only_in_pitch,
    )
    stats = summarize_action_table(actions)
    return actions, stats


def summarize_action_table(actions: pd.DataFrame) -> ActionTableStats:
    if actions.empty or "action_type" not in actions.columns:
        return ActionTableStats(
            n_rows=0,
            n_matches=0,
            n_pass=0,
            n_carry=0,
            n_shot=0,
            n_successful_moves=0,
            n_goals=0,
            n_passes_with_skillcorner_xthreat=0,
            n_shots_with_actual_goal=0,
        )
    action_type = actions["action_type"].astype("string")
    move_mask = action_type.isin(["pass", "carry"])
    goal = actions.get("goal", pd.Series(False, index=actions.index)).fillna(False).astype(bool)
    move_success = actions.get("move_success", pd.Series(False, index=actions.index)).fillna(False).astype(bool)
    return ActionTableStats(
        n_rows=int(len(actions)),
        n_matches=int(actions["match_id"].astype("string").nunique()) if "match_id" in actions.columns else 0,
        n_pass=int(action_type.eq("pass").sum()),
        n_carry=int(action_type.eq("carry").sum()),
        n_shot=int(action_type.eq("shot").sum()),
        n_successful_moves=int((move_mask & move_success).sum()),
        n_goals=int(goal.sum()),
        n_passes_with_skillcorner_xthreat=int(actions.get("skillcorner_target_xthreat", pd.Series(index=actions.index, dtype=float)).notna().sum()),
        n_shots_with_actual_goal=int(actions.get("shot_goal_actual", pd.Series(index=actions.index, dtype=object)).notna().sum()),
    )


def write_actions(
    actions: pd.DataFrame,
    stats: ActionTableStats,
    *,
    output_path: str | Path = DEFAULT_ACTIONS_PATH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actions.to_parquet(output_path, index=False)
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(stats), handle, indent=2, ensure_ascii=False)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SkillCorner center-origin xT action table.")
    parser.add_argument(
        "--skillcorner-root",
        type=Path,
        default=DEFAULT_SKILLCORNER_ROOT,
        help=f"SkillCorner Open Data or local match-bundle root. Can also be set via {SKILLCORNER_ROOT_ENV}.",
    )
    parser.add_argument(
        "--source",
        choices=["opendata", "cache"],
        default="opendata",
        help="Input source. 'opendata' reads SkillCorner Open Data Dynamic Events; 'cache' reads legacy events_aug parquet files.",
    )
    parser.add_argument(
        "--download-opendata",
        action="store_true",
        help=f"Download SkillCorner Open Data match metadata and Dynamic Events into {DEFAULT_OPENDATA_ROOT}.",
    )
    parser.add_argument("--opendata-out-dir", type=Path, default=DEFAULT_OPENDATA_ROOT)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--passes-path", type=Path, default=DEFAULT_XPASS_PASSES_PATH)
    parser.add_argument("--shots-path", type=Path, default=DEFAULT_XG_SHOTS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_ACTIONS_PATH)
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--include-set-pieces", action="store_true")
    parser.add_argument("--include-out-of-pitch", action="store_true")
    parser.add_argument("--carry-min-distance", type=float, default=CARRY_MIN_DISTANCE)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    skillcorner_root = args.skillcorner_root
    if args.download_opendata:
        match_ids = None
        if args.limit_matches is not None:
            match_ids = list_skillcorner_opendata_match_ids()[: int(args.limit_matches)]
        skillcorner_root = download_skillcorner_opendata(
            args.opendata_out_dir,
            match_ids=match_ids,
            force=args.force_download,
        )
    actions, stats = build_skillcorner_xthreat_actions(
        skillcorner_root,
        source=args.source,
        cache_dir=args.cache_dir,
        passes_path=args.passes_path,
        shots_path=args.shots_path,
        limit_matches=args.limit_matches,
        keep_only_open_play=not args.include_set_pieces,
        keep_only_in_pitch=not args.include_out_of_pitch,
        carry_min_distance=args.carry_min_distance,
    )
    path = write_actions(actions, stats, output_path=args.out)
    print(f"[xT/actions] wrote {len(actions):,} rows to {path}")
    print(json.dumps(asdict(stats), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
