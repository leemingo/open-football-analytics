"""Build Bepro action tables for expected Threat analysis.

The active Bepro path consumes the canonical SPADL store produced by
``pipelines/bepro_ingest.py``. That store is created from Google Drive Bepro data
through the shared football-cdf Bepro v2 preprocessor, then metric builders
attack-normalize coordinates when they consume it:

    x in [-52.5, 52.5], y in [-34, 34], attacking left-to-right.

Legacy pass/shot-cache conversion helpers remain in this module for comparison,
but ``build_bepro_xthreat_actions`` now defaults to the SPADL store so xT shares
the same data lineage as xG and xPass.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from football_cdf.constants import PITCH_X, PITCH_Y
except ModuleNotFoundError:
    PITCH_X = 105.0
    PITCH_Y = 68.0
from xg.bepro_drive_shots import (
    DEFAULT_EVENT_REMOTE,
    DEFAULT_LEAGUES,
    DEFAULT_RCLONE,
    DEFAULT_RCLONE_CONFIG,
    DEFAULT_SPADL_DIR,
    SHOT_TYPES,
    DEFAULT_TEST_SEASONS,
    DEFAULT_TRAIN_SEASONS,
    _normalize_id_series,
    _split_season_labels,
)
from xg.bepro_drive_players import DEFAULT_OUTPUT as DEFAULT_PLAYER_LOOKUP_PATH
from xpass.bepro_drive_passes import CROSS_TYPES, PASS_TYPES


DEFAULT_PASSES_PATH = Path("tmp/data/bepro_drive_xpass_k1/passes.parquet")
DEFAULT_SHOTS_PATH = Path("tmp/data/bepro_drive_xg_k1/shots.parquet")
DEFAULT_OUTPUT_DIR = Path("tmp/data/bepro_drive_xthreat_k1")
DEFAULT_ACTIONS_PATH = DEFAULT_OUTPUT_DIR / "actions.parquet"

CARRY_TYPES = ["dribble"]
MOVE_TYPES = [*PASS_TYPES, *CARRY_TYPES]
SPADL_TYPE_ID_FALLBACK = {
    20: "dribble",
}
SPADL_RESULT_ID_FALLBACK = {
    0: "fail",
    1: "success",
    2: "offside",
    3: "owngoal",
}


@dataclass(frozen=True)
class BeproXThreatActionStats:
    n_rows: int
    n_matches: int
    n_pass: int
    n_carry: int
    n_shot: int
    n_successful_moves: int
    n_goals: int
    n_open_play: int
    n_set_piece: int
    n_phases: int


def _series(frame: pd.DataFrame, column: str, default=pd.NA) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _numeric(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    return pd.to_numeric(_series(frame, column, default), errors="coerce")


def _bool(frame: pd.DataFrame, column: str, default=False) -> pd.Series:
    return _series(frame, column, default).fillna(default).astype(bool)


def _within_pitch(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    x = pd.to_numeric(frame[x_col], errors="coerce")
    y = pd.to_numeric(frame[y_col], errors="coerce")
    return x.between(-PITCH_X / 2, PITCH_X / 2) & y.between(-PITCH_Y / 2, PITCH_Y / 2)


def _normalized_spadl_type_name(frame: pd.DataFrame) -> pd.Series:
    type_name = _series(frame, "type_name", pd.NA).astype("string")
    if "type_id" not in frame.columns:
        return type_name
    fallback = pd.to_numeric(frame["type_id"], errors="coerce").map(SPADL_TYPE_ID_FALLBACK).astype("string")
    return type_name.fillna(fallback)


def _normalized_result_name(frame: pd.DataFrame) -> pd.Series:
    result_name = _series(frame, "result_name", pd.NA).astype("string")
    if "result_id" not in frame.columns:
        return result_name
    fallback = pd.to_numeric(frame["result_id"], errors="coerce").map(SPADL_RESULT_ID_FALLBACK).astype("string")
    return result_name.fillna(fallback)


def _mode_string(series: pd.Series) -> str | pd.NA:
    values = series.dropna().astype("string")
    values = values[values.ne("")]
    if values.empty:
        return pd.NA
    mode = values.mode()
    return mode.iloc[0] if not mode.empty else values.iloc[-1]


def _load_xpass_player_lookup(path: str | Path = DEFAULT_PASSES_PATH) -> pd.DataFrame:
    """Build a compact player lookup from the xPass pass table if available."""
    path = Path(path)
    columns = ["source_season_name", "team_id", "player_id", "player_name", "player_position"]
    empty = pd.DataFrame(columns=["source_season_name", "team_id", "player_id", "player_full_name", "position_name"])
    if not path.exists():
        return empty
    try:
        lookup = pd.read_parquet(path, columns=columns)
    except (FileNotFoundError, KeyError, ValueError):
        return empty
    for col in ["source_season_name", "team_id", "player_id"]:
        lookup[col] = lookup[col].astype("string")
    lookup["player_full_name"] = lookup["player_name"].astype("string")
    lookup["position_name"] = lookup["player_position"].astype("string")
    lookup = lookup.dropna(subset=["source_season_name", "team_id", "player_id", "player_full_name"])
    lookup = (
        lookup.groupby(["source_season_name", "team_id", "player_id"], dropna=False)
        .agg(player_full_name=("player_full_name", _mode_string), position_name=("position_name", _mode_string))
        .reset_index()
    )
    return lookup[["source_season_name", "team_id", "player_id", "player_full_name", "position_name"]]


def _load_player_lookup(path: str | Path | None) -> pd.DataFrame:
    lookups: list[pd.DataFrame] = [_load_xpass_player_lookup()]
    if path is not None and Path(path).exists():
        lookup = pd.read_parquet(path).copy()
        for col in ["source_season_name", "team_id", "player_id"]:
            if col in lookup.columns:
                lookup[col] = lookup[col].astype("string")
        keep = [c for c in ["source_season_name", "team_id", "player_id", "player_full_name", "position_name"] if c in lookup.columns]
        lookup = lookup[keep].copy()
        lookups.append(lookup)
    lookups = [lookup for lookup in lookups if not lookup.empty]
    if not lookups:
        return pd.DataFrame(columns=["source_season_name", "team_id", "player_id", "player_full_name", "position_name"])
    lookup = pd.concat(lookups, ignore_index=True, sort=False)
    for col in ["source_season_name", "team_id", "player_id", "player_full_name", "position_name"]:
        if col not in lookup.columns:
            lookup[col] = pd.NA
        lookup[col] = lookup[col].astype("string")
    lookup = lookup.dropna(subset=["source_season_name", "team_id", "player_id"])
    return lookup.drop_duplicates(["source_season_name", "team_id", "player_id"], keep="last")


def _attach_player_lookup(frame: pd.DataFrame, player_lookup: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if player_lookup.empty:
        if "player_name" not in out.columns:
            out["player_name"] = "ID " + out["player_id"].astype("string")
        else:
            out["player_name"] = out["player_name"].astype("string").fillna("ID " + out["player_id"].astype("string"))
        if "player_position" not in out.columns:
            out["player_position"] = "missing"
        else:
            out["player_position"] = out["player_position"].astype("string").fillna("missing")
        return out

    keys = ["source_season_name", "team_id", "player_id"]
    for col in keys:
        if col in out.columns:
            out[col] = out[col].astype("string")
    out = out.merge(player_lookup, on=keys, how="left", suffixes=("", "_lookup"))
    if "player_name" not in out.columns:
        out["player_name"] = pd.NA
    out["player_name"] = out["player_name"].astype("string")
    out["player_name"] = out["player_name"].fillna(out.get("player_full_name", pd.Series(pd.NA, index=out.index)).astype("string"))
    if "player_position" not in out.columns:
        out["player_position"] = pd.NA
    if "position_name" in out.columns:
        out["player_position"] = out["player_position"].astype("string").fillna(out["position_name"].astype("string"))
    still_missing = out["player_name"].isna()
    if still_missing.any():
        names_per_id = (
            player_lookup.dropna(subset=["player_id", "player_full_name"])
            .groupby("player_id")["player_full_name"]
            .nunique()
        )
        unique_ids = names_per_id[names_per_id.eq(1)].index
        global_lookup = player_lookup[player_lookup["player_id"].isin(unique_ids)].copy()
        if not global_lookup.empty:
            global_lookup = global_lookup.drop_duplicates("player_id", keep="last")
            global_lookup = global_lookup[["player_id", "player_full_name", "position_name"]].rename(
                columns={"player_full_name": "player_full_name_global", "position_name": "position_name_global"}
            )
            out = out.merge(global_lookup, on="player_id", how="left")
            out["player_name"] = out["player_name"].fillna(out["player_full_name_global"].astype("string"))
            out["player_position"] = out["player_position"].astype("string").fillna(out["position_name_global"].astype("string"))
            out = out.drop(columns=["player_full_name_global", "position_name_global"])
    out["player_name"] = out["player_name"].fillna("ID " + out["player_id"].astype("string"))
    out["player_position"] = out["player_position"].astype("string").fillna("missing")
    return out


def _game_state(team_score: pd.Series, opponent_score: pd.Series) -> pd.Series:
    diff = pd.to_numeric(team_score, errors="coerce") - pd.to_numeric(opponent_score, errors="coerce")
    out = pd.Series("missing", index=team_score.index, dtype="string")
    out.loc[diff.gt(0).fillna(False)] = "winning"
    out.loc[diff.eq(0).fillna(False)] = "drawing"
    out.loc[diff.lt(0).fillna(False)] = "losing"
    return out


def _attack_normalized_xy(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    is_home: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Convert CDF home-left coordinates to attacker-left-to-right coordinates."""
    x = _numeric(frame, x_col)
    y = _numeric(frame, y_col)
    home = is_home.fillna(False).astype(bool).to_numpy()
    return (
        pd.Series(np.where(home, x, -x), index=frame.index, dtype="float64"),
        pd.Series(np.where(home, -y, y), index=frame.index, dtype="float64"),
    )


def _set_piece_flags(frame: pd.DataFrame) -> pd.DataFrame:
    type_name = _series(frame, "type_name", "").astype("string")
    set_piece = _series(frame, "set_piece_type", pd.NA).astype("string")
    set_piece_known = set_piece.notna() & ~set_piece.eq("<NA>") & ~set_piece.str.lower().eq("nan")

    out = pd.DataFrame(index=frame.index)
    out["is_corner_start"] = type_name.isin(["corner_short", "corner_crossed"]) | set_piece.str.contains("Corner", case=False, na=False)
    out["is_free_kick_start"] = (
        type_name.isin(["freekick_short", "freekick_crossed", "shot_freekick"])
        | set_piece.str.contains("Free", case=False, na=False)
    )
    out["is_throw_in_start"] = type_name.eq("throw_in") | set_piece.str.contains("Throw", case=False, na=False)
    out["is_goal_kick_start"] = type_name.eq("goalkick") | set_piece.str.contains("Goal Kick", case=False, na=False)
    out["is_penalty_start"] = type_name.eq("shot_penalty") | set_piece.str.contains("Penalty", case=False, na=False)
    out["is_set_piece_start"] = out.any(axis=1) | set_piece_known
    return out


def _open_play_mask(frame: pd.DataFrame, *, source: str) -> pd.Series:
    set_piece = _bool(frame, "is_set_piece_start")
    if source == "shots":
        spadl = _series(frame, "spadl_type", "").astype("string")
        set_piece = set_piece | spadl.isin(["shot_penalty", "shot_freekick"])
    return ~set_piece


def passes_to_xthreat_actions(passes: pd.DataFrame) -> pd.DataFrame:
    """Convert a Bepro xPass pass table into xT move actions."""
    frame = passes.copy()
    for col in ["match_id", "event_id", "team_id", "player_id"]:
        if col in frame.columns:
            frame[col] = _normalize_id_series(frame[col])

    start_x = _numeric(frame, "passer_x")
    start_y = _numeric(frame, "passer_y")
    end_x = _numeric(frame, "target_x")
    end_y = _numeric(frame, "target_y")
    dx = end_x - start_x
    dy = end_y - start_y
    completed = _bool(frame, "pass_completed")
    is_cross = _bool(frame, "is_cross")
    is_open_play = _open_play_mask(frame, source="passes")
    time_seconds = _numeric(frame, "time_seconds")

    out = pd.DataFrame(
        {
            "event_id": frame["event_id"].astype("string"),
            "parent_pp_id": frame["event_id"].astype("string"),
            "match_id": frame["match_id"].astype("string"),
            "season_name": _series(frame, "season_name").astype("string"),
            "source_season_name": _series(frame, "source_season_name").astype("string"),
            "league": _series(frame, "league").astype("string"),
            "competition_round_name": _series(frame, "competition_round_name").astype("string"),
            "kickoff_time": _series(frame, "kickoff_time"),
            "period": _numeric(frame, "period_id").astype("Int64"),
            "period_id": _numeric(frame, "period_id").astype("Int64"),
            "period_name": _series(frame, "period_name").astype("string"),
            "event_time": _numeric(frame, "event_time"),
            "time_start": time_seconds,
            "time_end": time_seconds,
            "frame_event": np.rint(time_seconds * 10.0),
            "frame_start": np.rint(time_seconds * 10.0),
            "frame_end": np.rint(time_seconds * 10.0),
            "team_id": frame["team_id"].astype("string"),
            "team_name": _series(frame, "team_name").astype("string"),
            "team_shortname": _series(frame, "team_shortname").astype("string"),
            "player_id": frame["player_id"].astype("string"),
            "player_name": _series(frame, "player_name").astype("string").fillna("ID " + frame["player_id"].astype("string")),
            "player_position": _series(frame, "player_position", "missing").astype("string").fillna("missing"),
            "home_away": _series(frame, "home_away").astype("string"),
            "action_type": "pass",
            "source_action_type": np.where(is_cross, "cross", "pass"),
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "move_success": completed,
            "goal": False,
            "outcome_flag": completed,
            "is_open_play": is_open_play,
            "is_set_piece_start": ~is_open_play,
            "is_corner_start": _bool(frame, "is_corner_start"),
            "is_free_kick_start": _bool(frame, "is_free_kick_start"),
            "is_throw_in_start": _bool(frame, "is_throw_in_start"),
            "is_goal_kick_start": _bool(frame, "is_goal_kick_start"),
            "pass_outcome": _series(frame, "pass_outcome").astype("string"),
            "is_cross": is_cross,
            "is_key_pass": _bool(frame, "is_key_pass"),
            "is_assist": _bool(frame, "is_assist"),
            "start_type": _series(frame, "set_piece_sub_type", "open_play").astype("string").fillna("open_play"),
            "end_type": "target",
            "team_in_possession_phase_type": _series(frame, "game_state", "missing").astype("string"),
            "home_team_id": _series(frame, "home_team_id").astype("string"),
            "home_team_name": _series(frame, "home_team_name").astype("string"),
            "away_team_id": _series(frame, "away_team_id").astype("string"),
            "away_team_name": _series(frame, "away_team_name").astype("string"),
            "final_home_score": _series(frame, "final_home_score"),
            "final_away_score": _series(frame, "final_away_score"),
            "distance_covered": np.hypot(dx, dy),
            "trajectory_angle": np.arctan2(dy, dx),
        }
    )
    return out


def shots_to_xthreat_actions(shots: pd.DataFrame, *, player_lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Convert a Bepro xG shot table into xT shot actions."""
    frame = shots.copy()
    for col in ["match_id", "event_id", "team_id", "player_id"]:
        if col in frame.columns:
            frame[col] = _normalize_id_series(frame[col])
    if player_lookup is not None:
        frame = _attach_player_lookup(frame, player_lookup)
    elif "player_name" not in frame.columns:
        frame["player_name"] = "ID " + frame["player_id"].astype("string")

    start_x = _numeric(frame, "shot_x")
    start_y = _numeric(frame, "shot_y")
    goal = _bool(frame, "goal")
    is_open_play = _open_play_mask(frame, source="shots")
    time_seconds = _numeric(frame, "time_seconds")

    out = pd.DataFrame(
        {
            "event_id": frame["event_id"].astype("string"),
            "parent_pp_id": frame["event_id"].astype("string"),
            "match_id": frame["match_id"].astype("string"),
            "season_name": _series(frame, "season_name").astype("string"),
            "source_season_name": _series(frame, "source_season_name").astype("string"),
            "league": _series(frame, "league").astype("string"),
            "competition_round_name": _series(frame, "competition_round_name").astype("string"),
            "kickoff_time": _series(frame, "kickoff_time"),
            "period": _numeric(frame, "period_id").astype("Int64"),
            "period_id": _numeric(frame, "period_id").astype("Int64"),
            "period_name": _series(frame, "period_name").astype("string"),
            "event_time": _numeric(frame, "event_time"),
            "time_start": time_seconds,
            "time_end": time_seconds,
            "frame_event": np.rint(time_seconds * 10.0),
            "frame_start": np.rint(time_seconds * 10.0),
            "frame_end": np.rint(time_seconds * 10.0),
            "team_id": frame["team_id"].astype("string"),
            "team_name": _series(frame, "team_name").astype("string"),
            "team_shortname": _series(frame, "team_shortname").astype("string"),
            "player_id": frame["player_id"].astype("string"),
            "player_name": _series(frame, "player_name").astype("string").fillna("ID " + frame["player_id"].astype("string")),
            "player_position": _series(frame, "player_position", "missing").astype("string").fillna("missing"),
            "home_away": _series(frame, "home_away").astype("string"),
            "action_type": "shot",
            "source_action_type": _series(frame, "spadl_type", "shot").astype("string").fillna("shot"),
            "start_x": start_x,
            "start_y": start_y,
            "end_x": np.nan,
            "end_y": np.nan,
            "move_success": False,
            "goal": goal,
            "shot_goal_actual": goal,
            "outcome_flag": goal,
            "is_open_play": is_open_play,
            "is_set_piece_start": ~is_open_play,
            "is_corner_start": _bool(frame, "is_corner_start"),
            "is_free_kick_start": _bool(frame, "is_free_kick_start"),
            "is_throw_in_start": _bool(frame, "is_throw_in_start"),
            "is_goal_kick_start": _bool(frame, "is_goal_kick_start"),
            "provider_xg": _numeric(frame, "provider_xg"),
            "shot_bodypart": _series(frame, "bodypart_name").astype("string"),
            "shot_outcome": _series(frame, "cdf_outcome_detailed").astype("string"),
            "start_type": _series(frame, "spadl_type", "shot").astype("string").fillna("shot"),
            "end_type": "shot",
            "team_in_possession_phase_type": _series(frame, "game_state", "missing").astype("string"),
            "home_team_id": _series(frame, "home_team_id").astype("string"),
            "home_team_name": _series(frame, "home_team_name").astype("string"),
            "away_team_id": _series(frame, "away_team_id").astype("string"),
            "away_team_name": _series(frame, "away_team_name").astype("string"),
            "final_home_score": _series(frame, "final_home_score"),
            "final_away_score": _series(frame, "final_away_score"),
            "distance_covered": np.nan,
            "trajectory_angle": np.nan,
        }
    )
    return out


def spadl_to_xthreat_actions(
    spadl_actions: pd.DataFrame,
    match_meta: pd.DataFrame,
    *,
    player_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert canonical Bepro SPADL actions into xT pass/carry/shot rows."""
    if spadl_actions.empty:
        return pd.DataFrame()

    type_name = _normalized_spadl_type_name(spadl_actions)
    frame = spadl_actions[type_name.isin([*MOVE_TYPES, *SHOT_TYPES])].copy()
    frame["type_name"] = type_name.loc[frame.index].astype("string")
    if frame.empty:
        return pd.DataFrame()

    meta_cols = [
        "match_id",
        "competition_name",
        "competition_round_id",
        "competition_round_name",
        "kickoff_time",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "final_home_score",
        "final_away_score",
        "final_score",
        "season_id",
        "season_group_name",
    ]
    meta_use = match_meta[[c for c in meta_cols if c in match_meta.columns]].copy()
    if "match_id" in meta_use.columns:
        meta_use["match_id"] = _normalize_id_series(meta_use["match_id"])
    frame["match_id"] = _normalize_id_series(frame["match_id"])
    frame = frame.merge(meta_use, on="match_id", how="left")

    for col in ["team_id", "player_id", "home_team_id", "away_team_id"]:
        if col in frame.columns:
            frame[col] = _normalize_id_series(frame[col])

    type_name = _normalized_spadl_type_name(frame)
    frame["type_name"] = type_name
    is_pass = type_name.isin(PASS_TYPES)
    is_carry = type_name.isin(CARRY_TYPES)
    is_shot = type_name.isin(SHOT_TYPES)
    action_type = pd.Series(
        np.select([is_pass, is_carry, is_shot], ["pass", "carry", "shot"], default="other"),
        index=frame.index,
        dtype="string",
    )

    is_home = frame["team_id"].eq(frame["home_team_id"]).fillna(False).astype(bool)
    is_away = frame["team_id"].eq(frame["away_team_id"]).fillna(False).astype(bool)
    home_away = pd.Series(pd.NA, index=frame.index, dtype="string")
    home_away.loc[is_home.fillna(False)] = "home"
    home_away.loc[is_away.fillna(False)] = "away"
    start_x, start_y = _attack_normalized_xy(frame, "start_x", "start_y", is_home=is_home)
    end_x, end_y = _attack_normalized_xy(frame, "end_x", "end_y", is_home=is_home)

    result_name = _normalized_result_name(frame)
    frame["result_name"] = result_name
    success = result_name.eq("success").fillna(False).astype(bool)
    move_success = action_type.isin(["pass", "carry"]) & success
    goal = action_type.eq("shot") & success
    time_seconds = _numeric(frame, "time_seconds")
    home_team_name = _series(frame, "home_team_name").astype("string")
    away_team_name = _series(frame, "away_team_name").astype("string")
    team_name = _series(frame, "team_id").astype("string")
    team_name.loc[is_home] = home_team_name.loc[is_home]
    team_name.loc[is_away] = away_team_name.loc[is_away]
    team_name = team_name.fillna(_series(frame, "team_id").astype("string"))

    hs = _numeric(frame, "home_score")
    as_ = _numeric(frame, "away_score")
    team_score = pd.Series(np.where(is_home.to_numpy(dtype=bool), hs, as_), index=frame.index)
    opponent_team_score = pd.Series(np.where(is_home.to_numpy(dtype=bool), as_, hs), index=frame.index)
    set_piece_flags = _set_piece_flags(frame)
    dx = end_x - start_x
    dy = end_y - start_y

    event_id_source = (
        _series(frame, "original_event_id")
        if "original_event_id" in frame.columns
        else _series(frame, "action_id")
    )
    action_id = _series(frame, "action_id", pd.NA)
    player_id = _series(frame, "player_id").astype("string")

    out = pd.DataFrame(
        {
            "event_id": event_id_source.astype("string"),
            "parent_pp_id": event_id_source.astype("string"),
            "source_action_id": action_id.astype("string"),
            "match_id": frame["match_id"].astype("string"),
            "season_name": _series(frame, "season_name").astype("string"),
            "source_season_name": _series(frame, "source_season_name").astype("string"),
            "league": _series(frame, "league").astype("string"),
            "competition_name": _series(frame, "competition_name").astype("string"),
            "competition_round_id": _series(frame, "competition_round_id").astype("string"),
            "competition_round_name": _series(frame, "competition_round_name").astype("string"),
            "kickoff_time": _series(frame, "kickoff_time"),
            "period": _numeric(frame, "period_id").astype("Int64"),
            "period_id": _numeric(frame, "period_id").astype("Int64"),
            "period_name": _series(frame, "period_name", pd.NA).astype("string"),
            "event_time": time_seconds * 1000.0,
            "time_start": time_seconds,
            "time_end": time_seconds,
            "frame_event": np.rint(time_seconds * 10.0),
            "frame_start": np.rint(time_seconds * 10.0),
            "frame_end": np.rint(time_seconds * 10.0),
            "team_id": _series(frame, "team_id").astype("string"),
            "team_name": team_name,
            "team_shortname": team_name,
            "player_id": player_id,
            "player_name": pd.NA,
            "player_position": "missing",
            "home_away": home_away,
            "action_type": action_type,
            "source_action_type": type_name,
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x.where(~is_shot, np.nan),
            "end_y": end_y.where(~is_shot, np.nan),
            "move_success": move_success,
            "goal": goal,
            "shot_goal_actual": goal.where(is_shot, pd.NA),
            "outcome_flag": np.where(is_shot, goal, move_success),
            "result_name": result_name,
            "is_open_play": ~set_piece_flags["is_set_piece_start"],
            "is_set_piece_start": set_piece_flags["is_set_piece_start"],
            "is_corner_start": set_piece_flags["is_corner_start"],
            "is_free_kick_start": set_piece_flags["is_free_kick_start"],
            "is_throw_in_start": set_piece_flags["is_throw_in_start"],
            "is_goal_kick_start": set_piece_flags["is_goal_kick_start"],
            "is_penalty_start": set_piece_flags["is_penalty_start"],
            "set_piece_sub_type": _series(frame, "set_piece_type", pd.NA).astype("string"),
            "provider_xg": _numeric(frame, "provider_xg"),
            "shot_bodypart": _series(frame, "prop_body_part", pd.NA).astype("string"),
            "is_header": _series(frame, "prop_body_part", "").astype("string").str.contains("head", case=False, na=False),
            "pass_outcome": pd.Series(
                np.where(success, "successful", np.where(result_name.eq("offside").fillna(False).astype(bool), "offside", "unsuccessful")),
                index=frame.index,
            ).astype("string"),
            "is_cross": type_name.isin(CROSS_TYPES),
            "is_key_pass": _bool(frame, "is_key_pass"),
            "is_assist": _bool(frame, "is_assist"),
            "start_type": np.where(set_piece_flags["is_set_piece_start"], _series(frame, "set_piece_type", pd.NA), "open_play"),
            "end_type": np.select([is_pass, is_carry, is_shot], ["target", "carry_end", "shot"], default="missing"),
            "team_score": team_score,
            "opponent_team_score": opponent_team_score,
            "team_in_possession_phase_type": _game_state(team_score, opponent_team_score),
            "home_team_id": _series(frame, "home_team_id").astype("string"),
            "home_team_name": _series(frame, "home_team_name").astype("string"),
            "away_team_id": _series(frame, "away_team_id").astype("string"),
            "away_team_name": _series(frame, "away_team_name").astype("string"),
            "final_home_score": _series(frame, "final_home_score"),
            "final_away_score": _series(frame, "final_away_score"),
            "final_score": _series(frame, "final_score"),
            "season_id": _series(frame, "season_id").astype("string"),
            "vendor_name": "Bepro Drive API (SPADL)",
            "distance_covered": np.where(action_type.isin(["pass", "carry"]), np.hypot(dx, dy), np.nan),
            "trajectory_angle": np.where(action_type.isin(["pass", "carry"]), np.arctan2(dy, dx), np.nan),
        }
    )
    if player_lookup is not None:
        out = _attach_player_lookup(out, player_lookup)
    return out


def add_phase_index(actions: pd.DataFrame, *, max_gap_seconds: float = 20.0) -> pd.DataFrame:
    """Add a conservative possession-phase proxy for Bepro event tables.

    Bepro Drive pass/shot tables do not expose a stable possession id. We start
    a new phase after period/match changes, team changes, long gaps, shots, and
    unsuccessful moves.
    """
    out = actions.copy()
    sort_cols = [c for c in ["match_id", "period_id", "time_start", "event_time", "event_id"] if c in out.columns]
    out = out.sort_values(sort_cols, kind="mergesort", ignore_index=True)
    out["action_idx"] = out.groupby(["match_id", "period_id"], dropna=False).cumcount()

    match_period = out["match_id"].astype("string") + "|" + out["period_id"].astype("string")
    prev_match_period = match_period.shift()
    prev_team = out["team_id"].astype("string").shift()
    prev_time = pd.to_numeric(out["time_start"], errors="coerce").shift()
    current_time = pd.to_numeric(out["time_start"], errors="coerce")
    gap = current_time - prev_time

    prev_action_type = out["action_type"].astype("string").shift()
    prev_move_success = out["move_success"].fillna(False).astype(bool).shift(fill_value=False).astype(bool)
    previous_ended_phase = (
        prev_action_type.eq("shot")
        | (prev_action_type.isin(["pass", "carry", "drive", "dribble", "cross"]) & ~prev_move_success)
    )
    new_phase = (
        match_period.ne(prev_match_period)
        | out["team_id"].astype("string").ne(prev_team)
        | gap.gt(float(max_gap_seconds)).fillna(False)
        | previous_ended_phase.fillna(False)
    )
    new_phase = new_phase.fillna(False).astype(bool)
    if len(new_phase) > 0:
        new_phase.iloc[0] = True
    out["phase_index"] = new_phase.astype(int).groupby(out["match_id"].astype("string"), sort=False).cumsum()
    return out


def prepare_bepro_xthreat_actions(
    passes: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    player_lookup: pd.DataFrame | None = None,
    keep_only_open_play: bool = True,
    keep_only_in_pitch: bool = True,
    max_phase_gap_seconds: float = 20.0,
) -> pd.DataFrame:
    """Combine Bepro pass and shot tables into the canonical xT action schema."""
    pass_actions = passes_to_xthreat_actions(passes) if not passes.empty else pd.DataFrame()
    shot_actions = shots_to_xthreat_actions(shots, player_lookup=player_lookup) if not shots.empty else pd.DataFrame()
    frame = pd.concat([pass_actions, shot_actions], ignore_index=True, sort=False)
    if frame.empty:
        return frame

    if keep_only_open_play and "is_open_play" in frame.columns:
        frame = frame[frame["is_open_play"].fillna(False).astype(bool)].copy()

    move_mask = frame["action_type"].isin(["pass", "carry"])
    shot_mask = frame["action_type"].eq("shot")
    start_ok = _within_pitch(frame, "start_x", "start_y")
    end_ok = _within_pitch(frame, "end_x", "end_y")
    if keep_only_in_pitch:
        frame = frame[(shot_mask & start_ok) | (move_mask & start_ok & end_ok)].copy()
    else:
        frame = frame[(shot_mask & start_ok) | move_mask].copy()

    frame = add_phase_index(frame, max_gap_seconds=max_phase_gap_seconds)
    preferred = [
        "event_id",
        "parent_pp_id",
        "match_id",
        "season_name",
        "source_season_name",
        "league",
        "competition_round_name",
        "kickoff_time",
        "period",
        "period_id",
        "time_start",
        "time_end",
        "frame_event",
        "frame_start",
        "frame_end",
        "phase_index",
        "action_idx",
        "team_id",
        "team_name",
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
        "is_open_play",
        "is_set_piece_start",
        "distance_covered",
        "trajectory_angle",
    ]
    rest = [c for c in frame.columns if c not in preferred]
    return frame[[c for c in preferred if c in frame.columns] + rest].reset_index(drop=True)


def prepare_bepro_xthreat_actions_from_spadl(
    spadl_actions: pd.DataFrame,
    match_meta: pd.DataFrame,
    *,
    player_lookup: pd.DataFrame | None = None,
    keep_only_open_play: bool = True,
    keep_only_in_pitch: bool = True,
    max_phase_gap_seconds: float = 20.0,
    train_seasons: Iterable[str] = DEFAULT_TRAIN_SEASONS,
    test_seasons: Iterable[str] = DEFAULT_TEST_SEASONS,
) -> pd.DataFrame:
    """Build the canonical Bepro xT schema directly from the SPADL store."""
    frame = spadl_to_xthreat_actions(spadl_actions, match_meta, player_lookup=player_lookup)
    if frame.empty:
        return frame

    if keep_only_open_play and "is_open_play" in frame.columns:
        frame = frame[frame["is_open_play"].fillna(False).astype(bool)].copy()

    move_mask = frame["action_type"].isin(["pass", "carry"])
    shot_mask = frame["action_type"].eq("shot")
    start_ok = _within_pitch(frame, "start_x", "start_y")
    end_ok = _within_pitch(frame, "end_x", "end_y")
    if keep_only_in_pitch:
        frame = frame[(shot_mask & start_ok) | (move_mask & start_ok & end_ok)].copy()
    else:
        frame = frame[(shot_mask & start_ok) | move_mask].copy()

    if frame.empty:
        return frame.reset_index(drop=True)

    frame = _split_season_labels(frame, train_seasons=train_seasons, test_seasons=test_seasons)
    frame = add_phase_index(frame, max_gap_seconds=max_phase_gap_seconds)
    preferred = [
        "event_id",
        "parent_pp_id",
        "source_action_id",
        "match_id",
        "season_name",
        "source_season_name",
        "model_split",
        "league",
        "competition_name",
        "competition_round_id",
        "competition_round_name",
        "kickoff_time",
        "period",
        "period_id",
        "time_start",
        "time_end",
        "frame_event",
        "frame_start",
        "frame_end",
        "phase_index",
        "action_idx",
        "team_id",
        "team_name",
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
        "is_open_play",
        "is_set_piece_start",
        "distance_covered",
        "trajectory_angle",
    ]
    rest = [c for c in frame.columns if c not in preferred]
    return frame[[c for c in preferred if c in frame.columns] + rest].reset_index(drop=True)


def build_bepro_xthreat_actions(
    *,
    spadl_dir: str | Path = DEFAULT_SPADL_DIR,
    source: str = "spadl",
    passes_path: str | Path = DEFAULT_PASSES_PATH,
    shots_path: str | Path = DEFAULT_SHOTS_PATH,
    player_lookup_path: str | Path | None = DEFAULT_PLAYER_LOOKUP_PATH,
    build_missing_spadl: bool = False,
    build_missing_tables: bool = False,
    event_remote: str = DEFAULT_EVENT_REMOTE,
    leagues: Iterable[str] = DEFAULT_LEAGUES,
    seasons: Iterable[str] = (*DEFAULT_TRAIN_SEASONS, *DEFAULT_TEST_SEASONS),
    train_seasons: Iterable[str] = DEFAULT_TRAIN_SEASONS,
    test_seasons: Iterable[str] = DEFAULT_TEST_SEASONS,
    rclone: str | Path = DEFAULT_RCLONE,
    rclone_config: str | Path = DEFAULT_RCLONE_CONFIG,
    limit_matches: int | None = None,
    limit_matches_per_season: int | None = None,
    include_lineup: bool = True,
    workers: int = 1,
    keep_only_open_play: bool = True,
    keep_only_in_pitch: bool = True,
    max_phase_gap_seconds: float = 20.0,
) -> tuple[pd.DataFrame, BeproXThreatActionStats]:
    """Load/build Bepro source tables and return xT-ready actions plus stats.

    ``source="spadl"`` is the active path and reads
    ``tmp/data/bepro_spadl_k1/actions.parquet`` + ``match_meta.parquet`` by
    default. ``source="derived_tables"`` keeps the older pass/shot-cache path
    available for reproducibility checks.
    """
    if source not in {"spadl", "derived_tables"}:
        raise ValueError("source must be 'spadl' or 'derived_tables'")

    player_lookup = _load_player_lookup(player_lookup_path)
    if source == "spadl":
        spadl_dir = Path(spadl_dir)
        actions_path = spadl_dir / "actions.parquet"
        meta_path = spadl_dir / "match_meta.parquet"
        if not actions_path.exists() or not meta_path.exists():
            if not build_missing_spadl and not build_missing_tables:
                missing = [str(p) for p in [actions_path, meta_path] if not p.exists()]
                raise FileNotFoundError(f"Bepro SPADL store not found: {', '.join(missing)}")
            from pipelines.bepro_ingest import build_bepro_spadl_store, write_store

            spadl_actions, match_meta = build_bepro_spadl_store(
                event_remote=event_remote,
                leagues=leagues,
                seasons=seasons,
                rclone=rclone,
                rclone_config=rclone_config,
                limit_matches=limit_matches,
                limit_matches_per_season=limit_matches_per_season,
                workers=workers,
            )
            write_store(spadl_actions, match_meta, spadl_dir)
        else:
            spadl_actions = pd.read_parquet(actions_path)
            match_meta = pd.read_parquet(meta_path)

        actions = prepare_bepro_xthreat_actions_from_spadl(
            spadl_actions,
            match_meta,
            player_lookup=player_lookup,
            keep_only_open_play=keep_only_open_play,
            keep_only_in_pitch=keep_only_in_pitch,
            max_phase_gap_seconds=max_phase_gap_seconds,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
        )
        stats = summarize_action_table(actions)
        return actions, stats

    passes_path = Path(passes_path)
    shots_path = Path(shots_path)

    if passes_path.exists():
        passes = pd.read_parquet(passes_path)
    elif build_missing_tables:
        from xpass.bepro_drive_passes import build_pass_table_from_spadl, write_pass_table

        passes = build_pass_table_from_spadl(
            spadl_dir=spadl_dir,
            player_lookup_path=player_lookup_path,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
        )
        write_pass_table(passes, passes_path)
    else:
        raise FileNotFoundError(f"Pass table not found: {passes_path}")

    if shots_path.exists():
        shots = pd.read_parquet(shots_path)
    elif build_missing_tables:
        from xg.bepro_drive_shots import build_shot_table_from_spadl, write_shot_table

        shots = build_shot_table_from_spadl(
            spadl_dir=spadl_dir,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
        )
        write_shot_table(shots, shots_path)
    else:
        raise FileNotFoundError(f"Shot table not found: {shots_path}")

    actions = prepare_bepro_xthreat_actions(
        passes,
        shots,
        player_lookup=player_lookup,
        keep_only_open_play=keep_only_open_play,
        keep_only_in_pitch=keep_only_in_pitch,
        max_phase_gap_seconds=max_phase_gap_seconds,
    )
    stats = summarize_action_table(actions)
    return actions, stats


def summarize_action_table(actions: pd.DataFrame) -> BeproXThreatActionStats:
    action_type = actions.get("action_type", pd.Series(dtype="string")).astype("string")
    move_mask = action_type.isin(["pass", "carry"])
    goal = actions.get("goal", pd.Series(False, index=actions.index)).fillna(False).astype(bool)
    move_success = actions.get("move_success", pd.Series(False, index=actions.index)).fillna(False).astype(bool)
    is_open_play = actions.get("is_open_play", pd.Series(False, index=actions.index)).fillna(False).astype(bool)
    n_phases = 0
    if {"match_id", "phase_index"} <= set(actions.columns):
        n_phases = int(actions[["match_id", "phase_index"]].drop_duplicates().shape[0])

    return BeproXThreatActionStats(
        n_rows=int(len(actions)),
        n_matches=int(actions["match_id"].astype("string").nunique()) if "match_id" in actions.columns else 0,
        n_pass=int(action_type.eq("pass").sum()),
        n_carry=int(action_type.eq("carry").sum()),
        n_shot=int(action_type.eq("shot").sum()),
        n_successful_moves=int((move_mask & move_success).sum()),
        n_goals=int(goal.sum()),
        n_open_play=int(is_open_play.sum()),
        n_set_piece=int((~is_open_play).sum()),
        n_phases=n_phases,
    )


def write_actions(
    actions: pd.DataFrame,
    stats: BeproXThreatActionStats,
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
    parser = argparse.ArgumentParser(description="Build a Bepro xT action table from the SPADL store.")
    parser.add_argument("--spadl-dir", type=Path, default=DEFAULT_SPADL_DIR,
                        help="Directory holding actions.parquet + match_meta.parquet (SPADL store).")
    parser.add_argument("--source", choices=["spadl", "derived_tables"], default="spadl",
                        help="Use canonical SPADL store (default) or older derived pass/shot caches.")
    parser.add_argument("--passes-path", type=Path, default=DEFAULT_PASSES_PATH)
    parser.add_argument("--shots-path", type=Path, default=DEFAULT_SHOTS_PATH)
    parser.add_argument("--player-lookup-path", type=Path, default=DEFAULT_PLAYER_LOOKUP_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_ACTIONS_PATH)
    parser.add_argument("--build-missing-spadl", action="store_true")
    parser.add_argument("--build-missing-tables", action="store_true")
    parser.add_argument("--include-set-pieces", action="store_true")
    parser.add_argument("--include-out-of-pitch", action="store_true")
    parser.add_argument("--max-phase-gap-seconds", type=float, default=20.0)
    parser.add_argument("--event-remote", default=DEFAULT_EVENT_REMOTE)
    parser.add_argument("--rclone", type=Path, default=DEFAULT_RCLONE)
    parser.add_argument("--rclone-config", type=Path, default=DEFAULT_RCLONE_CONFIG)
    parser.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    parser.add_argument("--seasons", nargs="+", default=[*DEFAULT_TRAIN_SEASONS, *DEFAULT_TEST_SEASONS])
    parser.add_argument("--train-seasons", nargs="+", default=DEFAULT_TRAIN_SEASONS)
    parser.add_argument("--test-seasons", nargs="+", default=DEFAULT_TEST_SEASONS)
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--limit-matches-per-season", type=int, default=None)
    parser.add_argument("--no-lineup", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    actions, stats = build_bepro_xthreat_actions(
        spadl_dir=args.spadl_dir,
        source=args.source,
        passes_path=args.passes_path,
        shots_path=args.shots_path,
        player_lookup_path=args.player_lookup_path,
        build_missing_spadl=args.build_missing_spadl,
        build_missing_tables=args.build_missing_tables,
        event_remote=args.event_remote,
        leagues=args.leagues,
        seasons=args.seasons,
        train_seasons=args.train_seasons,
        test_seasons=args.test_seasons,
        rclone=args.rclone,
        rclone_config=args.rclone_config,
        limit_matches=args.limit_matches,
        limit_matches_per_season=args.limit_matches_per_season,
        include_lineup=not args.no_lineup,
        workers=args.workers,
        keep_only_open_play=not args.include_set_pieces,
        keep_only_in_pitch=not args.include_out_of_pitch,
        max_phase_gap_seconds=args.max_phase_gap_seconds,
    )
    path = write_actions(actions, stats, output_path=args.out)
    print(f"[bepro-drive/xthreat] wrote {len(actions):,} rows to {path}")
    print(json.dumps(asdict(stats), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
