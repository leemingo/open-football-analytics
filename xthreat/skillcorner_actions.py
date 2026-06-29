"""Build SkillCorner action tables for center-origin xT experiments."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CACHE_DIR = Path("tmp/data/cache")
DEFAULT_XPASS_PASSES_PATH = Path("tmp/data/skillcorner_xpass/passes.parquet")
DEFAULT_XG_SHOTS_PATH = Path("tmp/data/skillcorner_xg/shots.parquet")
DEFAULT_OUTPUT_DIR = Path("tmp/data/skillcorner_xthreat")
DEFAULT_ACTIONS_PATH = DEFAULT_OUTPUT_DIR / "actions.parquet"

PITCH_X = 105.0
PITCH_Y = 68.0


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


def _event_files(cache_dir: str | Path) -> list[Path]:
    return sorted(Path(cache_dir).glob("*/events_aug.parquet"))


def _as_str_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


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
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    passes_path: str | Path = DEFAULT_XPASS_PASSES_PATH,
    shots_path: str | Path = DEFAULT_XG_SHOTS_PATH,
    limit_matches: int | None = None,
    keep_only_open_play: bool = True,
) -> tuple[pd.DataFrame, ActionTableStats]:
    """Build the canonical center-origin xT action table."""
    actions = load_cached_epv_actions(cache_dir, limit_matches=limit_matches)
    actions = attach_match_metadata(actions, passes_path=passes_path, shots_path=shots_path)
    actions = attach_skillcorner_pass_xthreat(actions, passes_path=passes_path)
    actions = attach_actual_shot_goals(actions, shots_path=shots_path)
    actions = prepare_xthreat_actions(actions, keep_only_open_play=keep_only_open_play)
    stats = summarize_action_table(actions)
    return actions, stats


def summarize_action_table(actions: pd.DataFrame) -> ActionTableStats:
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
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--passes-path", type=Path, default=DEFAULT_XPASS_PASSES_PATH)
    parser.add_argument("--shots-path", type=Path, default=DEFAULT_XG_SHOTS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_ACTIONS_PATH)
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--include-set-pieces", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    actions, stats = build_skillcorner_xthreat_actions(
        cache_dir=args.cache_dir,
        passes_path=args.passes_path,
        shots_path=args.shots_path,
        limit_matches=args.limit_matches,
        keep_only_open_play=not args.include_set_pieces,
    )
    path = write_actions(actions, stats, output_path=args.out)
    print(f"[xT/actions] wrote {len(actions):,} rows to {path}")
    print(json.dumps(asdict(stats), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
