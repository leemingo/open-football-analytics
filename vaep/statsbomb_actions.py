"""Build a SPADL-style action table from StatsBomb Open Data for VAEP.

Data source: the public `statsbomb/open-data <https://github.com/statsbomb/open-data>`_
GitHub repository (StatsBomb's open data licence -- attribution, non-commercial). VAEP
needs a typed action stream with real end coordinates; StatsBomb Open Data already
provides one -- including explicit ``Carry`` events -- converted to SPADL with
``football_cdf``'s ``StatsbombDataPreprocessor``:

    raw event JSON -> preprocess_cdf_events -> preprocess_spadl_events

Unlike DFL/Sportec's point events, StatsBomb events already carry real end
coordinates and explicit dribble/carry actions, so no end-coordinate reconstruction
step is needed (contrast with the Sportec version this replaces).

Default sample: the full FIFA World Cup 2022 (``competition_id=43``,
``season_id=106``), 64 matches -- downloaded on demand as individual JSON files from
GitHub's raw file host and cached locally (no ``git clone`` of the full repo needed).
"""
from __future__ import annotations

import argparse
import json
import urllib.request as request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from football_cdf.constants import CDF_PERIOD_MAP
from football_cdf.statsbomb_preprocessing import StatsbombDataPreprocessor as _Statsbomb

RAW_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# FIFA World Cup 2022. Any other statsbomb/open-data competition/season works too --
# pass its ids to build_actions()/list_competition_match_ids().
DEFAULT_COMPETITION_ID = 43
DEFAULT_SEASON_ID = 106

DEFAULT_CACHE = Path("tmp/data/statsbomb_open")

SHOT_TYPES = ["shot", "shot_penalty", "shot_freekick"]
PERIOD_ID_FROM_NAME = {name: pid for pid, name in CDF_PERIOD_MAP.items()}

# DFL point-events needed end-coordinate reconstruction; StatsBomb events already
# carry real end coordinates (including explicit Carry -> dribble actions), so this
# module has no equivalent of Sportec's MOVE_TYPES / link_end_to_next.


# --------------------------------------------------------------------------- #
# Download (individual files from GitHub's raw host; no full repo clone needed)
# --------------------------------------------------------------------------- #
def _fetch(url: str, out: Path) -> None:
    if out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    request.urlretrieve(url, out)


def ensure_competition_cached(
    competition_id: int = DEFAULT_COMPETITION_ID,
    season_id: int = DEFAULT_SEASON_ID,
    cache_dir: str | Path = DEFAULT_CACHE,
) -> Path:
    """Download ``competitions.json`` + the competition's match list; return the data root."""
    data_root = Path(cache_dir) / "data"
    _fetch(f"{RAW_BASE_URL}/competitions.json", data_root / "competitions.json")
    _fetch(
        f"{RAW_BASE_URL}/matches/{competition_id}/{season_id}.json",
        data_root / "matches" / str(competition_id) / f"{season_id}.json",
    )
    return data_root.parent


def list_competition_match_ids(
    competition_id: int = DEFAULT_COMPETITION_ID,
    season_id: int = DEFAULT_SEASON_ID,
    cache_dir: str | Path = DEFAULT_CACHE,
) -> list[str]:
    """All match ids in one statsbomb/open-data competition+season."""
    root = ensure_competition_cached(competition_id, season_id, cache_dir)
    matches_path = root / "data" / "matches" / str(competition_id) / f"{season_id}.json"
    matches = json.loads(matches_path.read_text(encoding="utf-8"))
    return [str(m["match_id"]) for m in matches]


def download_statsbomb_match(
    match_id: str,
    cache_dir: str | Path = DEFAULT_CACHE,
    *,
    competition_id: int = DEFAULT_COMPETITION_ID,
    season_id: int = DEFAULT_SEASON_ID,
) -> Path:
    """Download one match's events + lineups; return the data root (``resolve_statsbomb_data_root``-compatible)."""
    root = ensure_competition_cached(competition_id, season_id, cache_dir)
    data_root = root / "data"
    _fetch(f"{RAW_BASE_URL}/events/{match_id}.json", data_root / "events" / f"{match_id}.json")
    _fetch(f"{RAW_BASE_URL}/lineups/{match_id}.json", data_root / "lineups" / f"{match_id}.json")
    return root


def download_matches(
    match_ids: list[str],
    cache_dir: str | Path = DEFAULT_CACHE,
    *,
    competition_id: int = DEFAULT_COMPETITION_ID,
    season_id: int = DEFAULT_SEASON_ID,
    max_workers: int = 8,
) -> Path:
    """Download many matches concurrently; return the shared data root."""
    root = ensure_competition_cached(competition_id, season_id, cache_dir)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(download_statsbomb_match, mid, cache_dir,
                        competition_id=competition_id, season_id=season_id)
            for mid in match_ids
        ]
        for future in as_completed(futures):
            future.result()  # surface any download error immediately
    return root


# --------------------------------------------------------------------------- #
# Raw JSON -> SPADL actions (football_cdf StatsBomb preprocessor, no tracking)
# --------------------------------------------------------------------------- #
def _spadl_frame(root: Path, match_id: str) -> tuple[pd.DataFrame, dict]:
    sb = _Statsbomb(str(root), str(match_id), load_360=False)
    spadl = sb.preprocess_spadl_events()
    return spadl, sb.match_metadata


def _to_canonical(spadl: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Reduce the SPADL frame to the canonical VAEP action schema."""
    df = spadl[spadl["spadl_type"].notna()].copy()

    out = pd.DataFrame(index=df.index)
    out["match_id"] = str(meta.get("match_id"))
    out["season_name"] = str(meta.get("season_name"))
    out["type_name"] = df["spadl_type"].astype("string")
    # StatsBomb's own result_name already distinguishes fail/success/offside/cards,
    # so (unlike the Sportec version) we use it directly instead of collapsing to
    # a plain success/fail flag.
    out["result_name"] = df["result_name"].astype("string")
    out["bodypart_name"] = df["bodypart_name"].astype("string").fillna("other")
    out["team_id"] = df["team_id"].astype("string")
    out["player_id"] = df["player_id"].astype("string")
    out["start_x"] = pd.to_numeric(df["start_x"], errors="coerce")
    out["start_y"] = pd.to_numeric(df["start_y"], errors="coerce")
    # StatsBomb already has real end coordinates (incl. Carry -> dribble); only
    # point-like actions (tackles, interceptions, ...) fall back to start == end.
    out["end_x"] = pd.to_numeric(df["end_x"], errors="coerce").fillna(out["start_x"])
    out["end_y"] = pd.to_numeric(df["end_y"], errors="coerce").fillna(out["start_y"])
    out["period_id"] = df["period_id"].map(PERIOD_ID_FROM_NAME).fillna(0).astype(int)
    # time_seconds is already period-relative elapsed time (StatsBomb has no
    # reliable utc_timestamp -- see football-cdf's StatsBomb README section).
    out["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce").fillna(0.0)

    home_id = str(meta.get("home_team_id"))
    away_id = str(meta.get("away_team_id"))
    out["home_team_id"] = home_id
    out["away_team_id"] = away_id

    out = out.dropna(subset=["start_x", "start_y"])
    out = out.sort_values(["period_id", "time_seconds"], kind="mergesort", ignore_index=True)
    return out


# --------------------------------------------------------------------------- #
# Goal bookkeeping
# --------------------------------------------------------------------------- #
def add_bookkeeping(actions: pd.DataFrame) -> pd.DataFrame:
    """Per-game action_id, opponent id, goal (scoring_team) + cumulative score."""
    out = actions.sort_values(["match_id", "period_id", "time_seconds"],
                              kind="mergesort", ignore_index=True).copy()
    out["action_id"] = out.groupby("match_id").cumcount()
    out["opp_team_id"] = out["away_team_id"].where(out["team_id"] == out["home_team_id"], out["home_team_id"])

    is_goal = (out["type_name"].isin(SHOT_TYPES) & (out["result_name"] == "success")).fillna(False)
    out["scoring_team_id"] = out["team_id"].where(is_goal).astype("string")

    home_goal = (out["scoring_team_id"] == out["home_team_id"]).fillna(False).astype(int)
    away_goal = (out["scoring_team_id"] == out["away_team_id"]).fillna(False).astype(int)
    home_cum = home_goal.groupby(out["match_id"], sort=False).cumsum() - home_goal
    away_cum = away_goal.groupby(out["match_id"], sort=False).cumsum() - away_goal
    is_home = (out["team_id"] == out["home_team_id"]).fillna(False)
    out["team_goals_before"] = home_cum.where(is_home, away_cum).astype(int)
    out["opp_goals_before"] = away_cum.where(is_home, home_cum).astype(int)
    out["goal_diff_before"] = out["team_goals_before"] - out["opp_goals_before"]
    return out


# --------------------------------------------------------------------------- #
# Public builders
# --------------------------------------------------------------------------- #
def match_actions(root: str | Path, match_id: str) -> pd.DataFrame:
    """Build the canonical VAEP action table for one already-downloaded match."""
    spadl, meta = _spadl_frame(Path(root), str(match_id))
    return _to_canonical(spadl, meta)


def build_actions(
    match_ids: list[str] | None = None,
    *,
    competition_id: int = DEFAULT_COMPETITION_ID,
    season_id: int = DEFAULT_SEASON_ID,
    cache_dir: str | Path = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Download the given (or all) matches for one competition+season and build one action table."""
    match_ids = match_ids or list_competition_match_ids(competition_id, season_id, cache_dir)
    root = download_matches(match_ids, cache_dir, competition_id=competition_id, season_id=season_id)
    frames = [match_actions(root, mid) for mid in match_ids]
    actions = pd.concat(frames, ignore_index=True)
    return add_bookkeeping(actions)


def player_directory(
    match_ids: list[str] | None = None,
    *,
    competition_id: int = DEFAULT_COMPETITION_ID,
    season_id: int = DEFAULT_SEASON_ID,
    cache_dir: str | Path = DEFAULT_CACHE,
) -> pd.DataFrame:
    """``player_id -> player_name / playing_position / team`` from the match lineups."""
    match_ids = match_ids or list_competition_match_ids(competition_id, season_id, cache_dir)
    root = download_matches(match_ids, cache_dir, competition_id=competition_id, season_id=season_id)
    rows = []
    for mid in match_ids:
        sb = _Statsbomb(str(root), str(mid), load_360=False)
        keep = [c for c in ["player_id", "player_name", "playing_position", "team_id", "team_name"] if c in sb.lineup.columns]
        rows.append(sb.lineup[keep])
    directory = pd.concat(rows, ignore_index=True)
    directory["player_id"] = directory["player_id"].astype("string")
    return directory.dropna(subset=["player_id"]).drop_duplicates("player_id", keep="first")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the StatsBomb Open Data VAEP action table.")
    parser.add_argument("--match-ids", nargs="+", default=None, help="StatsBomb match ids (default: all matches in the competition/season)")
    parser.add_argument("--competition-id", type=int, default=DEFAULT_COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=DEFAULT_SEASON_ID)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    actions = build_actions(args.match_ids, competition_id=args.competition_id,
                            season_id=args.season_id, cache_dir=args.cache_dir)
    print(f"[statsbomb-actions] {len(actions):,} actions / {actions['match_id'].nunique()} matches")
    print(actions["type_name"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
