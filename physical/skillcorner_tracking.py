"""Load SkillCorner Open Data tracking into the canonical physical-metrics frame.

This is the open-data counterpart to the licensed provider adapters: it stops
**before** any kinematics and emits one canonical row per tracked object per
timestamp, which :mod:`physical.kinematics` then turns into speed/acceleration.

Coordinates are metres on a pitch-centred origin (0, 0 at the centre spot),
rescaled to the canonical 105 x 68 pitch. Each period is oriented so the home
team attacks +x. ``timestamp`` is period-relative seconds; ``match_seconds`` is
the match clock (used only to filter a player's appearance window).

**Ball-in-play.** The public SkillCorner physical workflow defines P60 BIP as
TIP + OTIP: a frame is included when the possession label identifies the home or
away team. This matches the published aggregate fields
``minutes_full_tip + minutes_full_otip``. The football-cdf loader is reused for
all parsing and its broader ball-state label is then overwritten with this
possession-based rule. Frames with no owning team remain available for coverage
diagnostics, but are not part of the canonical P60 BIP denominator.

Files come from https://github.com/SkillCorner/opendata (10 A-League 2024/25
matches). The tracking JSONL files are stored with **Git LFS**, so they must be
fetched from the ``media.githubusercontent.com`` endpoint;
``raw.githubusercontent.com`` returns a ~133-byte pointer file instead.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from football_cdf.constants import CDF_PERIOD_MAP, PITCH_X, PITCH_Y
from football_cdf.skillcorner_preprocessing import SkillcornerDataPreprocessor

from physical.definitions import POSITION_GROUPS
from physical.kinematics import normalize_position_group

__all__ = [
    "OPENDATA_MATCH_IDS",
    "ProviderBundle",
    "canonical_tracking_frame",
    "download_skillcorner_opendata",
    "iter_skillcorner_tracking",
    "load_skillcorner_physical_aggregate",
    "load_dynamic_events",
    "load_phases_of_play",
    "load_skillcorner_bundle",
    "resolve_match_files",
]

SKILLCORNER_ROOT_ENV = "SKILLCORNER_ROOT"
DEFAULT_OPENDATA_ROOT = Path("tmp/data/skillcorner_opendata")

# The 10 matches published in the SkillCorner Open Data repo. Hard-coded rather
# than discovered through the GitHub API so the tutorial is reproducible offline
# once cached, matching the xg/xpass/xthreat notebooks.
OPENDATA_MATCH_IDS = [
    1886347, 1899585, 1925299, 1953632, 1996435,
    2006229, 2011166, 2013725, 2015213, 2017461,
]

_RAW_BASE = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches"
_LFS_BASE = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"
PHYSICAL_AGGREGATE_FILENAME = "aus1league_physicalaggregates_20242025.csv"
_AGGREGATE_URL = (
    "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/aggregates/"
    + PHYSICAL_AGGREGATE_FILENAME
)

TRACKING_SUFFIX = "tracking_extrapolated.jsonl"
_SMALL_SUFFIXES = ("match.json", "dynamic_events.csv", "phases_of_play.csv")

CANONICAL_TRACKING_COLUMNS = (
    "provider", "season", "match_id",
    "period_id", "period", "frame_id", "source_frame_id",
    "timestamp", "match_seconds", "source_timestamp",
    "object_id", "player_id", "team_id", "home_away", "ball",
    "x", "y", "z", "is_detected", "ball_state", "ball_owning_team_id",
)

BALL_STATE_DEFINITION = "possession.team_id-tip_plus_otip-v1"

# Undetected players are sometimes parked far outside the pitch. Reuse the
# provider tolerance so those placeholder coordinates become missing rather than
# impossible sprints.
_OUTLIER_TOLERANCE_M = SkillcornerDataPreprocessor.UNDETECTED_OUTLIER_TOL

_LINEUP_COLUMNS = (
    "provider", "season", "match_id", "team_id", "team_name", "home_away",
    "player_id", "object_id", "player_name", "uniform_number",
    "position", "position_group", "starting",
    "appearance_start_seconds", "appearance_end_seconds",
    "provider_playing_time_ms",
)


@dataclass
class ProviderBundle:
    """Canonical match dimensions plus the context needed to parse tracking."""

    match_metadata: dict[str, Any]
    lineup: pd.DataFrame
    context: dict[str, Any]


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def _remote_size(url: str, *, timeout: float = 60.0) -> int | None:
    """Return Content-Length for ``url``, or None when the server withholds it."""

    request = Request(url, method="HEAD")
    try:
        with urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
    except URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Could not HEAD {url}: {exc}") from exc
    return int(length) if length is not None else None


def _download(url: str, target: Path, *, timeout: float = 600.0) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(url, timeout=timeout) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    except URLError as exc:  # pragma: no cover - network dependent
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {url}: {exc}") from exc
    partial.replace(target)


def download_skillcorner_opendata(
    out_dir: str | Path = DEFAULT_OPENDATA_ROOT,
    *,
    match_ids: list[int | str] | None = None,
    include_tracking: bool = True,
    include_aggregate: bool = True,
    force: bool = False,
) -> Path:
    """Download SkillCorner Open Data match files into ``out_dir``.

    Tracking JSONL is Git-LFS backed and comes from the media endpoint. Each
    file is considered complete only when its local size matches the remote
    ``Content-Length``; a size check (rather than mere existence) is what stops a
    previously interrupted run -- or a 133-byte LFS pointer downloaded from the
    wrong endpoint -- from being mistaken for real tracking data.

    All ten matches with tracking total roughly 915 MB.
    """

    root = Path(out_dir)
    ids = [str(m) for m in (match_ids if match_ids is not None else OPENDATA_MATCH_IDS)]
    suffixes = list(_SMALL_SUFFIXES) + ([TRACKING_SUFFIX] if include_tracking else [])

    for match_id in ids:
        match_dir = root / "data" / "matches" / match_id
        for suffix in suffixes:
            filename = f"{match_id}_{suffix}"
            target = match_dir / filename
            base = _LFS_BASE if suffix == TRACKING_SUFFIX else _RAW_BASE
            url = f"{base}/{match_id}/{filename}"

            if target.exists() and not force:
                if suffix != TRACKING_SUFFIX:
                    continue
                expected = _remote_size(url)
                if expected is None or target.stat().st_size == expected:
                    continue
            _download(url, target)
    if include_aggregate:
        aggregate = root / "data" / "aggregates" / PHYSICAL_AGGREGATE_FILENAME
        if not aggregate.exists():
            _download(_AGGREGATE_URL, aggregate)
    return root


def load_skillcorner_physical_aggregate(
    root: str | Path = DEFAULT_OPENDATA_ROOT,
) -> pd.DataFrame:
    """Load SkillCorner's published A-League season Physical aggregate."""

    path = Path(root) / "data" / "aggregates" / PHYSICAL_AGGREGATE_FILENAME
    if not path.exists():
        alternate = Path(root) / "aggregates" / PHYSICAL_AGGREGATE_FILENAME
        path = alternate if alternate.exists() else path
    if not path.exists():
        raise FileNotFoundError(
            f"No {PHYSICAL_AGGREGATE_FILENAME} under {root}. "
            "Run download_skillcorner_opendata(..., include_tracking=False)."
        )
    return pd.read_csv(path, low_memory=False)


def resolve_match_files(
    match_id: int | str,
    root: str | Path | None = None,
) -> dict[str, Path | None]:
    """Locate one match's files, tolerating the several cache layouts in use.

    Honours ``$SKILLCORNER_ROOT`` when ``root`` is not given, and searches both
    the ``data/matches/<id>/`` layout written by
    :func:`download_skillcorner_opendata` and the flat layout the repo's other
    tutorials cache into.
    """

    match_id = str(match_id)
    base = Path(root) if root is not None else Path(
        os.environ.get(SKILLCORNER_ROOT_ENV) or DEFAULT_OPENDATA_ROOT
    )
    candidates = [
        base / "data" / "matches" / match_id,
        base / "matches" / match_id,
        base / match_id,
        base,
    ]
    found: dict[str, Path | None] = {}
    for key, suffix in (
        ("match", "match.json"),
        ("tracking", TRACKING_SUFFIX),
        ("dynamic_events", "dynamic_events.csv"),
        ("phases_of_play", "phases_of_play.csv"),
    ):
        found[key] = None
        for directory in candidates:
            path = directory / f"{match_id}_{suffix}"
            if path.exists():
                found[key] = path
                break
    if found["match"] is None:
        raise FileNotFoundError(
            f"No {match_id}_match.json under {base}. Run "
            "physical.skillcorner_tracking.download_skillcorner_opendata() first, "
            f"or set ${SKILLCORNER_ROOT_ENV}."
        )
    return found


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _string(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def skillcorner_ball_state(
    ball_data: dict[str, Any] | None = None,
    owning_team_id: Any | None = None,
) -> str:
    """Return the Open Data BIP state from TIP/OTIP ownership.

    ``ball_data`` remains an accepted argument for compatibility with callers
    of the old helper; the public physical definition uses the owning team.
    """

    return "alive" if owning_team_id is not None and not pd.isna(owning_team_id) else "dead"


def _appearance_seconds(
    player: dict[str, Any] | None, field: str, first_frame: int, fps: float
) -> float | None:
    total = ((player or {}).get("playing_time") or {}).get("total") or {}
    frame = _number(total.get(field))
    if frame is None:
        return None
    return max(0.0, (frame - first_frame) / fps)


def _playing_time_ms(player: dict[str, Any] | None) -> float | None:
    total = ((player or {}).get("playing_time") or {}).get("total") or {}
    minutes = _number(total.get("minutes_played"))
    return minutes * 60_000.0 if minutes is not None else None


def _xy(
    raw_x: Any, raw_y: Any, raw_length: float, raw_width: float, flip: bool
) -> tuple[float | None, float | None]:
    """Rescale provider metres onto the canonical pitch, flipping if needed."""

    x = _number(raw_x)
    y = _number(raw_y)
    if x is not None and raw_length > 0:
        x = x * (PITCH_X / raw_length)
    if y is not None and raw_width > 0:
        y = y * (PITCH_Y / raw_width)
    if flip:
        x = None if x is None else -x
        y = None if y is None else -y
    return x, y


# ---------------------------------------------------------------------------
# metadata + lineup
# ---------------------------------------------------------------------------


def _resolve_position_groups(
    raw_group: pd.Series, raw_position: pd.Series
) -> pd.Series:
    """Resolve one published position group per player from two provider fields.

    SkillCorner's two role fields disagree in three ways that all have to be
    handled, or players silently fall out of every position-group ranking:

    * ``position_group`` is ``"Wide Attacker"`` where the published group is
      ``"Winger"``, and ``"SUB"`` for unused substitutes.
    * ``position_group`` is ``"Other"`` for **goalkeepers**, whose real role is
      only visible in ``position`` (``"GK"``). An explicit goalkeeper signal must
      therefore always win -- misclassifying a keeper as an outfielder would drag
      every outfield distance benchmark down.
    * ``"Other"`` carries no information, so it must never beat a usable role.
    """

    group = raw_group.map(normalize_position_group)
    role = raw_position.map(normalize_position_group)

    def clean(value: object) -> str | None:
        # pd.NA must not reach a comparison: `pd.NA == "x"` is itself NA and
        # raises in a boolean context.
        if value is None or pd.isna(value):
            return None
        return value if value in POSITION_GROUPS else None

    resolved: list[str | None] = []
    for group_value, role_value in zip(group.map(clean), role.map(clean)):
        if "Goalkeeper" in (group_value, role_value):
            resolved.append("Goalkeeper")
        else:
            resolved.append(group_value or role_value)
    return pd.Series(resolved, index=raw_group.index, dtype="string")


def load_skillcorner_bundle(
    match_json: str | Path,
    *,
    season: str | int | None = None,
) -> ProviderBundle:
    """Build canonical match metadata and a lineup table from ``match.json``."""

    raw = _read_json(match_json)
    metadata = SkillcornerDataPreprocessor.extract_match_metadata(raw)
    lineup = SkillcornerDataPreprocessor.load_lineup_data(raw, metadata).copy()

    match_id = _string(metadata.get("match_id"))
    season_name = _string(season) or _string(metadata.get("season_name"))
    source_fps = _number(metadata.get("source_fps")) or 10.0
    if source_fps <= 0:
        source_fps = 10.0

    period_map = SkillcornerDataPreprocessor._build_period_frame_map(raw)
    starts = [
        int(value["start_frame"])
        for value in period_map.values()
        if value.get("start_frame") is not None
    ]
    if not starts:
        raise ValueError(f"{match_json} has no period start frames")
    first_frame = min(starts)

    raw_players = {_string(p.get("id")): p for p in raw.get("players") or []}

    lineup["provider"] = "skillcorner"
    lineup["season"] = season_name
    lineup["match_id"] = match_id
    raw_position = lineup.get("playing_position", pd.Series(pd.NA, index=lineup.index))
    raw_group = lineup.get("player_position_group", pd.Series(pd.NA, index=lineup.index))
    lineup["position"] = raw_position
    lineup["position_group"] = _resolve_position_groups(raw_group, raw_position)

    lineup["appearance_start_seconds"] = lineup["player_id"].map(
        lambda pid: _appearance_seconds(
            raw_players.get(_string(pid)), "start_frame", first_frame, source_fps
        )
    )
    lineup["appearance_end_seconds"] = lineup["player_id"].map(
        lambda pid: _appearance_seconds(
            raw_players.get(_string(pid)), "end_frame", first_frame, source_fps
        )
    )
    lineup["provider_playing_time_ms"] = lineup["player_id"].map(
        lambda pid: _playing_time_ms(raw_players.get(_string(pid)))
    )
    for column in _LINEUP_COLUMNS:
        if column not in lineup.columns:
            lineup[column] = pd.NA
    lineup = lineup.loc[:, list(_LINEUP_COLUMNS)].reset_index(drop=True)
    lineup["player_id"] = lineup["player_id"].map(_string).astype("string")
    lineup["team_id"] = lineup["team_id"].map(_string).astype("string")

    pitch_length = _number(metadata.get("pitch_length")) or PITCH_X
    pitch_width = _number(metadata.get("pitch_width")) or PITCH_Y

    match_metadata = {
        "provider": "skillcorner",
        "vendor_name": "SkillCorner",
        "season": season_name,
        "competition_name": metadata.get("competition_name"),
        "match_id": match_id,
        "kickoff_time": metadata.get("kickoff_time"),
        "home_team_id": _string(metadata.get("home_team_id")),
        "home_team_name": metadata.get("home_team_name"),
        "away_team_id": _string(metadata.get("away_team_id")),
        "away_team_name": metadata.get("away_team_name"),
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,
        "source_fps": source_fps,
        "ball_state_definition": BALL_STATE_DEFINITION,
        "coordinate_system": "metres_centered_home_attacks_positive_x",
    }
    context = {
        "raw_metadata": raw,
        "period_frame_map": period_map,
        "first_frame": first_frame,
        "source_fps": source_fps,
        "play_direction": metadata.get("play_direction")
        if isinstance(metadata.get("play_direction"), dict)
        else {},
        "raw_pitch_length": pitch_length,
        "raw_pitch_width": pitch_width,
        "home_team_id": match_metadata["home_team_id"],
        "away_team_id": match_metadata["away_team_id"],
    }
    return ProviderBundle(match_metadata, lineup, context)


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------


def canonical_tracking_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a dtype-stable canonical tracking frame from accumulated rows."""

    frame = pd.DataFrame(rows, columns=list(CANONICAL_TRACKING_COLUMNS))
    for column in ("provider", "season", "match_id", "period", "source_timestamp",
                   "object_id", "player_id", "team_id", "home_away",
                   "ball_state", "ball_owning_team_id"):
        frame[column] = frame[column].astype("string")
    for column in ("period_id", "frame_id", "source_frame_id"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("int64")
    for column in ("timestamp", "match_seconds", "x", "y", "z"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in ("ball", "is_detected"):
        frame[column] = frame[column].astype("boolean")
    return frame


def iter_skillcorner_tracking(
    tracking_path: str | Path,
    bundle: ProviderBundle,
    *,
    chunk_frames: int = 2_000,
    max_frames: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Load football-cdf's long tracking table and emit canonical chunks.

    JSON parsing, coordinate rescaling, period orientation and undetected
    outlier masking are owned by :class:`SkillcornerDataPreprocessor`. The only
    provider-specific post-processing here is the Open Data BIP convention:
    ``ball_state`` is rewritten from the frame's TIP/OTIP owning team and
    joined to player rows by ``(period_id, frame_id)``.
    """

    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    context = bundle.context
    metadata = bundle.match_metadata
    fps = float(context["source_fps"])
    first_frame = int(context["first_frame"])
    tracking = SkillcornerDataPreprocessor.load_tracking_long_data(
        tracking_path,
        context["raw_metadata"],
        bundle.lineup,
        metadata,
        fps,
    ).copy()
    if tracking.empty:
        return

    tracking["period_id"] = pd.to_numeric(tracking["period_id"], errors="coerce").astype("int64")
    tracking["frame_id"] = pd.to_numeric(tracking["frame_id"], errors="coerce").astype("int64")
    tracking["source_frame_id"] = tracking["frame_id"]
    tracking["period"] = tracking["period_id"].map(
        lambda value: CDF_PERIOD_MAP.get(int(value), f"period_{value}")
    )
    tracking["provider"] = "skillcorner"
    tracking["season"] = metadata.get("season")
    tracking["match_id"] = metadata.get("match_id")
    tracking["match_seconds"] = (tracking["frame_id"] - first_frame) / fps
    tracking["source_timestamp"] = tracking.get("utc_timestamp", pd.NA).astype("string")

    dimensions = bundle.lineup.set_index("player_id")[["team_id", "home_away"]]
    player_ids = tracking["player_id"].astype("string")
    tracking["team_id"] = player_ids.map(dimensions["team_id"])
    tracking["home_away"] = player_ids.map(dimensions["home_away"])

    # football-cdf preserves the raw owner on the ball row. Join it to every
    # player row at the same frame, then define BIP as TIP or OTIP.
    ball_owner = (
        tracking.loc[
            tracking["ball"].fillna(False),
            ["period_id", "frame_id", "ball_owning_team_id"],
        ]
        .drop_duplicates(["period_id", "frame_id"])
        .set_index(["period_id", "frame_id"])["ball_owning_team_id"]
    )
    frame_keys = pd.MultiIndex.from_frame(tracking[["period_id", "frame_id"]])
    owner = pd.Series(ball_owner.reindex(frame_keys).to_numpy(), index=tracking.index)
    tracking["ball_owning_team_id"] = owner.astype("string")
    tracking["ball_state"] = tracking["ball_owning_team_id"].notna().map(
        {True: "alive", False: "dead"}
    )

    if max_frames is not None:
        selected = tracking[["period_id", "frame_id"]].drop_duplicates(
            ignore_index=True
        ).head(max_frames)
        tracking = tracking.merge(
            selected, on=["period_id", "frame_id"], how="inner"
        )

    tracking = tracking.sort_values(
        ["period_id", "frame_id", "ball", "object_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    tracking = tracking.loc[:, [
        column for column in CANONICAL_TRACKING_COLUMNS if column in tracking
    ]]

    # Chunk by tracking frames, not rows. This preserves the old iterator API
    # while allowing football-cdf to own file parsing.
    keys = tracking[["period_id", "frame_id"]].drop_duplicates(ignore_index=True)
    for start in range(0, len(keys), chunk_frames):
        selected = keys.iloc[start : start + chunk_frames]
        chunk = tracking.merge(
            selected, on=["period_id", "frame_id"], how="inner"
        )
        yield canonical_tracking_frame(chunk.to_dict("records"))


# ---------------------------------------------------------------------------
# dynamic events / phases of play
# ---------------------------------------------------------------------------


def load_dynamic_events(
    path: str | Path,
    *,
    event_types: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load a Dynamic Events CSV, optionally keeping only some ``event_type``s.

    Pass ``event_types=("off_ball_run",)`` for the phase-of-play running profile.
    """

    events = pd.read_csv(path, low_memory=False)
    if event_types is not None:
        events = events.loc[events["event_type"].isin(event_types)].reset_index(drop=True)
    return events


def load_phases_of_play(path: str | Path) -> pd.DataFrame:
    """Load a phases-of-play CSV (one row per team-possession phase)."""

    return pd.read_csv(path, low_memory=False)
