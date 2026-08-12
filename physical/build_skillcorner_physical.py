"""Build a player-match physical metrics table from SkillCorner Open Data.

Runs one match at a time so a 90 MB tracking file is never held twice, and
writes a single tidy parquet plus a JSON manifest recording the definition
version and every eligibility choice, so a table can always be traced back to
the rules that produced it.

    python -m physical.build_skillcorner_physical --download
    python -m physical.build_skillcorner_physical --matches 1886347 1899585

Outputs (under ``tmp/data/physical_opendata/`` by default):

``player_matches.parquet``
    One row per (match, player) with raw volumes, per-60-BIP-minute rates,
    TIP/OTIP splits, PSV-99, and the coverage/quality columns.
``coverage.parquet``
    Per-match tracking quality: frame-level vs player-weighted ball-in-play,
    players observed per live/dead frame, and detected-vs-extrapolated share.
``manifest.json``
    Definition version, config, eligibility rules, and per-match row counts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from physical.definitions import PHYSICAL_DEFINITION_VERSION, PhysicalConfig
from physical.skillcorner_tracking import (
    OPENDATA_MATCH_IDS,
    download_skillcorner_opendata,
    iter_skillcorner_tracking,
    load_skillcorner_bundle,
    resolve_match_files,
)
from physical.physical_features import compute_player_match_metrics

DEFAULT_OUTPUT_DIR = Path("tmp/data/physical_opendata")
DEFAULT_SEASON = "2024/25"

# Open Data ships 10 matches, so a player can appear at most 10 times and the
# published Week 5 rule (>= 5 qualifying matches) would empty the table. Two
# performances is the smallest threshold that still excludes one-off cameos.
MINIMUM_PLAYED_MINUTES = 60.0
MINIMUM_MATCHES = 2

__all__ = [
    "MINIMUM_MATCHES",
    "MINIMUM_PLAYED_MINUTES",
    "add_group_percentiles",
    "build_match",
    "build_player_matches",
    "build_player_profiles",
    "match_coverage",
    "qualified_outfield",
]


def match_coverage(tracking: pd.DataFrame) -> dict[str, float]:
    """Summarise how completely one match was tracked.

    Two ball-in-play percentages fall out of the same file and must not be
    conflated: the **frame-weighted** share (one vote per frame, comparable to a
    published BIP figure) and the **player-observation-weighted** share (which is
    what the per-60-BIP-minute denominators are actually built from). They differ
    because the camera follows live play, so live frames carry more tracked
    players than dead ones.
    """

    ball = tracking.loc[tracking["ball"].fillna(False)]
    players = tracking.loc[~tracking["ball"].fillna(False)]
    live_frames = int((ball["ball_state"] == "alive").sum())
    dead_frames = int(len(ball) - live_frames)
    live_rows = int((players["ball_state"] == "alive").sum())
    detected = players["is_detected"].fillna(False)

    return {
        "match_id": str(tracking["match_id"].iloc[0]),
        "frames": int(len(ball)),
        "player_rows": int(len(players)),
        "bip_frame_weighted_pct": 100.0 * live_frames / len(ball) if len(ball) else float("nan"),
        "bip_player_weighted_pct": 100.0 * live_rows / len(players) if len(players) else float("nan"),
        "players_per_live_frame": live_rows / live_frames if live_frames else float("nan"),
        "players_per_dead_frame": (len(players) - live_rows) / dead_frames if dead_frames else float("nan"),
        # Every live frame carries the full XI on both sides, but a large share of
        # those positions are model estimates rather than detections -- the real
        # quality limit of broadcast tracking is extrapolation, not missing rows.
        "detected_share": float(detected.mean()) if len(players) else float("nan"),
        "detected_share_live": float(detected[players["ball_state"] == "alive"].mean())
        if live_rows else float("nan"),
    }


def build_match(
    match_id: int | str,
    *,
    root: str | Path | None = None,
    season: str = DEFAULT_SEASON,
    config: PhysicalConfig | None = None,
    chunk_frames: int = 20_000,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute one match's player-match metrics and its coverage summary."""

    files = resolve_match_files(match_id, root)
    if files["tracking"] is None:
        raise FileNotFoundError(
            f"No tracking file for match {match_id}. Re-run with --download "
            "(tracking is Git-LFS backed and is skipped by the other tutorials' "
            "downloaders)."
        )
    bundle = load_skillcorner_bundle(files["match"], season=season)
    tracking = pd.concat(
        iter_skillcorner_tracking(files["tracking"], bundle, chunk_frames=chunk_frames),
        ignore_index=True,
    )
    coverage = match_coverage(tracking)
    player_matches = compute_player_match_metrics(
        tracking, bundle.lineup, config or PhysicalConfig()
    )
    player_matches = player_matches.merge(
        bundle.lineup[["player_id", "team_id", "player_name", "uniform_number"]]
        .rename(columns={"team_id": "lineup_team_id"}),
        on="player_id",
        how="left",
        suffixes=("", "_lineup"),
    )
    player_matches["season"] = season
    player_matches["competition"] = bundle.match_metadata.get("competition_name")
    return player_matches, coverage


def build_player_matches(
    match_ids: list[int | str] | None = None,
    *,
    root: str | Path | None = None,
    season: str = DEFAULT_SEASON,
    config: PhysicalConfig | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the player-match and coverage tables across matches."""

    ids = match_ids if match_ids is not None else OPENDATA_MATCH_IDS
    frames: list[pd.DataFrame] = []
    coverages: list[dict[str, float]] = []
    for index, match_id in enumerate(ids, start=1):
        player_matches, coverage = build_match(
            match_id, root=root, season=season, config=config
        )
        frames.append(player_matches)
        coverages.append(coverage)
        if verbose:
            print(
                f"[{index}/{len(ids)}] {match_id}: {len(player_matches)} player-matches, "
                f"BIP {coverage['bip_frame_weighted_pct']:.1f}% (frames) / "
                f"{coverage['bip_player_weighted_pct']:.1f}% (player rows), "
                f"detected {100 * coverage['detected_share']:.0f}%",
                flush=True,
            )
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(coverages),
    )


def qualified_outfield(
    player_matches: pd.DataFrame,
    *,
    minimum_played_minutes: float = MINIMUM_PLAYED_MINUTES,
    minimum_matches: int = MINIMUM_MATCHES,
) -> pd.DataFrame:
    """Keep outfield performances that clear the open-data eligibility rules.

    Coverage is deliberately **not** a filter here. On this feed a low observed
    share is dead-ball time rather than lost live play, so gating on it would
    drop good performances; it is reported in ``coverage.parquet`` instead.
    """

    eligible = player_matches.loc[
        (player_matches["played_minutes"] >= minimum_played_minutes)
        & (player_matches["position_group"].notna())
        & (player_matches["position_group"] != "Goalkeeper")
    ].copy()
    counts = eligible.groupby("player_id")["match_id"].nunique()
    keep = counts[counts >= minimum_matches].index
    return eligible.loc[eligible["player_id"].isin(keep)].reset_index(drop=True)


#: Volumes that get pooled into a season rate, and the exposure each is divided by.
_PROFILE_SPECS = (
    ("bip_minutes", 60.0, "_p60_bip", (
        "total_distance_m", "running_distance_m", "hsr_distance_m", "sprint_distance_m",
        "high_intensity_distance_m", "sprint_count", "high_intensity_count",
        "high_acceleration_count", "high_deceleration_count",
    )),
    ("tip_minutes", 30.0, "_p30_tip", (
        "distance_tip_m", "high_intensity_distance_tip_m", "sprint_count_tip",
    )),
    ("otip_minutes", 30.0, "_p30_otip", (
        "distance_otip_m", "high_intensity_distance_otip_m", "sprint_count_otip",
    )),
)
#: Peak measures are not volumes and must never be divided by exposure.
_PROFILE_PEAKS = ("psv99_kmh", "max_speed_kmh")


def build_player_profiles(qualified: pd.DataFrame) -> pd.DataFrame:
    """Pool a player's qualifying performances into one season profile.

    A rate is ``sum(volume) / sum(exposure)``, **not** the mean of per-match rates:
    averaging rates would weight a 62-minute appearance the same as a full match.
    Peak measures are not averaged. The public SkillCorner Open Data workflow
    uses the strongest recorded value for each player, so PSV-99 and top speed
    are the maximum qualifying performance values.

    Players are grouped by ``player_id`` alone and their position group is taken as
    the modal one. Grouping by position as well would split anyone who changed
    role between matches into two half-profiles -- which silently produced
    "1 match" rows for six players even though every one of them had cleared the
    two-match rule.
    """

    volumes = [c for _, _, _, cols in _PROFILE_SPECS for c in cols]
    exposures = [spec[0] for spec in _PROFILE_SPECS]
    grouped = qualified.groupby("player_id", observed=True)

    totals = grouped[[c for c in volumes if c in qualified.columns]].sum()
    exposure = grouped[[c for c in exposures if c in qualified.columns]].sum()

    profiles = pd.DataFrame(index=totals.index)
    profiles["player_name"] = grouped["player_name"].first()
    profiles["position_group"] = grouped["position_group"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA
    )
    profiles["matches"] = grouped["match_id"].nunique()
    profiles["played_minutes"] = grouped["played_minutes"].sum()
    for exposure_col, window, suffix, columns in _PROFILE_SPECS:
        if exposure_col not in exposure.columns:
            continue
        denominator = exposure[exposure_col].replace(0, pd.NA)
        profiles[exposure_col] = exposure[exposure_col]
        for column in columns:
            if column in totals.columns:
                profiles[f"{column}{suffix}"] = totals[column] * window / denominator
    for column in _PROFILE_PEAKS:
        if column in qualified.columns:
            profiles[column] = grouped[column].max()
    return profiles.reset_index()


def add_group_percentiles(
    profiles: pd.DataFrame,
    metrics: list[str],
    *,
    group_column: str | None = "position_group",
) -> pd.DataFrame:
    """Add ``<metric>_pct`` percentile ranks, optionally within position group."""

    out = profiles.copy()
    frame = out.groupby(group_column, observed=True) if group_column else out
    for metric in metrics:
        if metric not in out.columns:
            continue
        ranks = (
            out.groupby(group_column, observed=True)[metric] if group_column else out[metric]
        )
        out[f"{metric}_pct"] = ranks.rank(pct=True) * 100
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", nargs="*", default=None, help="match ids (default: all 10)")
    parser.add_argument("--root", default=None, help="Open Data root (default: $SKILLCORNER_ROOT or tmp/data/skillcorner_opendata)")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--estimator", default=None, help="override the velocity estimator")
    parser.add_argument("--download", action="store_true", help="fetch missing files first (~915 MB with tracking)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ids = args.matches or OPENDATA_MATCH_IDS
    if args.download:
        download_skillcorner_opendata(
            args.root or "tmp/data/skillcorner_opendata", match_ids=list(ids)
        )
    config = PhysicalConfig(estimator=args.estimator) if args.estimator else PhysicalConfig()

    player_matches, coverage = build_player_matches(
        list(ids), root=args.root, season=args.season, config=config
    )
    qualified = qualified_outfield(player_matches)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    player_matches.to_parquet(out_dir / "player_matches.parquet", index=False)
    coverage.to_parquet(out_dir / "coverage.parquet", index=False)

    manifest = {
        "physical_definition_version": PHYSICAL_DEFINITION_VERSION,
        "source": "SkillCorner Open Data (github.com/SkillCorner/opendata)",
        "bip_definition": "TIP + OTIP (minutes_full_tip + minutes_full_otip)",
        "season": args.season,
        "match_ids": [str(m) for m in ids],
        "config": {k: v for k, v in asdict(config).items() if not isinstance(v, dict)},
        "speed_thresholds_kmh": {
            "running_min": config.thresholds.running_min_kmh,
            "hsr_min": config.thresholds.hsr_min_kmh,
            "sprint_min": config.thresholds.sprint_min_kmh,
        },
        "eligibility": {
            "minimum_played_minutes": MINIMUM_PLAYED_MINUTES,
            "minimum_matches": MINIMUM_MATCHES,
            "population": "outfield",
            "coverage_is_a_filter": False,
        },
        "rows": {
            "player_matches": int(len(player_matches)),
            "qualified_outfield": int(len(qualified)),
            "players_qualified": int(qualified["player_id"].nunique()),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"\nwrote {out_dir}/player_matches.parquet  "
        f"({len(player_matches)} rows, {len(qualified)} qualified outfield "
        f"from {qualified['player_id'].nunique()} players)"
    )


if __name__ == "__main__":
    main()
