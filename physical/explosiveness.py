"""Transparent reconstructions of time-to-speed physical metrics.

The public SkillCorner material explains the *concept* of Time to HSR/Sprint
and post-change-of-direction (COD) time, but not the proprietary V3 detector.
This module therefore does not claim to reproduce V3.  It provides a fully
specified reconstruction from a prepared 10 Hz tracking frame so that the
choice of onsets, turn angle and eligibility is inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .definitions import KMH_TO_MPS, PhysicalConfig
from .physical_features import find_efforts


@dataclass(frozen=True)
class ReconstructedExplosivenessConfig:
    """Published and analytical choices for reconstructed time-to-speed work."""

    walk_max_kmh: float = 9.0
    hsr_kmh: float = 20.0
    sprint_kmh: float = 25.0
    acceleration_mps2: float = 3.0
    acceleration_min_seconds: float = 0.7
    lookback_seconds: float = 3.0
    max_time_seconds: float = 5.0
    cod_min_angle_degrees: float = 60.0
    cod_heading_window_seconds: float = 0.5
    cod_cooldown_seconds: float = 1.0
    cod_min_speed_kmh: float = 9.0
    bip_min_share: float = 0.5


def _crossing_time(
    time: np.ndarray, speed: np.ndarray, start: int, stop: int, threshold: float
) -> tuple[float, int] | None:
    """First upward threshold crossing in a half-open interval, interpolated."""

    for index in range(max(start + 1, 1), stop):
        previous, current = speed[index - 1], speed[index]
        if not (np.isfinite(previous) and np.isfinite(current)):
            continue
        if previous < threshold <= current:
            fraction = (threshold - previous) / max(current - previous, 1e-12)
            return float(time[index - 1] + fraction * (time[index] - time[index - 1])), index
    return None


def _last_below(
    time: np.ndarray, speed: np.ndarray, segment: np.ndarray, index: int,
    threshold: float, lookback_seconds: float,
) -> int | None:
    """Last sample below a threshold before ``index`` in the same segment."""

    lower_time = time[index] - lookback_seconds
    candidates = np.flatnonzero(
        (segment[: index + 1] == segment[index])
        & (time[: index + 1] >= lower_time)
        & np.isfinite(speed[: index + 1])
        & (speed[: index + 1] < threshold)
    )
    return int(candidates[-1]) if candidates.size else None


def _bip_share(bip: np.ndarray, first: int, last: int) -> float:
    return float(np.asarray(bip[first : last + 1], dtype=float).mean())


def _candidate_row(
    frame: pd.DataFrame,
    *,
    family: str,
    target_name: str,
    onset_index: int,
    target_index: int,
    onset_time: float,
    target_time: float,
    source_index: int,
    cod_angle_degrees: float | None,
    bip_min_share: float,
) -> dict[str, object]:
    bip = frame["is_bip"].to_numpy(dtype=bool)
    share = _bip_share(bip, onset_index, target_index)
    row = frame.iloc[target_index]
    return {
        "match_id": str(row["match_id"]),
        "player_id": str(row["player_id"]),
        "player_name": row.get("player_name", pd.NA),
        "team_name": row.get("team_name", row.get("team", pd.NA)),
        "position_group": row.get("position_group", pd.NA),
        "period_id": row["period_id"],
        "family": family,
        "target": target_name,
        "time_seconds": float(target_time - onset_time),
        "onset_timestamp": float(onset_time),
        "target_timestamp": float(target_time),
        "bip_share": share,
        "bip_attributed": bool(share >= bip_min_share),
        "source_timestamp": float(frame.iloc[source_index]["timestamp"]),
        "cod_angle_degrees": cod_angle_degrees,
    }


def _circular_mean(angle: np.ndarray) -> float:
    return float(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()))


def _angle_difference(a: float, b: float) -> float:
    return float(np.degrees(np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))))


def _cod_candidates(
    frame: pd.DataFrame, config: ReconstructedExplosivenessConfig, dt: float
) -> list[tuple[int, float]]:
    """Return non-overlapping sharp turns from stable pre/post velocity headings."""

    vx = frame["vx"].to_numpy(dtype=float)
    vy = frame["vy"].to_numpy(dtype=float)
    speed = frame["speed_mps"].to_numpy(dtype=float)
    segment = frame["segment_id"].to_numpy()
    valid = frame["valid_interval"].to_numpy(dtype=bool)
    minimum_speed = config.cod_min_speed_kmh * KMH_TO_MPS
    # Prepared input is a uniform 10 Hz grid. The defensive fallback keeps the
    # detector useful in synthetic unit tests.
    if not np.isfinite(dt) or dt <= 0 or dt > 1:
        dt = .1
    window = max(2, round(config.cod_heading_window_seconds / dt))
    cooldown = max(1, round(config.cod_cooldown_seconds / dt))
    heading = np.arctan2(vy, vx)
    found: list[tuple[int, float]] = []
    for index in range(window, len(frame) - window):
        if segment[index - window] != segment[index + window]:
            continue
        pre = slice(index - window, index)
        post = slice(index + 1, index + 1 + window)
        if not (valid[pre].all() and valid[post].all()):
            continue
        if speed[pre].min() < minimum_speed or speed[post].min() < minimum_speed:
            continue
        angle = _angle_difference(_circular_mean(heading[pre]), _circular_mean(heading[post]))
        if angle >= config.cod_min_angle_degrees:
            found.append((index, angle))
    # A sustained curved path gives several adjacent candidates.  Retain its
    # sharpest point, with a deterministic earliest-index tie break.
    if not found:
        return []
    # Adjacent candidate samples belong to the same curved movement.  Cluster
    # them in time and retain the sharpest point.  This is O(n), unlike global
    # non-maximum suppression which becomes prohibitive across a season.
    found.sort(key=lambda item: item[0])
    selected: list[tuple[int, float]] = []
    cluster: list[tuple[int, float]] = [found[0]]
    for candidate in found[1:]:
        if candidate[0] - cluster[-1][0] < cooldown:
            cluster.append(candidate)
        else:
            selected.append(min(cluster, key=lambda item: (-item[1], item[0])))
            cluster = [candidate]
    selected.append(min(cluster, key=lambda item: (-item[1], item[0])))
    return selected


def extract_reconstructed_efforts(
    frame: pd.DataFrame,
    physical_config: PhysicalConfig | None = None,
    config: ReconstructedExplosivenessConfig | None = None,
) -> pd.DataFrame:
    """Extract transparent time-to-speed and post-COD candidates for one match.

    ``frame`` must be the output of :func:`physical.kinematics.prepare_kinematics`.
    Accelerative candidates use an acceleration effort (>3 m/s² for >=0.7 s),
    locate the last <9 km/h sample within three seconds, then measure the
    interpolated time from the 9 km/h upward crossing to HSR/Sprint.  COD
    candidates use the same target crossings after a >=60° stable-heading turn.
    """

    physical_config = physical_config or PhysicalConfig()
    config = config or ReconstructedExplosivenessConfig(
        bip_min_share=physical_config.effort_bip_min_share
    )
    required = {"match_id", "player_id", "period_id", "timestamp", "segment_id", "speed_mps", "acceleration_mps2", "is_bip", "valid_interval", "vx", "vy"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prepared kinematics missing: {missing}")
    if frame.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for _, player in frame.groupby(["player_id", "period_id"], sort=False):
        player = player.reset_index(drop=True)
        time = player["timestamp"].to_numpy(dtype=float)
        speed = player["speed_mps"].to_numpy(dtype=float)
        accel = player["acceleration_mps2"].to_numpy(dtype=float)
        segment = player["segment_id"].to_numpy()
        valid = player["valid_interval"].to_numpy(dtype=bool)
        walk = config.walk_max_kmh * KMH_TO_MPS
        high_accel = valid & np.isfinite(accel) & (accel > config.acceleration_mps2)
        first, _, _ = find_efforts(
            high_accel, segment, dt=physical_config.dt,
            min_duration=config.acceleration_min_seconds,
            duration_rule=physical_config.effort_duration_rule,
        )
        targets = (("hsr", config.hsr_kmh * KMH_TO_MPS), ("sprint", config.sprint_kmh * KMH_TO_MPS))
        for candidate in first:
            below = _last_below(time, speed, segment, int(candidate), walk, config.lookback_seconds)
            if below is None:
                continue
            onset = _crossing_time(time, speed, below, len(player), walk)
            if onset is None:
                continue
            onset_time, onset_index = onset
            for target_name, threshold in targets:
                crossing = _crossing_time(time, speed, onset_index, len(player), threshold)
                if crossing is None:
                    continue
                target_time, target_index = crossing
                if segment[target_index] != segment[onset_index] or target_time - onset_time > config.max_time_seconds:
                    continue
                records.append(_candidate_row(player, family="time_to_speed", target_name=target_name, onset_index=onset_index, target_index=target_index, onset_time=onset_time, target_time=target_time, source_index=int(candidate), cod_angle_degrees=None, bip_min_share=config.bip_min_share))
        for candidate, angle in _cod_candidates(player, config, physical_config.dt):
            # The time starts at the first >=9 km/h sample after the turn.
            onset_index = next((i for i in range(candidate, len(player)) if segment[i] == segment[candidate] and speed[i] >= walk), None)
            if onset_index is None:
                continue
            onset_time = float(time[onset_index])
            for target_name, threshold in targets:
                crossing = _crossing_time(time, speed, onset_index, len(player), threshold)
                if crossing is None:
                    continue
                target_time, target_index = crossing
                if segment[target_index] != segment[onset_index] or target_time - onset_time > config.max_time_seconds:
                    continue
                records.append(_candidate_row(player, family="post_cod_time_to_speed", target_name=target_name, onset_index=onset_index, target_index=target_index, onset_time=onset_time, target_time=target_time, source_index=int(candidate), cod_angle_degrees=float(angle), bip_min_share=config.bip_min_share))
    return pd.DataFrame.from_records(records)
