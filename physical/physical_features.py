"""Per player-match physical features: bands, discrete efforts, PSV-99, phases.

The unit of observation is the **player-match performance**, matching the
article's sample definition. Every volume is accumulated over the public
SkillCorner BIP proxy (TIP + OTIP) samples and paired with the corresponding
TIP+OTIP seconds, so a rate is never a function of frame count.

Discrete efforts are found on the *continuous* speed signal and only then
attributed to ball-in-play.  Masking frames before detection would split a
single physiological effort at every possession change, which is the dominant
source of over-counting.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .definitions import (
    EFFORT_DURATION_SPAN,
    KMH_TO_MPS,
    MPS_TO_KMH,
    PLAYED_TIME_SOURCE_APPEARANCE,
    PLAYED_TIME_SOURCE_OBSERVED,
    PLAYED_TIME_SOURCE_PROVIDER,
    SECONDS_PER_HOUR,
    PhysicalConfig,
)
from .kinematics import bip_source, prepare_kinematics

_EPS = 1e-9
_SPEED_EPS_MPS = 1e-4

#: Additive quantities that carry a ``_p60_bip`` rate.
VOLUME_COLUMNS = (
    "total_distance_m",
    "running_distance_m",
    "hsr_distance_m",
    "sprint_distance_m",
    "high_intensity_distance_m",
    "hir_distance_m",
    "walking_jogging_distance_m",
    "running_count",
    "hsr_count",
    "sprint_count",
    "high_intensity_count",
    "hir_count",
    "high_acceleration_count",
    "high_deceleration_count",
    "distance_tip_m",
    "distance_otip_m",
    "high_intensity_distance_tip_m",
    "high_intensity_distance_otip_m",
    "sprint_count_tip",
    "sprint_count_otip",
    "high_intensity_count_tip",
    "high_intensity_count_otip",
)


# --------------------------------------------------------------------------- #
# discrete effort detection
# --------------------------------------------------------------------------- #
def find_efforts(
    mask: np.ndarray,
    segment_id: np.ndarray,
    *,
    dt: float,
    min_duration: float = 0.0,
    dip_tolerance: float = 0.0,
    duration_rule: str = "samples",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate discrete activities where ``mask`` holds.

    Args:
        mask: Above-threshold indicator per sample.
        segment_id: Segment label per sample.  An activity never spans a
            segment boundary, so a tracking gap always ends the activity and
            each side has to qualify on its own.
        dt: Sample interval in seconds.
        min_duration: Minimum above-threshold duration to qualify, inclusive.
        dip_tolerance: Above-threshold runs separated by a shorter
            sub-threshold dip merge into one activity.  ``0.0`` requires a
            strictly continuous run.
        duration_rule: ``"samples"`` counts ``n_above * dt`` (the time integral
            of the indicator); ``"span"`` counts ``(n - 1) * dt`` between the
            first and last above-threshold sample.  At 10 Hz the two differ by
            one sample, i.e. by a seventh of a 0.7 s requirement.

    Returns:
        ``(first_index, last_index, above_seconds)`` for the qualifying
        activities, with inclusive sample indices into the input arrays.
    """

    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    empty = np.empty(0, dtype=np.int64)
    if n == 0:
        return empty, empty, np.empty(0, dtype=float)

    boundary = np.zeros(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = (mask[1:] != mask[:-1]) | (segment_id[1:] != segment_id[:-1])
    run_start = np.flatnonzero(boundary)
    run_stop = np.append(run_start[1:], n)
    run_length = run_stop - run_start
    run_mask = mask[run_start]
    run_segment = np.asarray(segment_id)[run_start]

    active = np.flatnonzero(run_mask)
    if active.size == 0:
        return empty, empty, np.empty(0, dtype=float)

    # Chain consecutive above-runs across a single short sub-threshold dip.
    if dip_tolerance > 0 and active.size > 1:
        previous, following = active[:-1], active[1:]
        adjacent = (following == previous + 2) & (
            run_segment[following] == run_segment[previous]
        )
        dip_seconds = np.where(adjacent, run_length[np.minimum(previous + 1, len(run_length) - 1)] * dt, np.inf)
        merge = adjacent & (dip_seconds <= dip_tolerance + _EPS)
    else:
        merge = np.zeros(max(active.size - 1, 0), dtype=bool)

    chain = np.concatenate(([0], np.cumsum(~merge)))
    n_efforts = int(chain[-1]) + 1
    order = np.argsort(chain, kind="stable")
    first_run = active[order][np.searchsorted(chain[order], np.arange(n_efforts))]
    last_run = active[order][
        np.searchsorted(chain[order], np.arange(n_efforts), side="right") - 1
    ]

    above_samples = np.bincount(chain, weights=run_length[active], minlength=n_efforts)
    first_index = run_start[first_run]
    last_index = run_stop[last_run] - 1
    if duration_rule == EFFORT_DURATION_SPAN:
        duration = (last_index - first_index) * dt
    else:
        duration = above_samples * dt

    keep = duration >= min_duration - _EPS
    return (
        first_index[keep].astype(np.int64),
        last_index[keep].astype(np.int64),
        (above_samples[keep] * dt).astype(float),
    )


def _effort_share(
    flag: np.ndarray, first: np.ndarray, last: np.ndarray
) -> np.ndarray:
    """Share of each activity's samples for which ``flag`` holds."""

    if first.size == 0:
        return np.empty(0, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(np.asarray(flag, dtype=float))))
    covered = cumulative[last + 1] - cumulative[first]
    return covered / (last - first + 1)


def count_efforts(
    mask: np.ndarray,
    segment_id: np.ndarray,
    bip: np.ndarray,
    config: PhysicalConfig,
    *,
    threshold_seconds: float | None = None,
) -> int:
    """Number of qualifying activities attributed to ball-in-play."""

    first, last, _ = find_efforts(
        mask,
        segment_id,
        dt=config.dt,
        min_duration=(
            config.speed_activity_min_seconds
            if threshold_seconds is None
            else threshold_seconds
        ),
        dip_tolerance=config.effort_dip_tolerance_seconds,
        duration_rule=config.effort_duration_rule,
    )
    if first.size == 0:
        return 0
    share = _effort_share(bip, first, last)
    return int((share >= config.effort_bip_min_share - _EPS).sum())


def activity_peak_speeds(
    speed: np.ndarray,
    segment_id: np.ndarray,
    config: PhysicalConfig,
    *,
    min_kmh: float | None = None,
    min_duration: float = 0.0,
) -> np.ndarray:
    """Peak speed (m/s) of every activity above the PSV entry threshold."""

    entry = (config.psv_min_kmh if min_kmh is None else min_kmh) * KMH_TO_MPS
    finite = np.isfinite(speed)
    first, last, _ = find_efforts(
        finite & (speed > entry),
        segment_id,
        dt=config.dt,
        min_duration=min_duration,
        dip_tolerance=config.effort_dip_tolerance_seconds,
        duration_rule=config.effort_duration_rule,
    )
    if first.size == 0:
        return np.empty(0, dtype=float)
    filled = np.where(finite, speed, -np.inf)
    # ``reduceat`` requires every index to be strictly smaller than the input
    # length.  The sentinel makes the exclusive stop of a final-sample effort
    # (``last + 1 == len(speed)``) valid without changing any effort interval.
    padded = np.append(filled, -np.inf)
    return np.maximum.reduceat(
        padded, np.stack((first, last + 1), axis=1).ravel()
    )[::2]


def psv(
    speed: np.ndarray,
    segment_id: np.ndarray,
    config: PhysicalConfig,
    *,
    min_duration: float = 0.0,
) -> float:
    """Peak sprint velocity percentile, in km/h.

    The 99th percentile of the *per-activity peak* velocities of activities
    above ``config.psv_min_kmh``.  Peaks beyond ``config.psv_max_kmh`` are
    dropped as tracking artefacts before the percentile is taken.
    """

    peaks = activity_peak_speeds(speed, segment_id, config, min_duration=min_duration)
    peaks = peaks[np.isfinite(peaks) & (peaks <= config.psv_max_kmh * KMH_TO_MPS)]
    if peaks.size == 0:
        return float("nan")
    return float(np.percentile(peaks, config.psv_percentile) * MPS_TO_KMH)


def _frame_percentile(
    speed: np.ndarray, weights: np.ndarray, config: PhysicalConfig
) -> float:
    """Frame-level duration-weighted percentile, retained for sensitivity."""

    keep = (
        np.isfinite(speed)
        & np.isfinite(weights)
        & (weights > 0)
        & (speed > config.psv_min_kmh * KMH_TO_MPS)
        & (speed <= config.psv_max_kmh * KMH_TO_MPS)
    )
    if not keep.any():
        return float("nan")
    values, w = speed[keep], weights[keep]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(w[order])
    target = config.psv_percentile / 100.0 * cumulative[-1]
    return float(values[order][np.searchsorted(cumulative, target, side="left")] * MPS_TO_KMH)


# --------------------------------------------------------------------------- #
# player-match assembly
# --------------------------------------------------------------------------- #
def _player_match_row(
    frame: pd.DataFrame, config: PhysicalConfig, source: str
) -> dict[str, object]:
    if "effective_sample_hz" in frame:
        effective_hz = pd.to_numeric(
            frame["effective_sample_hz"], errors="coerce"
        ).dropna()
        if not effective_hz.empty and effective_hz.iloc[0] > 0:
            config = replace(config, sample_hz=float(effective_hz.iloc[0]))
    thresholds = config.thresholds
    # Distance and effort volumes use the plausibility-capped signal. PSV-99
    # is the public SkillCorner peak metric, so it uses the pre-cap signal and
    # applies its own 15–40 km/h activity range in ``psv``.
    speed = frame["speed_mps"].to_numpy(dtype=float)
    psv_speed = frame.get("raw_speed_mps", frame["speed_mps"]).to_numpy(dtype=float)
    accel = frame["acceleration_mps2"].to_numpy(dtype=float)
    step = frame["step_distance_m"].to_numpy(dtype=float)
    seconds = frame["dt"].to_numpy(dtype=float)
    segment = frame["segment_id"].to_numpy()
    bip = frame["is_bip"].to_numpy(dtype=bool)
    valid = frame["valid_interval"].to_numpy(dtype=bool)
    included = valid & bip

    running = (speed >= thresholds.running_min_mps - _SPEED_EPS_MPS) & (
        speed < thresholds.hsr_min_mps - _SPEED_EPS_MPS
    )
    hsr = (speed >= thresholds.hsr_min_mps - _SPEED_EPS_MPS) & (
        speed <= thresholds.sprint_min_mps + _SPEED_EPS_MPS
    )
    sprint = speed > thresholds.sprint_min_mps + _SPEED_EPS_MPS
    # High intensity is the single ``> 20`` km/h band.  For *distance* this is
    # identical to the HSR-plus-Sprint union; for *counts* it is not, because a
    # sprint accelerates up through the 20-25 km/h band and back down again.
    high_intensity = speed > thresholds.hsr_min_mps + _EPS
    low = valid & (speed < thresholds.running_min_mps - _EPS)

    def distance(mask: np.ndarray, extra: np.ndarray | None = None) -> float:
        selected = included & mask
        if extra is not None:
            selected = selected & extra
        return float(np.nansum(step[selected]))

    bip_seconds = float(np.nansum(seconds[included]))
    valid_seconds = float(np.nansum(seconds[valid]))
    observed_seconds = float(np.nansum(seconds[np.isfinite(seconds)]))

    owner = frame.get("ball_owning_team_id")
    team = frame.get("team_id")
    if owner is not None and team is not None:
        owner_values = pd.to_numeric(owner, errors="coerce").to_numpy(dtype=float)
        team_values = pd.to_numeric(team, errors="coerce").to_numpy(dtype=float)
        tip = bip & np.isfinite(owner_values) & (owner_values == team_values)
        otip = bip & np.isfinite(owner_values) & (owner_values != team_values)
    else:
        tip = np.zeros(len(frame), dtype=bool)
        otip = np.zeros(len(frame), dtype=bool)
    tip_seconds = float(np.nansum(seconds[tip]))
    otip_seconds = float(np.nansum(seconds[otip]))

    high_accel = np.isfinite(accel) & (accel > config.high_acceleration_mps2)
    high_decel = np.isfinite(accel) & (accel < -config.high_deceleration_mps2)

    hsr_count = count_efforts(hsr & valid, segment, bip, config)
    sprint_count = count_efforts(sprint & valid, segment, bip, config)
    hsr_count_tip = count_efforts(hsr & valid, segment, tip, config)
    hsr_count_otip = count_efforts(hsr & valid, segment, otip, config)
    sprint_count_tip = count_efforts(sprint & valid, segment, tip, config)
    sprint_count_otip = count_efforts(sprint & valid, segment, otip, config)
    counts = {
        "running_count": count_efforts(running & valid, segment, bip, config),
        "hsr_count": hsr_count,
        "sprint_count": sprint_count,
        "high_intensity_count": hsr_count + sprint_count,
        "high_acceleration_count": count_efforts(
            high_accel, segment, bip, config,
            threshold_seconds=config.high_acceleration_min_seconds,
        ),
        "high_deceleration_count": count_efforts(
            high_decel, segment, bip, config,
            threshold_seconds=config.high_acceleration_min_seconds,
        ),
        "sprint_count_tip": sprint_count_tip,
        "sprint_count_otip": sprint_count_otip,
        "high_intensity_count_tip": hsr_count_tip + sprint_count_tip,
        "high_intensity_count_otip": hsr_count_otip + sprint_count_otip,
    }

    first = frame.iloc[0]
    provider_playing_time_ms = pd.to_numeric(
        pd.Series([first.get("provider_playing_time_ms", np.nan)]),
        errors="coerce",
    ).iloc[0]
    appearance_start = pd.to_numeric(
        pd.Series([first.get("appearance_start_seconds", np.nan)]),
        errors="coerce",
    ).iloc[0]
    appearance_end = pd.to_numeric(
        pd.Series([first.get("appearance_end_seconds", np.nan)]),
        errors="coerce",
    ).iloc[0]
    if np.isfinite(provider_playing_time_ms) and provider_playing_time_ms > 0:
        played_seconds = float(provider_playing_time_ms) / 1_000.0
        played_source = PLAYED_TIME_SOURCE_PROVIDER
    elif (
        np.isfinite(appearance_start)
        and np.isfinite(appearance_end)
        and appearance_end > appearance_start
    ):
        played_seconds = float(appearance_end - appearance_start)
        played_source = PLAYED_TIME_SOURCE_APPEARANCE
    else:
        played_seconds = observed_seconds
        played_source = PLAYED_TIME_SOURCE_OBSERVED
    played_minutes = played_seconds / 60.0
    segment_count = int(pd.unique(segment).size)
    period_count = int(frame["period_id"].nunique())
    gap_count = max(segment_count - period_count, 0)

    row: dict[str, object] = {
        "match_id": first.get("match_id"),
        "player_id": first.get("player_id"),
        "team_id": first.get("team_id", pd.NA),
        "position": first.get("position", pd.NA),
        "position_group": first.get("position_group", pd.NA),
        "player_name": first.get("player_name", pd.NA),
        "estimator": config.estimator,
        "sample_count": int(len(frame)),
        "segment_count": segment_count,
        "gap_count": gap_count,
        "period_count": period_count,
        "played_seconds": played_seconds,
        "played_minutes": float(played_minutes),
        "played_time_source": played_source,
        "observed_seconds": observed_seconds,
        "valid_seconds": valid_seconds,
        "bip_seconds": bip_seconds,
        "bip_minutes": bip_seconds / 60.0,
        "tip_seconds": tip_seconds,
        "tip_minutes": tip_seconds / 60.0,
        "otip_seconds": otip_seconds,
        "otip_minutes": otip_seconds / 60.0,
        "provider_minutes_tip": first.get("minutes_tip", np.nan),
        "provider_minutes_otip": first.get("minutes_otip", np.nan),
        "detected_share": (
            float(frame["is_detected"].mean()) if "is_detected" in frame else np.nan
        ),
        "coverage_ratio": valid_seconds / observed_seconds if observed_seconds > 0 else np.nan,
        "played_tracking_coverage_ratio": (
            valid_seconds / played_seconds if played_seconds > 0 else np.nan
        ),
        "bip_coverage_ratio": bip_seconds / valid_seconds if valid_seconds > 0 else np.nan,
        "capped_speed_count": int(frame["capped_speed"].sum()),
        "capped_acceleration_count": int(frame["capped_acceleration"].sum()),
        "invalid_interval_count": int((~valid).sum()),
        "total_distance_all_m": float(np.nansum(step[valid])),
        "total_distance_m": distance(np.ones(len(frame), dtype=bool)),
        "walking_jogging_distance_m": distance(low),
        "running_distance_m": distance(running),
        "hsr_distance_m": distance(hsr),
        "sprint_distance_m": distance(sprint),
        "high_intensity_distance_m": distance(high_intensity),
        "distance_tip_m": float(np.nansum(step[valid & tip])),
        "distance_otip_m": float(np.nansum(step[valid & otip])),
        "high_intensity_distance_tip_m": float(np.nansum(step[valid & tip & high_intensity])),
        "high_intensity_distance_otip_m": float(np.nansum(step[valid & otip & high_intensity])),
        "max_speed_kmh": (
            float(np.nanmax(speed) * MPS_TO_KMH) if np.isfinite(speed).any() else np.nan
        ),
        "psv99_kmh": psv(psv_speed, segment, config),
        "psv99_activity_min07_kmh": psv(psv_speed, segment, config, min_duration=0.7),
        "psv99_frame_kmh": _frame_percentile(psv_speed, seconds, config),
        "bip_source": source,
        "sample_rate": (
            "native"
            if config.target_hz is None
            else f"{config.target_hz:g}Hz"
        ),
        "effective_sample_hz": config.sample_hz,
        **counts,
    }
    row["high_intensity_count_definition"] = "hsr_count_plus_sprint_count"
    row["hir_distance_m"] = row["high_intensity_distance_m"]
    row["hir_count"] = row["high_intensity_count"]

    scale = SECONDS_PER_HOUR / bip_seconds if bip_seconds > 0 else np.nan
    for name in VOLUME_COLUMNS:
        row[f"{name}_p60_bip"] = float(row[name]) * scale
    # TIP and OTIP volumes are each normalised by *their own* phase exposure, so
    # "high-intensity distance while our team has the ball" and "…while the
    # opponent has it" are directly comparable even though a player spends
    # different amounts of time in each phase. Dividing both by the same window
    # (or by ball-in-play) would smuggle possession share into the comparison.
    tip_scale = 1800.0 / tip_seconds if tip_seconds > 0 else np.nan
    for name in ("distance_tip_m", "high_intensity_distance_tip_m", "sprint_count_tip",
                 "high_intensity_count_tip"):
        row[f"{name}_p30_tip"] = float(row[name]) * tip_scale
    otip_scale = 1800.0 / otip_seconds if otip_seconds > 0 else np.nan
    for name in ("distance_otip_m", "high_intensity_distance_otip_m", "sprint_count_otip",
                 "high_intensity_count_otip"):
        row[f"{name}_p30_otip"] = float(row[name]) * otip_scale
    return row


def compute_player_match_metrics(
    tracking: pd.DataFrame,
    lineup: pd.DataFrame | None = None,
    config: PhysicalConfig | None = None,
    *,
    prepared: bool = False,
) -> pd.DataFrame:
    """One row of physical features per match and player.

    Args:
        tracking: Canonical contract tracking, or the output of
            :func:`physical.kinematics.add_kinematics` when ``prepared=True``.
        lineup: Optional per-match player metadata (contract §2).
        prepared: Skip canonicalisation and kinematics when the caller has
            already produced them, so an estimator sweep pays for them once.
    """

    config = config or PhysicalConfig()
    if prepared:
        frame = tracking
    else:
        frame = prepare_kinematics(tracking, lineup=lineup, config=config)
    required = {
        "match_id", "player_id", "period_id", "timestamp", "dt", "step_distance_m",
        "speed_mps", "acceleration_mps2", "valid_interval", "is_bip", "segment_id",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prepared tracking is missing: {', '.join(missing)}")

    source = bip_source(frame)
    rows = [
        _player_match_row(group, config, source)
        for _, group in frame.groupby(["match_id", "player_id"], sort=False, dropna=False)
    ]
    return pd.DataFrame(rows)


def effort_count_sensitivity(
    frame: pd.DataFrame,
    config: PhysicalConfig,
    *,
    threshold_kmh: float,
    dip_tolerances: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
) -> pd.DataFrame:
    """Effort counts under both duration rules and a sweep of dip tolerances.

    Operates on a prepared kinematics frame and returns one row per
    (duration rule, dip tolerance), so the counting convention can be reported
    as a measured sensitivity instead of an assumption.
    """

    from dataclasses import replace

    speed = frame["speed_mps"].to_numpy(dtype=float)
    segment = frame["segment_id"].to_numpy()
    bip = frame["is_bip"].to_numpy(dtype=bool)
    mask = np.isfinite(speed) & (speed > threshold_kmh * KMH_TO_MPS)
    rows = []
    for rule in ("samples", "span"):
        for tolerance in dip_tolerances:
            variant = replace(
                config, effort_duration_rule=rule, effort_dip_tolerance_seconds=tolerance
            )
            rows.append(
                {
                    "threshold_kmh": threshold_kmh,
                    "duration_rule": rule,
                    "dip_tolerance_s": tolerance,
                    "count": count_efforts(mask, segment, bip, variant),
                }
            )
    return pd.DataFrame(rows)


def compute_rate_sensitivity(
    tracking: pd.DataFrame,
    lineup: pd.DataFrame | None = None,
    config: PhysicalConfig | None = None,
    *,
    common_hz: float = 10.0,
) -> pd.DataFrame:
    """Run the common-rate and native-rate analyses side by side.

    Concatenates player-match tables computed on the common 10 Hz grid and at the
    feed's native rate, so a rate-driven difference in effort counts is visible
    rather than assumed away.
    """

    base = config or PhysicalConfig()
    return pd.concat(
        [
            compute_player_match_metrics(
                tracking, lineup=lineup, config=replace(base, target_hz=common_hz)
            ),
            compute_player_match_metrics(
                tracking, lineup=lineup, config=replace(base, target_hz=None)
            ),
        ],
        ignore_index=True,
    )
