"""Position to velocity and acceleration for canonical long tracking data.

Everything here is a pure function over the PHYSICAL_CONTRACT §1 ``tracking``
table.  The module owns three concerns and nothing else:

1. **Segmentation.** A *segment* is a maximal run of observations for one player
   inside one period whose consecutive gaps are at most
   ``config.max_gap_seconds``.  Nothing is derived, smoothed, interpolated or
   accumulated across a segment or period boundary.
2. **Velocity estimation.** Position is smoothed and *then* differentiated.
   Four estimators are provided so the choice can be evidenced rather than
   asserted: ``central`` (no smoothing), ``savgol``, ``butter`` and ``kalman``.
   The superseded ``trailing_mean`` smooths *speed* instead — it lags the signal
   by half its window and smears sprint onsets — and is kept only as the
   baseline in that comparison.
3. **Plausibility caps.** Samples implying a speed or acceleration beyond human
   limits are flagged invalid, never clipped, so the removed fraction stays
   measurable and reportable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

from .definitions import (
    DEAD_BALL_STATUSES,
    ESTIMATOR_BUTTER,
    ESTIMATOR_CENTRAL,
    ESTIMATOR_KALMAN,
    ESTIMATOR_SAVGOL,
    KMH_TO_MPS,
    LIVE_BALL_STATUSES,
    PhysicalConfig,
)

_PLAYER_ALIASES = ("player_id", "object_id")
_TIME_ALIASES = ("timestamp", "timestamp_seconds", "time_seconds")
_PERIOD_ALIASES = ("period_id", "period")
_BALL_STATE_ALIASES = ("ball_state", "ball_status")
_BALL_OWNER_ALIASES = ("ball_owning_team_id", "ball_poss_team_id")
_BALL_ALIASES = ("is_ball", "ball")
_START_ALIASES = (
    "appearance_start_seconds",
    "start_time",
    "start_time_seconds",
    "active_from",
    "appearance_start",
)
_END_ALIASES = (
    "appearance_end_seconds",
    "end_time",
    "end_time_seconds",
    "active_to",
    "appearance_end",
)
_PLAYING_TIME_MS_ALIASES = ("provider_playing_time_ms", "playing_time_ms")

KINEMATIC_COLUMNS = (
    "segment_id",
    "dt",
    "vx",
    "vy",
    "raw_speed_mps",
    "speed_mps",
    "acceleration_mps2",
    "step_distance_m",
    "raw_step_distance_m",
    "valid_interval",
    "capped_speed",
    "capped_acceleration",
    "is_bip",
)


def normalize_position_group(value: object) -> object:
    """Map provider position labels onto the published outfield groups."""

    if value is None or value is pd.NA:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    label = str(value).strip()
    key = (
        label.casefold()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
    )
    compact = "".join(key.split())
    # Bench entries carry no on-pitch role. SkillCorner Open Data labels unused
    # substitutes "SUB" in both `position` and `player_role.position_group`;
    # leaving that through would create a phantom seventh position group.
    if compact in {"sub", "substitute", "unused", "bench", "na", "none", "unknown"}:
        return pd.NA
    if compact in {"gk", "goalkeeper", "goalie", "keeper", "골키퍼"}:
        return "Goalkeeper"
    if compact in {
        "cb", "lcb", "rcb", "cd", "centraldefender", "centreback",
        "centerback", "defender", "df",
    }:
        return "Central Defender"
    if compact in {
        "lb", "rb", "lwb", "rwb", "wb", "fb", "fullback", "wingback",
    }:
        return "Full Back"
    if compact in {
        "dm", "cdm", "cm", "cam", "am", "mf", "midfield", "midfielder",
        "centralmidfield", "defensivemidfield", "attackingmidfield",
        "lcm", "rcm", "ldm", "rdm", "lam", "ram",
    }:
        return "Midfield"
    if compact in {
        "lw", "rw", "lm", "rm", "wm", "winger", "leftwing", "rightwing",
        # SkillCorner names this group "Wide Attacker" and the roles "LWF"/"RWF";
        # the published Week 5 group is "Winger". Without these the wingers fall
        # through to the raw label and drop out of every position-group ranking.
        "wideattacker", "wideforward", "lwf", "rwf", "wf",
        "lwm", "rwm", "widemidfield",
    }:
        return "Winger"
    if compact in {
        "st", "cf", "fw", "f", "ss", "forward", "striker", "centreforward",
        "centerforward",
    }:
        return "Forward"
    if "goal" in key and ("keep" in key or "keeper" in key):
        return "Goalkeeper"
    if "wing back" in key or "full back" in key:
        return "Full Back"
    if "wing" in key or "wide midfield" in key:
        return "Winger"
    if "midfield" in key:
        return "Midfield"
    if "forward" in key or "striker" in key:
        return "Forward"
    if "back" in key or "defender" in key:
        return "Central Defender"
    return label


def _first_present(columns: Iterable[str], choices: Iterable[str]) -> str | None:
    columns = set(columns)
    return next((name for name in choices if name in columns), None)


# --------------------------------------------------------------------------- #
# ball state
# --------------------------------------------------------------------------- #
def _status_to_bip(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").eq(1)
    normalized = values.astype("string").str.strip().str.lower()
    known_live = normalized.isin(LIVE_BALL_STATUSES)
    known_dead = normalized.isin(DEAD_BALL_STATUSES)
    # Unknown non-null states are not assumed live: false positives bias every
    # P60 denominator downward and are harder to diagnose than missing BIP.
    return known_live & ~known_dead


def derive_bip(frame: pd.DataFrame) -> pd.Series:
    """Ball-in-play mask, the P60 BIP denominator.

    BIP comes from the canonical frame-level ball state. The SkillCorner Open
    Data adapter defines that state as ``TIP + OTIP`` (an owning team is
    present), matching the public aggregate's
    ``minutes_full_tip + minutes_full_otip`` denominator. Generic adapters may
    still provide their own explicit ``alive`` state.
    """

    state_col = _first_present(frame.columns, _BALL_STATE_ALIASES)
    if state_col is not None:
        return _status_to_bip(frame[state_col])
    owner_col = _first_present(frame.columns, _BALL_OWNER_ALIASES)
    if owner_col is not None:
        # Possession-only feeds cannot see contested ball; BIP degrades to
        # TIP+OTIP and under-reports the denominator. Surfaced via bip_source.
        possession = frame[owner_col]
        if pd.api.types.is_numeric_dtype(possession):
            bip = possession.notna()
        else:
            norm = possession.astype("string").str.strip().str.lower()
            bip = possession.notna() & ~norm.isin({"", "0", "none", "nan", "<na>"})
        return bip.fillna(False).astype(bool)
    # Absence of both fields means BIP cannot be separated; treating observed
    # tracking as BIP is useful for synthetic/open data and is surfaced in QC.
    return pd.Series(True, index=frame.index, dtype=bool)


def bip_source(frame: pd.DataFrame) -> str:
    """Name the field ball-in-play was derived from, for audit."""

    if _first_present(frame.columns, _BALL_STATE_ALIASES) is not None:
        return "ball_state"
    if _first_present(frame.columns, _BALL_OWNER_ALIASES) is not None:
        return "ball_owning_team_id"
    return "assumed_all_observed"


# --------------------------------------------------------------------------- #
# canonicalisation
# --------------------------------------------------------------------------- #
def canonicalize(tracking: pd.DataFrame) -> pd.DataFrame:
    """Rename provider aliases onto contract names and sort deterministically.

    Emits both ``period_id`` (contract) and ``period`` (legacy) so consumers of
    either name keep working.
    """

    if tracking.empty:
        raise ValueError("tracking must contain at least one observation")

    player_col = _first_present(tracking.columns, _PLAYER_ALIASES)
    time_col = _first_present(tracking.columns, _TIME_ALIASES)
    period_col = _first_present(tracking.columns, _PERIOD_ALIASES)
    missing = [
        label
        for label, value in (
            ("player_id/object_id", player_col),
            ("timestamp seconds", time_col),
            ("x", "x" if "x" in tracking else None),
            ("y", "y" if "y" in tracking else None),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"tracking is missing required columns: {', '.join(missing)}")

    out = tracking.copy()
    rename = {}
    if player_col != "player_id":
        rename[player_col] = "player_id"
    if time_col != "timestamp":
        rename[time_col] = "timestamp"
    if "match_id" not in out and "game_id" in out:
        rename["game_id"] = "match_id"
    out = out.rename(columns=rename)

    if period_col is None:
        out["period_id"] = np.int8(1)
    elif period_col != "period_id":
        out["period_id"] = out[period_col]
    if "match_id" not in out:
        out["match_id"] = "match"
    if "frame_id" not in out:
        out["frame_id"] = np.arange(len(out), dtype=np.int64)

    ball_col = _first_present(out.columns, _BALL_ALIASES)
    if ball_col is not None:
        values = out[ball_col]
        if values.dtype == bool:
            out = out.loc[~values].copy()
        else:
            normalized = values.astype("string").str.lower()
            out = out.loc[~normalized.isin({"1", "true", "ball"})].copy()
    out = out.loc[
        ~out["player_id"].astype("string").str.strip().str.lower().eq("ball")
    ].copy()

    for col in ("timestamp", "x", "y"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(
        subset=["match_id", "period_id", "player_id", "timestamp", "x", "y"]
    )
    if out.empty:
        raise ValueError("tracking contains no valid player observations")

    out = out.sort_values(
        ["match_id", "player_id", "period_id", "timestamp", "frame_id"], kind="stable"
    )
    # Exact duplicate player timestamps cannot define a unique velocity.  Keep
    # the last observation so provider correction rows win deterministically.
    out = out.drop_duplicates(
        subset=["match_id", "player_id", "period_id", "timestamp"], keep="last"
    )
    out["period"] = out["period_id"]
    return out.reset_index(drop=True)


def attach_lineup(
    tracking: pd.DataFrame, lineup: pd.DataFrame | None
) -> pd.DataFrame:
    """Attach stable player metadata and enforce supplied appearance intervals.

    Start/end values are interpreted in the same seconds timebase as
    ``tracking.timestamp``.  Provider adapters are responsible for converting
    period-relative lineup clocks before calling this function.
    """

    out = tracking
    if lineup is None or lineup.empty:
        return out
    lineup = lineup.copy()
    player_col = _first_present(lineup.columns, _PLAYER_ALIASES)
    if player_col is None:
        raise ValueError("lineup requires player_id or object_id")
    if player_col != "player_id":
        lineup = lineup.rename(columns={player_col: "player_id"})

    keys = ["player_id"]
    if "match_id" in lineup and "match_id" in out:
        keys.insert(0, "match_id")
    if lineup.duplicated(keys).any():
        raise ValueError("lineup must contain at most one row per match/player")

    start_col = _first_present(lineup.columns, _START_ALIASES)
    end_col = _first_present(lineup.columns, _END_ALIASES)
    playing_time_col = _first_present(lineup.columns, _PLAYING_TIME_MS_ALIASES)
    canonical_rename = {}
    if start_col is not None and start_col != "appearance_start_seconds":
        canonical_rename[start_col] = "appearance_start_seconds"
        start_col = "appearance_start_seconds"
    if end_col is not None and end_col != "appearance_end_seconds":
        canonical_rename[end_col] = "appearance_end_seconds"
        end_col = "appearance_end_seconds"
    if playing_time_col is not None and playing_time_col != "provider_playing_time_ms":
        canonical_rename[playing_time_col] = "provider_playing_time_ms"
        playing_time_col = "provider_playing_time_ms"
    lineup = lineup.rename(columns=canonical_rename)
    if "position_group" not in lineup:
        lineup["position_group"] = pd.NA
    position_source = (
        lineup["position"]
        if "position" in lineup
        else pd.Series(pd.NA, index=lineup.index)
    )
    lineup["position_group"] = lineup["position_group"].where(
        lineup["position_group"].notna(),
        position_source.map(normalize_position_group),
    )

    wanted = keys + [
        col
        for col in (
            "team_id",
            "team",
            "position",
            "position_group",
            "player_name",
            "minutes_played",
            "minutes_tip",
            "minutes_otip",
            start_col,
            end_col,
            playing_time_col,
        )
        if col is not None
        and col in lineup.columns
        and col not in keys
        and col not in out.columns
    ]
    wanted = list(dict.fromkeys(wanted))
    out = out.merge(lineup[wanted], on=keys, how="left", validate="many_to_one")

    clock = (
        pd.to_numeric(out["match_seconds"], errors="coerce")
        if "match_seconds" in out
        else out["timestamp"]
    )
    if start_col is not None and start_col in out:
        starts = pd.to_numeric(out[start_col], errors="coerce")
        out = out.loc[starts.isna() | clock.ge(starts)].copy()
        clock = clock.loc[out.index]
    if end_col is not None and end_col in out:
        ends = pd.to_numeric(out[end_col], errors="coerce")
        out = out.loc[ends.isna() | clock.le(ends)].copy()
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# segmentation and resampling
# --------------------------------------------------------------------------- #
def segment_bounds(
    frame: pd.DataFrame, *, max_gap_seconds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(segment_id, starts, stops)`` for a canonicalised frame.

    ``frame`` must already be sorted by match, player, period, timestamp.
    ``starts``/``stops`` are half-open row offsets, so segment ``k`` occupies
    ``frame.iloc[starts[k]:stops[k]]``.
    """

    n = len(frame)
    if n == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return np.empty(0, dtype=np.int32), empty_i, empty_i

    key = np.zeros(n, dtype=bool)
    for col in ("match_id", "player_id", "period_id"):
        values = frame[col].to_numpy()
        key[1:] |= values[1:] != values[:-1]
    timestamp = frame["timestamp"].to_numpy(dtype=float)
    dt = np.diff(timestamp, prepend=np.nan)
    key |= ~np.isfinite(dt) | (dt <= 0) | (dt > max_gap_seconds + 1e-9)
    key[0] = True

    segment_id = (np.cumsum(key) - 1).astype(np.int32)
    starts = np.flatnonzero(key).astype(np.int64)
    stops = np.append(starts[1:], n).astype(np.int64)
    return segment_id, starts, stops


def resample_to_grid(
    tracking: pd.DataFrame,
    target_hz: float,
    *,
    max_gap_seconds: float | None = None,
) -> pd.DataFrame:
    """Linearly interpolate each contiguous segment onto a uniform grid.

    The contract requires the loader to deliver a 10 Hz grid, so this is only a
    fallback for provider frames that arrive at their native rate.  State and
    categorical columns are held from the latest observed sample, so no value
    is ever taken from the future.
    """

    _, starts, stops = segment_bounds(
        tracking,
        max_gap_seconds=(
            1.0 / target_hz * 5
            if max_gap_seconds is None
            else max_gap_seconds
        ),
    )
    step = 1.0 / target_hz
    pieces = []
    for lo, hi in zip(starts, stops):
        segment = tracking.iloc[lo:hi]
        if len(segment) < 2:
            pieces.append(segment.copy())
            continue
        times = segment["timestamp"].to_numpy(dtype=float)
        grid = times[0] + np.arange(int(np.floor((times[-1] - times[0]) / step)) + 1) * step
        grid = np.unique(np.round(grid, 9))
        block = pd.DataFrame({"timestamp": grid})
        block["x"] = np.interp(grid, times, segment["x"].to_numpy(dtype=float))
        block["y"] = np.interp(grid, times, segment["y"].to_numpy(dtype=float))
        held = np.clip(
            np.searchsorted(times, grid + step * 1e-7, side="right") - 1,
            0,
            len(segment) - 1,
        )
        for col in segment.columns:
            if col in {"timestamp", "x", "y"}:
                continue
            block[col] = segment.iloc[held][col].to_numpy()
        pieces.append(block)
    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(
        ["match_id", "player_id", "period_id", "timestamp"], kind="stable"
    ).reset_index(drop=True)


def median_sample_seconds(frame: pd.DataFrame) -> float:
    """Median positive within-player sample interval, for grid validation."""

    _, starts, _ = segment_bounds(frame, max_gap_seconds=np.inf)
    dt = np.diff(frame["timestamp"].to_numpy(dtype=float), prepend=np.nan)
    dt[starts] = np.nan
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if dt.size else float("nan")


# --------------------------------------------------------------------------- #
# derivative helpers
# --------------------------------------------------------------------------- #
def central_gradient(
    values: np.ndarray, starts: np.ndarray, stops: np.ndarray, dt: float
) -> np.ndarray:
    """Second-order central difference inside each segment, one-sided at edges."""

    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n >= 3:
        out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    lengths = stops - starts
    multi = lengths > 1
    a, b = starts[multi], stops[multi]
    out[a] = (values[a + 1] - values[a]) / dt
    out[b - 1] = (values[b - 1] - values[b - 2]) / dt
    out[starts[lengths == 1]] = np.nan
    return out


def centred_moving_average(
    values: np.ndarray, starts: np.ndarray, stops: np.ndarray, window: int
) -> np.ndarray:
    """Centred moving average within each segment, shrinking at the edges."""

    if window <= 1:
        return values
    half = window // 2
    filled = np.nan_to_num(values, nan=0.0)
    ok = np.isfinite(values).astype(float)
    cs_v = np.concatenate(([0.0], np.cumsum(filled)))
    cs_n = np.concatenate(([0.0], np.cumsum(ok)))
    idx = np.arange(len(values))
    lengths = stops - starts
    seg_start = np.repeat(starts, lengths)
    seg_stop = np.repeat(stops, lengths)
    lo = np.maximum(idx - half, seg_start)
    hi = np.minimum(idx + half + 1, seg_stop)
    total = cs_v[hi] - cs_v[lo]
    count = cs_n[hi] - cs_n[lo]
    return np.divide(
        total, count, out=np.full(len(values), np.nan, dtype=float), where=count > 0
    )


def _savgol_velocity(
    x: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    dt: float,
    window: int,
    polyorder: int,
) -> tuple[np.ndarray, np.ndarray]:
    vx = np.full(len(x), np.nan, dtype=float)
    vy = np.full(len(y), np.nan, dtype=float)
    for lo, hi in zip(starts, stops):
        length = int(hi - lo)
        win = min(window, length if length % 2 else length - 1)
        if win <= polyorder:
            continue
        block = np.vstack((x[lo:hi], y[lo:hi]))
        deriv = savgol_filter(
            block,
            window_length=win,
            polyorder=polyorder,
            deriv=1,
            delta=dt,
            axis=-1,
            mode="interp",
        )
        vx[lo:hi] = deriv[0]
        vy[lo:hi] = deriv[1]
    # Segments too short for any polynomial fit fall back to central differences.
    fallback = ~np.isfinite(vx)
    if fallback.any():
        vx[fallback] = central_gradient(x, starts, stops, dt)[fallback]
        vy[fallback] = central_gradient(y, starts, stops, dt)[fallback]
    return vx, vy


def _butter_positions(
    x: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    sample_hz: float,
    cutoff_hz: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    b, a = butter(order, cutoff_hz / (0.5 * sample_hz), btype="low")
    padlen = 3 * max(len(a), len(b))
    xs, ys = x.astype(float).copy(), y.astype(float).copy()
    for lo, hi in zip(starts, stops):
        if hi - lo <= padlen:
            continue
        block = np.vstack((x[lo:hi], y[lo:hi]))
        smoothed = filtfilt(b, a, block, axis=-1, padlen=padlen)
        xs[lo:hi] = smoothed[0]
        ys[lo:hi] = smoothed[1]
    return xs, ys


def _kalman_gains(
    dt: float, position_sigma: float, accel_sigma: float, iterations: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Steady-state constant-velocity Kalman gain and RTS smoother gain.

    The covariance recursion of a constant-velocity model with fixed noise is
    data independent and converges within a second of samples, so the filter is
    run to steady state once and the limiting gains are reused.  The only
    approximation is in the first and last handful of samples of a segment.
    """

    transition = np.array([[1.0, dt], [0.0, 1.0]])
    process = accel_sigma**2 * np.array(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]
    )
    measurement = position_sigma**2
    covariance = np.eye(2) * max(position_sigma**2, 1.0)
    gain = np.zeros((2, 1))
    for _ in range(iterations):
        predicted = transition @ covariance @ transition.T + process
        gain = predicted[:, :1] / (predicted[0, 0] + measurement)
        covariance = predicted - gain @ predicted[:1, :]
    predicted = transition @ covariance @ transition.T + process
    rts = covariance @ transition.T @ np.linalg.inv(predicted)
    return gain.ravel(), rts


def _kalman_velocity(
    x: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    dt: float,
    position_sigma: float,
    accel_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity Kalman filter with RTS smoothing, vectorised.

    All segments are laid out in one padded ``(n_segments, max_length)`` index
    matrix, so both recursions become loops over *time within segment* with
    vector operations across segments and axes.
    """

    gain, rts = _kalman_gains(dt, position_sigma, accel_sigma)
    lengths = (stops - starts).astype(np.int64)
    vx = np.full(len(x), np.nan, dtype=float)
    vy = np.full(len(y), np.nan, dtype=float)
    if lengths.size == 0 or int(lengths.max()) < 2:
        return vx, vy

    max_len = int(lengths.max())
    grid = np.arange(max_len)
    active = grid[None, :] < lengths[:, None]
    rows = np.where(active, starts[:, None] + grid[None, :], 0)
    obs = np.stack((x[rows], y[rows]), axis=-1)

    pos = np.empty((len(lengths), max_len, 2))
    vel = np.empty_like(pos)
    p = obs[:, 0, :].copy()
    v = np.zeros((len(lengths), 2))
    pos[:, 0, :], vel[:, 0, :] = p, v
    for t in range(1, max_len):
        predicted = p + v * dt
        residual = obs[:, t, :] - predicted
        step = active[:, t][:, None]
        p = np.where(step, predicted + gain[0] * residual, p)
        v = np.where(step, v + gain[1] * residual, v)
        pos[:, t, :], vel[:, t, :] = p, v

    # RTS backward pass.  At each segment's own last sample the smoother equals
    # the filter, which is exactly what skipping the update there gives.
    smooth_p, smooth_v = pos.copy(), vel.copy()
    for t in range(max_len - 2, -1, -1):
        step = (t < lengths - 1)[:, None]
        dp = smooth_p[:, t + 1, :] - (pos[:, t, :] + vel[:, t, :] * dt)
        dv = smooth_v[:, t + 1, :] - vel[:, t, :]
        smooth_p[:, t, :] = np.where(
            step, pos[:, t, :] + rts[0, 0] * dp + rts[0, 1] * dv, smooth_p[:, t, :]
        )
        smooth_v[:, t, :] = np.where(
            step, vel[:, t, :] + rts[1, 0] * dp + rts[1, 1] * dv, smooth_v[:, t, :]
        )

    flat = rows[active]
    vx[flat] = smooth_v[:, :, 0][active]
    vy[flat] = smooth_v[:, :, 1][active]
    return vx, vy


def _trailing_time_weighted_speed(
    timestamp: np.ndarray,
    raw_speed: np.ndarray,
    dt: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    window: float,
) -> np.ndarray:
    """Superseded baseline: mean interval speed over ``(t - window, t]``."""

    out = np.full(len(timestamp), np.nan, dtype=float)
    for lo, hi in zip(starts, stops):
        t = timestamp[lo:hi]
        speed = raw_speed[lo:hi]
        step = dt[lo:hi]
        valid = np.isfinite(speed) & np.isfinite(step) & (step > 0)
        cumulative_distance = np.cumsum(np.where(valid, speed * step, 0.0))
        cumulative_time = np.cumsum(np.where(valid, step, 0.0))
        left = np.maximum(t[0], t - window)
        window_distance = cumulative_distance - np.interp(left, t, cumulative_distance)
        window_time = cumulative_time - np.interp(left, t, cumulative_time)
        block = np.divide(
            window_distance,
            window_time,
            out=np.full(len(t), np.nan, dtype=float),
            where=window_time > 0,
        )
        block[~valid] = np.nan
        out[lo:hi] = block
    return out


# --------------------------------------------------------------------------- #
# main entry points
# --------------------------------------------------------------------------- #
def add_kinematics(
    tracking: pd.DataFrame, config: PhysicalConfig | None = None
) -> pd.DataFrame:
    """Add segment ids, velocity, speed, acceleration and validity flags.

    ``tracking`` must be a canonicalised contract table on a uniform grid.
    Returns a copy with :data:`KINEMATIC_COLUMNS` appended.  Nothing is derived
    across a period boundary or a gap longer than ``config.max_gap_seconds``.

    The differentiation step is taken from the data, not from
    ``config.sample_hz``, and a segment whose internal sample spacing is not
    uniform is rejected.  Otherwise a permissive ``max_gap_seconds`` would keep
    missing frames inside a segment and every derivative across such a hole
    would be scaled by the ratio of the real gap to the nominal step — a silent
    multiple-fold speed overestimate exactly where tracking is worst.
    """

    config = config or PhysicalConfig()
    out = tracking.reset_index(drop=True).copy()
    segment_id, starts, stops = segment_bounds(
        out, max_gap_seconds=config.max_gap_seconds
    )
    out["segment_id"] = segment_id

    timestamp = out["timestamp"].to_numpy(dtype=float)
    x = out["x"].to_numpy(dtype=float)
    y = out["y"].to_numpy(dtype=float)
    step_seconds = np.diff(timestamp, prepend=np.nan)
    step_seconds[starts] = np.nan

    within = step_seconds[np.isfinite(step_seconds)]
    dt = float(np.median(within)) if within.size else config.dt
    if within.size and np.abs(within - dt).max() > 0.25 * dt:
        raise ValueError(
            f"tracking is not on a uniform grid: within-segment steps span "
            f"[{within.min():.4f}, {within.max():.4f}]s around a median of {dt:.4f}s. "
            f"Resample upstream (PHYSICAL_CONTRACT §6) or lower "
            f"config.max_gap_seconds below {within.max():.4f}."
        )

    raw_step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
    raw_step[starts] = np.nan

    velocity: tuple[np.ndarray, np.ndarray] | None = None
    if config.estimator == ESTIMATOR_CENTRAL:
        velocity = (
            central_gradient(x, starts, stops, dt),
            central_gradient(y, starts, stops, dt),
        )
    elif config.estimator == ESTIMATOR_SAVGOL:
        velocity = _savgol_velocity(
            x,
            y,
            starts,
            stops,
            dt,
            config.window_samples(config.savgol_window_seconds),
            config.savgol_polyorder,
        )
    elif config.estimator == ESTIMATOR_BUTTER:
        xs, ys = _butter_positions(
            x,
            y,
            starts,
            stops,
            config.sample_hz,
            config.butter_cutoff_hz,
            config.butter_order,
        )
        velocity = (
            central_gradient(xs, starts, stops, dt),
            central_gradient(ys, starts, stops, dt),
        )
    elif config.estimator == ESTIMATOR_KALMAN:
        velocity = _kalman_velocity(
            x,
            y,
            starts,
            stops,
            dt,
            config.kalman_position_sigma_m,
            config.kalman_accel_sigma_mps2,
        )

    if velocity is None:
        speed = _trailing_time_weighted_speed(
            timestamp, raw_step / step_seconds, step_seconds, starts, stops,
            config.speed_window_seconds,
        )
        out["vx"] = np.nan
        out["vy"] = np.nan
    else:
        speed = np.hypot(velocity[0], velocity[1])
        out["vx"] = velocity[0]
        out["vy"] = velocity[1]

    raw_speed = speed.copy()
    capped_speed = np.isfinite(raw_speed) & (
        raw_speed > config.max_speed_kmh * KMH_TO_MPS
    )
    speed = np.where(capped_speed, np.nan, raw_speed)

    acceleration = central_gradient(
        centred_moving_average(
            speed,
            starts,
            stops,
            config.window_samples(config.accel_smooth_window_seconds),
        ),
        starts,
        stops,
        dt,
    )
    capped_acceleration = np.isfinite(acceleration) & (
        np.abs(acceleration) > config.max_abs_acceleration_mps2
    )
    acceleration = np.where(capped_acceleration, np.nan, acceleration)

    interval_seconds = np.where(
        np.isfinite(step_seconds) & (step_seconds > 0), step_seconds, 0.0
    )
    out["dt"] = interval_seconds
    out["raw_speed_mps"] = raw_speed
    out["speed_mps"] = speed
    out["acceleration_mps2"] = acceleration
    # Band distances use the estimated speed, so they sum to the total distance
    # rather than to the noisier raw path length.
    out["step_distance_m"] = speed * interval_seconds
    out["raw_step_distance_m"] = raw_step
    out["valid_interval"] = np.isfinite(speed)
    out["capped_speed"] = capped_speed
    out["capped_acceleration"] = capped_acceleration
    out["is_bip"] = derive_bip(out).to_numpy(dtype=bool)
    return out


def prepare_kinematics(
    tracking: pd.DataFrame,
    lineup: pd.DataFrame | None = None,
    config: PhysicalConfig | None = None,
) -> pd.DataFrame:
    """Canonicalise, attach the lineup, and derive per-sample kinematics.

    ``distance_m`` is kept as an alias of ``step_distance_m`` for existing
    consumers.  When ``config.target_hz`` is set and the input is not already on
    that grid, the input is resampled first; the contract expects the loader to
    have done this, so it is a guard rather than the normal path.
    """

    config = config or PhysicalConfig()
    frame = attach_lineup(canonicalize(tracking), lineup)
    if config.target_hz is not None:
        _, starts, _ = segment_bounds(
            frame, max_gap_seconds=config.max_gap_seconds
        )
        steps = np.diff(frame["timestamp"].to_numpy(dtype=float), prepend=np.nan)
        steps[starts] = np.nan
        within = steps[np.isfinite(steps)]
        target_step = 1.0 / config.target_hz
        if within.size and np.abs(within - target_step).max() > 1e-6:
            frame = resample_to_grid(
                frame,
                config.target_hz,
                max_gap_seconds=config.max_gap_seconds,
            )
        effective_hz = config.target_hz
    else:
        step = median_sample_seconds(frame)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("native-rate kinematics requires at least two timestamps")
        effective_hz = 1.0 / step
    effective_config = replace(config, sample_hz=float(effective_hz))
    result = add_kinematics(frame, effective_config)
    result["effective_sample_hz"] = float(effective_hz)
    result["distance_m"] = result["step_distance_m"]
    return result


def cap_report(kinematics: pd.DataFrame) -> dict[str, float]:
    """Fraction of samples removed by each plausibility cap.

    Denominators are the samples that had a finite estimate before the cap, so
    the numbers describe the caps themselves rather than segment edges.
    """

    n = len(kinematics)
    speed_capped = float(kinematics["capped_speed"].sum())
    accel_capped = float(kinematics["capped_acceleration"].sum())
    speed_finite = float(kinematics["valid_interval"].sum()) + speed_capped
    accel_finite = float(kinematics["acceleration_mps2"].notna().sum()) + accel_capped
    return {
        "n_samples": float(n),
        "n_segments": float(kinematics["segment_id"].nunique()),
        "speed_cap_removed": speed_capped,
        "speed_cap_removed_frac": speed_capped / max(speed_finite, 1.0),
        "acceleration_cap_removed": accel_capped,
        "acceleration_cap_removed_frac": accel_capped / max(accel_finite, 1.0),
        "segment_edge_invalid_frac": float(
            ((~kinematics["valid_interval"]) & (~kinematics["capped_speed"])).sum()
        )
        / max(n, 1.0),
    }
