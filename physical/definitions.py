"""Definitions shared by the physical-metric pipeline.

Speed bands (SkillCorner published glossary, male outfield)
----------------------------------------------------------
* Running: ``15 <= v < 20`` km/h
* High-speed running (HSR): ``20 <= v <= 25`` km/h
* Sprinting: ``v > 25`` km/h
* High-intensity distance: the union of HSR and Sprint distance

Legacy Physical v2 estimates speed with the trailing one-second moving average.
The position-smoothing estimators remain available as method sensitivities.

Discrete activities ("efforts")
-------------------------------
Running, HSR, and Sprint counts are discrete band activities sustained for at
least 0.7 s. The official legacy Physical v2 ``Count HI`` is
``Count HSR + Count Sprint`` rather than a contiguous ``speed >= 20`` union.

Effort duration is the time spent above the threshold, ``n_samples * dt``
(:data:`EFFORT_DURATION_SAMPLES`) — the time integral of the above-threshold
indicator.  The stricter alternative measures the span from the first to the
last above-threshold sample, ``(n_samples - 1) * dt``
(:data:`EFFORT_DURATION_SPAN`).  At 10 Hz the two differ by exactly one sample,
i.e. by a seventh of the 0.7 s requirement, so both are selectable and the
difference is reported as a sensitivity rather than assumed away.

The legacy Physical ``Count High Acceleration`` used by the 2023 benchmark is a
contiguous effort *exceeding* the threshold with a *minimum* duration: strict
``acceleration > threshold`` and inclusive ``duration >= min_seconds``.  This
must not be conflated with SkillCorner's separate Physical V3 ``Explosive
Acceleration`` introduced in 2025, which additionally requires starting below
9 km/h and reaching HSR or Sprint.  The acceleration threshold is **not
published**, so it is reverse-calibrated against the article's K League 1 value
and must always be reported together with that calibration.

PSV-99 is the 99th percentile of the **peak velocities of the activities above
15 km/h and at or below 40 km/h** — one observation per activity, not one per
frame.  The frame-level percentile is a different estimand and is retained only
as ``psv99_frame_kmh`` for sensitivity.  PSV-99 is a peak output, never a
P60 BIP-normalized volume.  At season level, the public SkillCorner Open Data
workflow uses the strongest recorded player value rather than averaging peak
values across performances.

P60 BIP normalizes by **ball-in-play** time, identified in canonical data by
``ball_state == "alive"``. Conceptually BIP includes contested and loose play
without a team owner.

* For the public SkillCorner Open Data physical workflow, BIP is operationally
  ``TIP + OTIP``: a frame is live when the possession label identifies either
  team. This matches the published aggregate fields
  ``minutes_full_tip + minutes_full_otip``. Frames with no owning team are kept
  for tracking-quality diagnostics but are not part of the canonical P60 BIP
  denominator.
* Bepro ``alive`` = ``ball_state`` in ``{"home", "away", "neutral"}``;
  ``"ballout"`` and null are dead.

``ball_owning_team_id`` is a **separate** column carrying the provider's team
possession label. TIP and OTIP derive from it and are reported as
``tip_seconds`` / ``otip_seconds``; they must never be substituted for the BIP
denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

KMH_TO_MPS = 1.0 / 3.6
MPS_TO_KMH = 3.6
SECONDS_PER_HOUR = 3600.0
PHYSICAL_DEFINITION_VERSION = "skillcorner-public-physical-v3"
PLAYED_TIME_SOURCE_PROVIDER = "provider_playing_time_ms"
PLAYED_TIME_SOURCE_APPEARANCE = "appearance_interval"
PLAYED_TIME_SOURCE_OBSERVED = "observed_tracking"
QC_ALLOWED_PLAYED_TIME_SOURCES = (
    PLAYED_TIME_SOURCE_PROVIDER,
    PLAYED_TIME_SOURCE_APPEARANCE,
)

ESTIMATOR_CENTRAL = "central"
ESTIMATOR_SAVGOL = "savgol"
ESTIMATOR_BUTTER = "butter"
ESTIMATOR_KALMAN = "kalman"
ESTIMATOR_TRAILING_MEAN = "trailing_mean"
ESTIMATORS = (
    ESTIMATOR_CENTRAL,
    ESTIMATOR_SAVGOL,
    ESTIMATOR_BUTTER,
    ESTIMATOR_KALMAN,
    ESTIMATOR_TRAILING_MEAN,
)

EFFORT_DURATION_SAMPLES = "samples"
EFFORT_DURATION_SPAN = "span"
EFFORT_DURATION_RULES = (EFFORT_DURATION_SAMPLES, EFFORT_DURATION_SPAN)

#: Position groups of the article's breakdown.  Goalkeeper is ours; the article
#: reports outfield groups only.
POSITION_GROUPS = (
    "Goalkeeper",
    "Central Defender",
    "Full Back",
    "Midfield",
    "Winger",
    "Forward",
)


@dataclass(frozen=True)
class SpeedThresholds:
    """Male outfield speed-band thresholds in km/h."""

    running_min_kmh: float = 15.0
    hsr_min_kmh: float = 20.0
    sprint_min_kmh: float = 25.0

    @property
    def running_min_mps(self) -> float:
        return self.running_min_kmh * KMH_TO_MPS

    @property
    def hsr_min_mps(self) -> float:
        return self.hsr_min_kmh * KMH_TO_MPS

    @property
    def sprint_min_mps(self) -> float:
        return self.sprint_min_kmh * KMH_TO_MPS


@dataclass(frozen=True)
class PhysicalConfig:
    """Configuration for provider-neutral kinematics and metrics.

    Parameters
    ----------
    estimator:
        Velocity estimator, one of :data:`ESTIMATORS`.  Every estimator except
        ``"trailing_mean"`` smooths position and then differentiates.
    sample_hz:
        Rate of the canonical tracking grid.  Both vendors arrive resampled to
        10 Hz, so a smoothing window in seconds is identical across vendors.
    target_hz:
        ``10.0`` is the primary common-rate analysis.  Set to ``None`` for the
        native timestamp-rate sensitivity analysis.
    max_gap_seconds:
        Adjacent observations farther apart than this form separate segments.
        No velocity, interpolation, smoothing, or bout is allowed across them.
    savgol_window_seconds, savgol_polyorder:
        Savitzky-Golay position window and polynomial order.  Velocity is the
        analytic first derivative of the fitted polynomial.
    butter_cutoff_hz, butter_order:
        Zero-phase (``filtfilt``) Butterworth low-pass applied to position.
    kalman_position_sigma_m, kalman_accel_sigma_mps2:
        Constant-velocity Kalman/RTS measurement and process noise.
    speed_window_seconds:
        Width of the legacy v2 trailing time-weighted mean of speed.
    accel_smooth_window_seconds:
        Optional extra centred moving average applied to speed before
        differentiating for acceleration.  ``0.0`` disables it.
    max_speed_kmh, max_abs_acceleration_mps2:
        Physiological plausibility caps.  Offending samples are marked invalid,
        never clipped, so the removed fraction stays measurable.
    effort_dip_tolerance_seconds:
        Above-threshold runs separated by a shorter sub-threshold dip merge
        into one activity.  ``0.0`` requires a strictly continuous run.
    effort_bip_min_share:
        An activity is attributed to ball-in-play when at least this share of
        its duration is ball-alive.  Activities are detected on the continuous
        speed signal first, because masking frames before detection fragments
        physiologically single efforts at possession changes.
    """

    estimator: str = ESTIMATOR_TRAILING_MEAN
    sample_hz: float = 10.0
    target_hz: float | None = 10.0
    max_gap_seconds: float = 0.5

    savgol_window_seconds: float = 0.7
    savgol_polyorder: int = 2
    butter_cutoff_hz: float = 1.0
    butter_order: int = 2
    kalman_position_sigma_m: float = 0.15
    kalman_accel_sigma_mps2: float = 3.0
    speed_window_seconds: float = 1.0

    accel_smooth_window_seconds: float = 0.0
    max_speed_kmh: float = 40.0
    max_abs_acceleration_mps2: float = 8.0

    high_acceleration_mps2: float = 3.0
    high_acceleration_min_seconds: float = 0.7
    high_deceleration_mps2: float = 3.0
    speed_activity_min_seconds: float = 0.7
    effort_duration_rule: str = EFFORT_DURATION_SAMPLES
    effort_dip_tolerance_seconds: float = 0.0
    effort_bip_min_share: float = 0.5

    psv_percentile: float = 99.0
    psv_min_kmh: float = 15.0
    psv_max_kmh: float = 40.0

    thresholds: SpeedThresholds = field(default_factory=SpeedThresholds)

    def __post_init__(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}")
        if self.effort_duration_rule not in EFFORT_DURATION_RULES:
            raise ValueError(
                f"effort_duration_rule must be one of {EFFORT_DURATION_RULES}"
            )
        if self.sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        if self.target_hz is not None and self.target_hz <= 0:
            raise ValueError("target_hz must be positive or None")
        if self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        if self.speed_window_seconds <= 0:
            raise ValueError("speed_window_seconds must be positive")
        if self.savgol_window_seconds <= 0:
            raise ValueError("savgol_window_seconds must be positive")
        if self.savgol_polyorder < 1:
            raise ValueError("savgol_polyorder must be at least 1")
        if not 0.0 < self.butter_cutoff_hz < self.sample_hz / 2.0:
            raise ValueError("butter_cutoff_hz must lie in (0, sample_hz / 2)")
        if self.butter_order < 1:
            raise ValueError("butter_order must be at least 1")
        if self.kalman_position_sigma_m <= 0 or self.kalman_accel_sigma_mps2 <= 0:
            raise ValueError("Kalman noise scales must be positive")
        if self.accel_smooth_window_seconds < 0:
            raise ValueError("accel_smooth_window_seconds must be non-negative")
        if self.max_speed_kmh <= self.thresholds.sprint_min_kmh:
            raise ValueError("max_speed_kmh must exceed the sprint threshold")
        if self.max_abs_acceleration_mps2 <= 0:
            raise ValueError("max_abs_acceleration_mps2 must be positive")
        if self.high_acceleration_mps2 <= 0 or self.high_deceleration_mps2 <= 0:
            raise ValueError("acceleration thresholds are positive magnitudes")
        if self.high_acceleration_min_seconds <= 0:
            raise ValueError("high_acceleration_min_seconds must be positive")
        if self.speed_activity_min_seconds < 0:
            raise ValueError("speed_activity_min_seconds must be non-negative")
        if self.effort_dip_tolerance_seconds < 0:
            raise ValueError("effort_dip_tolerance_seconds must be non-negative")
        if not 0.0 <= self.effort_bip_min_share <= 1.0:
            raise ValueError("effort_bip_min_share must lie in [0, 1]")
        if not 0.0 < self.psv_percentile <= 100.0:
            raise ValueError("psv_percentile must lie in (0, 100]")
        if self.psv_min_kmh < 0 or self.psv_max_kmh <= self.psv_min_kmh:
            raise ValueError("PSV bounds must satisfy 0 <= min < max")

    @property
    def dt(self) -> float:
        """Nominal sample interval of the canonical grid, in seconds."""
        return 1.0 / self.sample_hz

    def window_samples(self, seconds: float, *, odd: bool = True) -> int:
        """Convert a window in seconds to a sample count on the canonical grid."""
        n = int(round(seconds * self.sample_hz))
        if odd and n % 2 == 0:
            n += 1
        return max(n, 1)


LIVE_BALL_STATUSES = frozenset(
    {
        "1",
        "alive",
        "in",
        "in_play",
        "inplay",
        "live",
        "playing",
        "true",
    }
)
DEAD_BALL_STATUSES = frozenset(
    {"0", "dead", "dead_ball", "false", "out", "out_of_play", "stopped"}
)
