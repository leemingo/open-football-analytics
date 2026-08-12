"""Effort-counting and speed-band tests on trajectories with known answers.

The effort tests drive :func:`physical.physical_features.find_efforts` from a
synthetic *speed* array so the expected count follows from the construction
rather than from an estimator's behaviour.  The end-to-end tests then build
synthetic *positions* and check that the same answers survive the full
canonicalise -> kinematics -> features path.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from physical.definitions import (
    EFFORT_DURATION_SAMPLES,
    EFFORT_DURATION_SPAN,
    PhysicalConfig,
)
from physical.kinematics import add_kinematics, canonicalize
from physical.physical_features import (
    compute_player_match_metrics,
    count_efforts,
    find_efforts,
    psv,
)

DT = 0.1
KMH = 1.0 / 3.6
SPRINT = 25.0
HIGH_INTENSITY = 20.0


def speed_profile(*blocks: tuple[float, float]) -> np.ndarray:
    """Concatenate ``(speed_kmh, seconds)`` blocks onto the 10 Hz grid."""

    return np.concatenate(
        [np.full(int(round(seconds / DT)), kmh * KMH) for kmh, seconds in blocks]
    )


def straight_line_tracking(
    speed_kmh_profile: np.ndarray,
    *,
    player_id: int = 1,
    period_id: int = 1,
    frame_offset: int = 0,
    alive: bool = True,
) -> pd.DataFrame:
    """Positions along +x whose exact finite-difference speed is the profile."""

    step = speed_kmh_profile * DT
    x = np.concatenate(([0.0], np.cumsum(step)))
    n = len(x)
    return pd.DataFrame(
        {
            "match_id": 1,
            "period_id": np.int8(period_id),
            "frame_id": np.arange(n, dtype=np.int32) + frame_offset,
            "timestamp": (np.arange(n) * DT + frame_offset * DT).astype("float32"),
            "ball_state": pd.Categorical(
                ["alive" if alive else "dead"] * n, categories=["alive", "dead"]
            ),
            "ball_owning_team_id": pd.array([10] * n, dtype="Int32"),
            "player_id": np.int32(player_id),
            "team_id": np.int32(10),
            "x": x.astype("float32") - 50.0,
            "y": np.zeros(n, dtype="float32"),
            "is_detected": True,
        }
    )


class FindEffortsTests(unittest.TestCase):
    """One activity, sustained for at least 0.7 s, counted once."""

    def _count(self, profile: np.ndarray, threshold: float = SPRINT, **kwargs) -> int:
        mask = profile > threshold * KMH
        first, _, _ = find_efforts(
            mask, np.zeros(len(profile), dtype=np.int32), dt=DT,
            min_duration=kwargs.pop("min_duration", 0.7), **kwargs,
        )
        return len(first)

    def test_constant_speed_run_is_one_effort(self):
        self.assertEqual(self._count(speed_profile((27.0, 3.0))), 1)

    def test_run_shorter_than_minimum_does_not_count(self):
        self.assertEqual(self._count(speed_profile((27.0, 0.6))), 0)

    def test_run_at_exactly_the_minimum_counts(self):
        # 7 samples above threshold is 0.7 s of above-threshold time.
        self.assertEqual(self._count(speed_profile((27.0, 0.7))), 1)

    def test_span_rule_needs_one_more_sample_than_samples_rule(self):
        profile = speed_profile((27.0, 0.7))
        self.assertEqual(
            self._count(profile, duration_rule=EFFORT_DURATION_SAMPLES), 1
        )
        self.assertEqual(self._count(profile, duration_rule=EFFORT_DURATION_SPAN), 0)
        self.assertEqual(
            self._count(speed_profile((27.0, 0.8)), duration_rule=EFFORT_DURATION_SPAN), 1
        )

    def test_dip_splits_effort_when_tolerance_is_zero(self):
        # Two 0.5 s halves either side of a 0.2 s dip: neither half qualifies.
        profile = speed_profile((27.0, 0.5), (24.0, 0.2), (27.0, 0.5))
        self.assertEqual(self._count(profile, dip_tolerance=0.0), 0)

    def test_dip_within_tolerance_merges_into_one_effort(self):
        profile = speed_profile((27.0, 0.5), (24.0, 0.2), (27.0, 0.5))
        self.assertEqual(self._count(profile, dip_tolerance=0.2), 1)

    def test_dip_longer_than_tolerance_still_splits(self):
        profile = speed_profile((27.0, 0.5), (24.0, 0.4), (27.0, 0.5))
        self.assertEqual(self._count(profile, dip_tolerance=0.2), 0)

    def test_merged_effort_duration_excludes_the_dip(self):
        # 0.4 s + dip 0.1 s + 0.2 s is 0.6 s above threshold, not 0.7 s.
        profile = speed_profile((27.0, 0.4), (24.0, 0.1), (27.0, 0.2))
        self.assertEqual(self._count(profile, dip_tolerance=0.1), 0)
        profile = speed_profile((27.0, 0.4), (24.0, 0.1), (27.0, 0.3))
        self.assertEqual(self._count(profile, dip_tolerance=0.1), 1)

    def test_two_separate_efforts_are_counted_separately(self):
        profile = speed_profile((27.0, 1.0), (10.0, 3.0), (27.0, 1.0))
        self.assertEqual(self._count(profile), 2)

    def test_effort_never_spans_a_tracking_gap(self):
        # One continuous above-threshold run, but the segment label changes
        # halfway: a gap must end the activity and each side must qualify alone.
        profile = speed_profile((27.0, 1.6))
        segment = np.repeat([0, 1], len(profile) // 2).astype(np.int32)
        first, _, _ = find_efforts(
            profile > SPRINT * KMH, segment, dt=DT, min_duration=0.7
        )
        self.assertEqual(len(first), 2)

    def test_gap_splitting_can_disqualify_both_halves(self):
        profile = speed_profile((27.0, 0.8))
        segment = np.repeat([0, 1], len(profile) // 2).astype(np.int32)
        first, _, _ = find_efforts(
            profile > SPRINT * KMH, segment, dt=DT, min_duration=0.7
        )
        self.assertEqual(len(first), 0)

    def test_effort_ending_on_the_final_sample_is_counted(self):
        first, last, _ = find_efforts(
            speed_profile((27.0, 1.0)) > SPRINT * KMH,
            np.zeros(10, dtype=np.int32), dt=DT, min_duration=0.7,
        )
        self.assertEqual(len(first), 1)
        self.assertEqual(int(last[0]), 9)


class BipAttributionTests(unittest.TestCase):
    def test_effort_mostly_out_of_play_is_not_counted(self):
        profile = speed_profile((27.0, 1.0))
        segment = np.zeros(len(profile), dtype=np.int32)
        config = PhysicalConfig(estimator="central")
        mask = profile > SPRINT * KMH
        alive = np.zeros(len(profile), dtype=bool)
        alive[:2] = True
        self.assertEqual(count_efforts(mask, segment, alive, config), 0)
        alive[:] = True
        self.assertEqual(count_efforts(mask, segment, alive, config), 1)


class BandTests(unittest.TestCase):
    """Band edges follow the published wording exactly."""

    def _bands(self, kmh: float) -> dict[str, float]:
        tracking = straight_line_tracking(speed_profile((kmh, 6.0)))
        config = PhysicalConfig(estimator="central")
        row = compute_player_match_metrics(
            add_kinematics(canonicalize(tracking), config), config=config, prepared=True
        ).iloc[0]
        return {
            "running": row["running_distance_m"],
            "hsr": row["hsr_distance_m"],
            "sprint": row["sprint_distance_m"],
            "high_intensity": row["high_intensity_distance_m"],
            "total": row["total_distance_m"],
        }

    def test_constant_running_speed_lands_only_in_running(self):
        bands = self._bands(18.0)
        self.assertAlmostEqual(bands["running"], bands["total"], places=6)
        self.assertEqual(bands["hsr"], 0.0)
        self.assertEqual(bands["sprint"], 0.0)

    def test_lower_running_edge_is_inclusive(self):
        self.assertGreater(self._bands(15.0)["running"], 0.0)
        self.assertEqual(self._bands(14.9)["running"], 0.0)

    def test_hsr_lower_edge_is_inclusive_and_sprint_edge_is_strict(self):
        self.assertGreater(self._bands(20.0)["hsr"], 0.0)
        self.assertEqual(self._bands(20.0)["running"], 0.0)
        self.assertGreater(self._bands(25.0)["hsr"], 0.0)
        self.assertEqual(self._bands(25.0)["sprint"], 0.0)
        self.assertGreater(self._bands(25.1)["sprint"], 0.0)

    def test_high_intensity_is_the_union_of_hsr_and_sprint(self):
        for kmh in (21.0, 27.0):
            bands = self._bands(kmh)
            self.assertAlmostEqual(
                bands["high_intensity"], bands["hsr"] + bands["sprint"], places=6
            )

    def test_bands_sum_to_total_distance(self):
        tracking = straight_line_tracking(
            speed_profile((8.0, 5.0), (17.0, 3.0), (22.0, 2.0), (27.0, 2.0))
        )
        config = PhysicalConfig(estimator="central")
        row = compute_player_match_metrics(
            add_kinematics(canonicalize(tracking), config), config=config, prepared=True
        ).iloc[0]
        parts = (
            row["walking_jogging_distance_m"] + row["running_distance_m"]
            + row["hsr_distance_m"] + row["sprint_distance_m"]
        )
        self.assertAlmostEqual(parts, row["total_distance_m"], places=4)


class HighIntensityCountTests(unittest.TestCase):
    def test_high_intensity_is_hsr_plus_sprint_activities(self):
        """Legacy Physical v2 defines Count HI as HSR count + Sprint count."""

        profile = speed_profile(
            (10.0, 3.0), (22.0, 1.0), (27.0, 1.5), (22.0, 1.0), (10.0, 3.0)
        )
        tracking = straight_line_tracking(profile)
        config = PhysicalConfig(estimator="central")
        row = compute_player_match_metrics(
            add_kinematics(canonicalize(tracking), config), config=config, prepared=True
        ).iloc[0]
        self.assertEqual(row["sprint_count"], 1)
        self.assertEqual(row["hsr_count"], 2)
        self.assertEqual(row["high_intensity_count"], 3)
        self.assertEqual(
            row["high_intensity_count"], row["hsr_count"] + row["sprint_count"]
        )


class PsvTests(unittest.TestCase):
    def test_psv_uses_activity_peaks_not_frames(self):
        # Three activities above 15 km/h with peaks 18, 22 and 30 km/h.
        profile = speed_profile(
            (18.0, 2.0), (5.0, 2.0), (22.0, 2.0), (5.0, 2.0), (30.0, 2.0)
        )
        config = PhysicalConfig(estimator="central")
        segment = np.zeros(len(profile), dtype=np.int32)
        peak = psv(profile, segment, config)
        self.assertAlmostEqual(peak, 29.84, places=6)

    def test_psv_ignores_peaks_above_the_plausibility_bound(self):
        profile = speed_profile((30.0, 2.0), (5.0, 2.0), (60.0, 2.0))
        config = PhysicalConfig(estimator="central", psv_max_kmh=37.0)
        segment = np.zeros(len(profile), dtype=np.int32)
        self.assertAlmostEqual(psv(profile, segment, config), 30.0, places=6)

    def test_public_psv_range_includes_39_and_excludes_above_40(self):
        profile = speed_profile((39.0, 2.0), (5.0, 2.0), (41.0, 2.0))
        config = PhysicalConfig(estimator="central")
        segment = np.zeros(len(profile), dtype=np.int32)
        self.assertAlmostEqual(psv(profile, segment, config), 39.0, places=6)


class SegmentAndCapTests(unittest.TestCase):
    def test_nothing_is_derived_across_a_period_boundary(self):
        first = straight_line_tracking(speed_profile((18.0, 2.0)), period_id=1)
        second = straight_line_tracking(
            speed_profile((18.0, 2.0)), period_id=2, frame_offset=1000
        )
        # Teleport the second period so a cross-period difference would be huge.
        second["x"] = second["x"] + 40.0
        config = PhysicalConfig(estimator="central")
        kinematics = add_kinematics(
            canonicalize(pd.concat([first, second], ignore_index=True)), config
        )
        self.assertEqual(kinematics["segment_id"].nunique(), 2)
        self.assertLess(kinematics["speed_mps"].max() * 3.6, 20.0)

    def test_long_gap_starts_a_new_segment(self):
        first = straight_line_tracking(speed_profile((18.0, 2.0)))
        second = straight_line_tracking(speed_profile((18.0, 2.0)), frame_offset=500)
        second["timestamp"] = second["timestamp"] + 50.0
        config = PhysicalConfig(estimator="central", max_gap_seconds=0.15)
        kinematics = add_kinematics(
            canonicalize(pd.concat([first, second], ignore_index=True)), config
        )
        self.assertEqual(kinematics["segment_id"].nunique(), 2)

    def test_speed_cap_marks_invalid_and_does_not_clip(self):
        tracking = straight_line_tracking(speed_profile((18.0, 2.0), (50.0, 0.5)))
        config = PhysicalConfig(estimator="central", max_speed_kmh=37.0)
        kinematics = add_kinematics(canonicalize(tracking), config)
        self.assertTrue(kinematics["capped_speed"].any())
        capped = kinematics.loc[kinematics["capped_speed"]]
        self.assertTrue(capped["speed_mps"].isna().all())
        self.assertLessEqual(kinematics["speed_mps"].max() * 3.6, 37.0 + 1e-6)


class NormalizationTests(unittest.TestCase):
    def test_p60_bip_divides_by_measured_ball_in_play_time(self):
        # 30 s alive at 18 km/h then 30 s dead at 18 km/h.  The rate must use
        # only the alive half, so it is independent of the dead-ball tail.
        alive = straight_line_tracking(speed_profile((18.0, 30.0)), alive=True)
        dead = straight_line_tracking(
            speed_profile((18.0, 30.0)), frame_offset=1000, alive=False
        )
        dead["timestamp"] = dead["timestamp"] + 100.0
        config = PhysicalConfig(estimator="central")
        row = compute_player_match_metrics(
            add_kinematics(
                canonicalize(pd.concat([alive, dead], ignore_index=True)), config
            ),
            config=config,
            prepared=True,
        ).iloc[0]
        self.assertAlmostEqual(row["bip_minutes"], 30.0 / 60.0, places=3)
        self.assertAlmostEqual(row["total_distance_m_p60_bip"], 18000.0, delta=60.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
