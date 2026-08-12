from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from physical import (
    PhysicalConfig,
    aggregate_p60,
    compute_player_match_metrics,
    compute_rate_sensitivity,
    prepare_kinematics,
)
from physical.build_skillcorner_physical import build_player_profiles
from physical.physical_features import activity_peak_speeds


def constant_speed_tracking(
    speed_mps: float,
    *,
    duration: float = 10.0,
    dt: float = 0.1,
    player_id: str = "p1",
    period: int = 1,
    x_offset: float = 0.0,
) -> pd.DataFrame:
    times = np.arange(0.0, duration + dt / 2.0, dt)
    return pd.DataFrame(
        {
            "match_id": "m1",
            "period": period,
            "timestamp": times,
            "frame_id": np.arange(len(times)),
            "player_id": player_id,
            "x": x_offset + speed_mps * times,
            "y": 0.0,
            "ball_state": "alive",
            "ball_owning_team_id": "home",
        }
    )


class KinematicsTests(unittest.TestCase):
    def test_football_cdf_column_aliases_and_ball_filter(self) -> None:
        player = pd.DataFrame(
            {
                "game_id": "m1",
                "period_id": 1,
                "timestamp": [0.0, 0.1, 0.2],
                "frame_id": [0, 1, 2],
                "object_id": "home_7",
                "x": [0.0, 0.5, 1.0],
                "y": 0.0,
                "ball": False,
                "ball_state": True,
                "ball_owning_team_id": 0,
            }
        )
        ball = player.copy()
        ball["object_id"] = "ball"
        ball["ball"] = True
        result = compute_player_match_metrics(
            pd.concat([player, ball], ignore_index=True)
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "player_id"], "home_7")
        self.assertAlmostEqual(result.loc[0, "total_distance_m"], 1.0)
        self.assertEqual(result.loc[0, "bip_source"], "ball_state")

    def test_stationary_and_constant_speed_distance(self) -> None:
        stationary = constant_speed_tracking(0.0, player_id="still")
        moving = constant_speed_tracking(5.0, player_id="moving")
        tracking = pd.concat([stationary, moving], ignore_index=True)
        metrics = compute_player_match_metrics(tracking)
        by_player = metrics.set_index("player_id")

        self.assertAlmostEqual(by_player.loc["still", "total_distance_m"], 0.0, places=9)
        self.assertAlmostEqual(by_player.loc["moving", "total_distance_m"], 50.0, places=6)
        self.assertAlmostEqual(
            by_player.loc["moving", "running_distance_m"], 50.0, places=6
        )

    def test_primary_resamples_native_30hz_and_labels_effective_rate(self) -> None:
        times = np.arange(0.0, 3.0 + 1e-9, 1.0 / 30.0)
        tracking = pd.DataFrame(
            {
                "match_id": "m1",
                "period": 1,
                "timestamp": times,
                "player_id": "p1",
                "x": 5.0 * times,
                "y": 0.0,
                "ball_state": "alive",
                "ball_owning_team_id": "home",
            }
        )

        prepared = prepare_kinematics(
            tracking, config=PhysicalConfig(target_hz=10.0)
        )
        result = compute_player_match_metrics(
            tracking, config=PhysicalConfig(target_hz=10.0)
        ).iloc[0]

        np.testing.assert_allclose(np.diff(prepared["timestamp"]), 0.1)
        self.assertEqual(result["sample_rate"], "10Hz")
        self.assertEqual(result["effective_sample_hz"], 10.0)
        self.assertAlmostEqual(result["total_distance_m"], 15.0, places=6)

    def test_native_irregular_timestamps_are_rejected(self) -> None:
        times = np.array([0.0, 0.07, 0.20, 0.31, 0.50])
        speed = 6.0
        tracking = pd.DataFrame(
            {
                "match_id": "m1",
                "period": 1,
                "timestamp": times,
                "player_id": "p1",
                "x": times * speed,
                "y": 0.0,
                "ball_state": "alive",
                "ball_owning_team_id": "home",
            }
        )
        config = PhysicalConfig(target_hz=None)
        with self.assertRaisesRegex(ValueError, "not on a uniform grid"):
            prepare_kinematics(tracking, config=config)
        with self.assertRaisesRegex(ValueError, "not on a uniform grid"):
            compute_rate_sensitivity(tracking)

    def test_long_gap_is_not_interpolated_or_counted(self) -> None:
        times = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
        speed = 5.0
        tracking = pd.DataFrame(
            {
                "match_id": "m1",
                "period": 1,
                "timestamp": times,
                "player_id": "p1",
                "x": speed * times,
                "y": 0.0,
                "ball_state": "alive",
                "ball_owning_team_id": "home",
            }
        )
        result = compute_player_match_metrics(
            tracking, config=PhysicalConfig(target_hz=10.0, max_gap_seconds=0.5)
        ).iloc[0]
        self.assertAlmostEqual(result.total_distance_m, 1.5, places=8)
        self.assertEqual(result.segment_count, 2)
        self.assertEqual(result.gap_count, 1)

    def test_period_break_never_creates_a_pitch_jump(self) -> None:
        first = constant_speed_tracking(7.0, duration=1.0, period=1)
        second = constant_speed_tracking(
            7.0, duration=1.0, period=2, x_offset=1000.0
        )
        result = compute_player_match_metrics(
            pd.concat([first, second], ignore_index=True)
        ).iloc[0]
        self.assertAlmostEqual(result.total_distance_m, 14.0, places=6)
        self.assertEqual(result.sprint_count, 2)
        self.assertEqual(result.segment_count, 2)

    def test_lineup_interval_prevents_substitution_bridging(self) -> None:
        tracking = constant_speed_tracking(5.0, duration=10.0)
        lineup = pd.DataFrame(
            {
                "match_id": ["m1"],
                "player_id": ["p1"],
                "team_id": ["home"],
                "position": ["CM"],
                "start_time": [2.0],
                "end_time": [8.0],
            }
        )
        result = compute_player_match_metrics(tracking, lineup=lineup).iloc[0]
        self.assertAlmostEqual(result.total_distance_m, 30.0, places=6)
        self.assertAlmostEqual(result.observed_seconds, 6.0, places=6)
        self.assertAlmostEqual(result.played_minutes, 0.1, places=6)
        self.assertEqual(result.played_time_source, "appearance_interval")
        self.assertEqual(result.team_id, "home")
        self.assertEqual(result.position, "CM")

    def test_played_time_prefers_guarded_provider_duration(self) -> None:
        tracking = constant_speed_tracking(5.0, duration=10.0)
        lineup = pd.DataFrame(
            {
                "match_id": ["m1"],
                "player_id": ["p1"],
                "appearance_start": [0.0],
                "appearance_end": [3900.0],
                "provider_playing_time_ms": [3_600_000.0],
            }
        )
        result = compute_player_match_metrics(tracking, lineup=lineup).iloc[0]
        self.assertEqual(result.played_minutes, 60.0)
        self.assertEqual(result.played_time_source, "provider_playing_time_ms")
        self.assertAlmostEqual(result.bip_minutes, 10.0 / 60.0)

    def test_invalid_played_time_sources_fall_back_safely(self) -> None:
        tracking = constant_speed_tracking(5.0, duration=10.0)
        invalid_provider = pd.DataFrame(
            {
                "match_id": ["m1"],
                "player_id": ["p1"],
                "appearance_start": [2.0],
                "appearance_end": [8.0],
                "provider_playing_time_ms": [-1.0],
            }
        )
        appearance = compute_player_match_metrics(
            tracking, lineup=invalid_provider
        ).iloc[0]
        self.assertAlmostEqual(appearance.played_seconds, 6.0)
        self.assertEqual(appearance.played_time_source, "appearance_interval")

        no_lineup = compute_player_match_metrics(tracking).iloc[0]
        self.assertAlmostEqual(no_lineup.played_seconds, 10.0)
        self.assertEqual(no_lineup.played_time_source, "observed_tracking")


class MetricDefinitionTests(unittest.TestCase):
    def test_activity_peak_speed_handles_effort_ending_at_final_sample(self) -> None:
        speed = np.array([0.0, 5.0, 6.0, 0.0, 7.0, 8.0])
        segment = np.zeros(speed.size, dtype=np.int64)

        peaks = activity_peak_speeds(speed, segment, PhysicalConfig())

        np.testing.assert_allclose(peaks, [6.0, 8.0])

    def test_speed_threshold_boundaries_do_not_overlap(self) -> None:
        cases = {
            "run15": 15.0 / 3.6,
            "hsr20": 20.0 / 3.6,
            "hsr25": 25.0 / 3.6,
            "sprint": 25.01 / 3.6,
        }
        tracking = pd.concat(
            [
                constant_speed_tracking(speed, duration=2.0, player_id=player)
                for player, speed in cases.items()
            ],
            ignore_index=True,
        )
        result = compute_player_match_metrics(tracking).set_index("player_id")

        self.assertGreater(result.loc["run15", "running_distance_m"], 0)
        self.assertEqual(result.loc["run15", "hsr_distance_m"], 0)
        self.assertGreater(result.loc["hsr20", "hsr_distance_m"], 0)
        self.assertEqual(result.loc["hsr20", "running_distance_m"], 0)
        self.assertGreater(result.loc["hsr25", "hsr_distance_m"], 0)
        self.assertEqual(result.loc["hsr25", "sprint_distance_m"], 0)
        self.assertGreater(result.loc["sprint", "sprint_distance_m"], 0)
        self.assertEqual(result.loc["sprint", "hsr_distance_m"], 0)

    def test_sprint_transition_is_one_bout(self) -> None:
        dt = 0.1
        times = np.arange(0, 12 + dt / 2, dt)
        interval_speed = np.where(
            times <= 4, 5.0, np.where(times <= 8, 8.0, 5.0)
        )
        x = np.zeros_like(times)
        x[1:] = np.cumsum(interval_speed[1:] * np.diff(times))
        tracking = pd.DataFrame(
            {
                "match_id": "m1",
                "period": 1,
                "timestamp": times,
                "player_id": "p1",
                "x": x,
                "y": 0.0,
                "ball_state": "alive",
                "ball_owning_team_id": "home",
            }
        )
        result = compute_player_match_metrics(tracking).iloc[0]
        self.assertEqual(result.sprint_count, 1)
        self.assertEqual(
            result.high_intensity_count,
            result.hsr_count + result.sprint_count,
        )
        self.assertEqual(result.hir_count, result.high_intensity_count)
        self.assertEqual(
            result.high_intensity_distance_m, result.hir_distance_m
        )
        self.assertGreater(result.sprint_distance_m, 0)
        self.assertGreater(result.psv99_kmh, 25.0)

    def test_speed_activity_minimum_duration_and_hi_sum(self) -> None:
        tracking = constant_speed_tracking(0.0, duration=4.0)
        prepared = prepare_kinematics(tracking)
        prepared["speed_mps"] = 0.0

        # Six 0.1-second intervals are threshold chatter and must not count.
        chatter = prepared["timestamp"].between(0.2, 0.7)
        prepared.loc[chatter, "speed_mps"] = 26.0 / 3.6

        # Exactly seven intervals in each band: both are valid activities.
        hsr = prepared["timestamp"].between(1.0, 1.6)
        sprint = prepared["timestamp"].between(1.7, 2.3 + 1e-9)
        prepared.loc[hsr, "speed_mps"] = 22.0 / 3.6
        prepared.loc[sprint, "speed_mps"] = 26.0 / 3.6

        result = compute_player_match_metrics(
            prepared, prepared=True
        ).iloc[0]
        self.assertEqual(result.hsr_count, 1)
        self.assertEqual(result.sprint_count, 1)
        self.assertEqual(result.high_intensity_count, 2)
        self.assertEqual(result.hir_count, 2)
        self.assertEqual(
            result.high_intensity_count_definition,
            "hsr_count_plus_sprint_count",
        )

    def test_high_acceleration_requires_point_seven_seconds(self) -> None:
        dt = 0.1
        times = np.arange(0, 4 + dt / 2, dt)
        acceleration = 3.5
        x = 0.5 * acceleration * times**2
        tracking = pd.DataFrame(
            {
                "match_id": "m1",
                "period": 1,
                "timestamp": times,
                "player_id": "p1",
                "x": x,
                "y": 0.0,
                "ball_state": "alive",
                "ball_owning_team_id": "home",
            }
        )
        config = PhysicalConfig(max_speed_kmh=80.0)
        result = compute_player_match_metrics(tracking, config=config).iloc[0]
        self.assertEqual(result.high_acceleration_count, 1)

        prepared = prepare_kinematics(tracking, config=config)
        prepared["acceleration_mps2"] = 0.0
        prepared.loc[
            prepared["timestamp"].between(1.0, 1.5), "acceleration_mps2"
        ] = 3.5
        short = compute_player_match_metrics(
            prepared, config=config, prepared=True
        ).iloc[0]
        self.assertEqual(short.high_acceleration_count, 0)

        prepared.loc[
            prepared["timestamp"].between(1.0, 1.6), "acceleration_mps2"
        ] = 3.5
        exact = compute_player_match_metrics(
            prepared, config=config, prepared=True
        ).iloc[0]
        self.assertEqual(exact.high_acceleration_count, 1)

    def test_high_acceleration_threshold_is_strictly_above_three(self) -> None:
        tracking = constant_speed_tracking(0.0, duration=3.0)
        prepared = prepare_kinematics(tracking)
        prepared["acceleration_mps2"] = 0.0
        effort = prepared["timestamp"].between(0.5, 1.3)
        prepared.loc[effort, "acceleration_mps2"] = 3.0
        at_threshold = compute_player_match_metrics(
            prepared, prepared=True
        ).iloc[0]
        self.assertEqual(at_threshold.high_acceleration_count, 0)

        prepared.loc[effort, "acceleration_mps2"] = 3.000001
        above_threshold = compute_player_match_metrics(
            prepared, prepared=True
        ).iloc[0]
        self.assertEqual(above_threshold.high_acceleration_count, 1)

    def test_bip_mask_controls_distance_and_denominator(self) -> None:
        tracking = constant_speed_tracking(5.0, duration=10.0)
        tracking.loc[tracking["timestamp"] > 5.0, "ball_state"] = "dead"
        result = compute_player_match_metrics(tracking).iloc[0]
        self.assertAlmostEqual(result.total_distance_m, 25.0, places=6)
        self.assertAlmostEqual(result.bip_seconds, 5.0, places=6)
        self.assertAlmostEqual(result.total_distance_m_p60_bip, 18000.0, places=4)
        self.assertAlmostEqual(result.total_distance_all_m, 50.0, places=6)

    def test_psv99_uses_15_to_40_kmh_all_activity(self) -> None:
        tracking = constant_speed_tracking(8.0, duration=10.0)
        result = compute_player_match_metrics(tracking).iloc[0]
        self.assertAlmostEqual(result.psv99_kmh, 28.8, places=6)
        running = constant_speed_tracking(5.0, duration=10.0)
        running.loc[running["timestamp"] > 5.0, "ball_state"] = "dead"
        running_result = compute_player_match_metrics(running).iloc[0]
        self.assertAlmostEqual(running_result.psv99_kmh, 18.0, places=6)
        below = compute_player_match_metrics(
            constant_speed_tracking(4.0, duration=10.0)
        ).iloc[0]
        self.assertTrue(np.isnan(below.psv99_kmh))

    def test_aggregate_filters_played_minutes_and_player_datapoints(self) -> None:
        rows = []
        for player, n_matches in (("eligible", 5), ("too_few", 4)):
            for index in range(n_matches):
                rows.append(
                    {
                        "match_id": f"{player}-{index}",
                        "player_id": player,
                        "observed_seconds": 3600.0,
                        "played_minutes": 60.0,
                        "bip_seconds": 2000.0,
                        "total_distance_m_p60_bip": 8000.0,
                        "psv99_kmh": 30.0,
                    }
                )
        # This otherwise eligible performance fails the 60 played-minute rule.
        rows.append(
            {
                "match_id": "eligible-short",
                "player_id": "eligible",
                "observed_seconds": 3599.0,
                "played_minutes": 59.0,
                "bip_seconds": 2500.0,
                "total_distance_m_p60_bip": 9999.0,
                "psv99_kmh": 40.0,
            }
        )
        result = aggregate_p60(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result.player_count, 2)
        self.assertEqual(result.performance_count, 9)
        self.assertEqual(result.total_distance_m_p60_bip, 8000.0)

    def test_profile_uses_strongest_recorded_peak(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "player_name": "Player",
                    "position_group": "Winger",
                    "match_id": "m1",
                    "played_minutes": 70.0,
                    "bip_minutes": 50.0,
                    "tip_minutes": 25.0,
                    "otip_minutes": 25.0,
                    "total_distance_m": 5000.0,
                    "psv99_kmh": 30.0,
                    "max_speed_kmh": 31.0,
                },
                {
                    "player_id": "p1",
                    "player_name": "Player",
                    "position_group": "Winger",
                    "match_id": "m2",
                    "played_minutes": 70.0,
                    "bip_minutes": 50.0,
                    "tip_minutes": 25.0,
                    "otip_minutes": 25.0,
                    "total_distance_m": 5000.0,
                    "psv99_kmh": 33.0,
                    "max_speed_kmh": 34.0,
                },
            ]
        )
        profile = build_player_profiles(rows).iloc[0]
        self.assertEqual(profile["psv99_kmh"], 33.0)
        self.assertEqual(profile["max_speed_kmh"], 34.0)


if __name__ == "__main__":
    unittest.main()
