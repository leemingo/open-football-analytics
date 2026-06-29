"""Feature engineering for SkillCorner shot-level xG models."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from football_cdf.constants import PITCH_X, PITCH_Y


GOAL_WIDTH = 7.32
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_HALF_WIDTH = 20.16
SIX_YARD_DEPTH = 5.5
SIX_YARD_HALF_WIDTH = 9.16


NUMERIC_FEATURES = [
    "shot_x",
    "shot_y",
    "abs_shot_y",
    "distance_to_goal",
    "distance_to_goal_sq",
    "shot_angle",
    "shot_angle_deg",
    "shot_angle_sin",
    "shot_angle_cos",
    "possession_distance_to_shot",
    "distance_covered",
    "duration",
    "speed_avg",
    "trajectory_angle",
    "pass_distance_received",
    "n_passing_options",
    "n_off_ball_runs",
    "n_passing_options_line_break",
    "n_passing_options_ahead",
    "last_defensive_line_x_start",
    "last_defensive_line_x_end",
    "delta_to_last_defensive_line_start",
    "delta_to_last_defensive_line_end",
    "delta_to_last_defensive_line_gain",
    "last_defensive_line_height_start",
    "last_defensive_line_height_end",
    "last_defensive_line_height_gain",
    "n_teammates_ahead_start",
    "n_teammates_ahead_end",
]


BINARY_FEATURES = [
    "is_header",
    "one_touch",
    "quick_pass",
    "carry",
    "forward_momentum",
    "in_box",
    "in_six_yard_box",
    "central_shot",
    "wide_shot",
    "very_close_shot",
    "long_shot",
    "is_set_piece_start",
    "is_corner_start",
    "is_free_kick_start",
    "is_throw_in_start",
    "is_goal_kick_start",
    "inside_defensive_shape_start",
    "inside_defensive_shape_end",
    "organised_defense",
    "is_player_possession_start_matched",
    "is_player_possession_end_matched",
    "fully_extrapolated",
]


CATEGORICAL_FEATURES = [
    "start_type",
    "team_in_possession_phase_type",
    "game_state",
    "player_position",
    "channel_end",
    "third_end",
    "penalty_area_end",
    "pass_range_received",
    "trajectory_direction",
    "speed_avg_band",
    "home_away",
]


TARGET_COLUMN = "goal"


def _as_numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _as_bool(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    s = df[column]
    if s.dtype == bool:
        return s.fillna(default).astype(bool)
    mapped = (
        s.astype("string")
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    return mapped.fillna(default).astype(bool)


def shot_angle_to_goal(shot_x: pd.Series, shot_y: pd.Series) -> pd.Series:
    """Return the open angle to the goal mouth in radians.

    Coordinates are expected to be centre-origin and attacker left-to-right, so
    the goal is centred at ``(+52.5, 0)``.
    """
    x = pd.to_numeric(shot_x, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(shot_y, errors="coerce").to_numpy(dtype=float)
    goal_x = PITCH_X / 2.0
    upper = np.column_stack([goal_x - x, GOAL_WIDTH / 2.0 - y])
    lower = np.column_stack([goal_x - x, -GOAL_WIDTH / 2.0 - y])
    dot = np.einsum("ij,ij->i", upper, lower)
    denom = np.linalg.norm(upper, axis=1) * np.linalg.norm(lower, axis=1)
    cosang = np.divide(dot, denom, out=np.full_like(dot, np.nan, dtype=float), where=denom > 0)
    angle = np.arccos(np.clip(cosang, -1.0, 1.0))
    return pd.Series(angle, index=shot_x.index, dtype=float)


def add_xg_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Return ``shots`` with model-ready xG feature columns added."""
    out = shots.copy()

    out["shot_x"] = _as_numeric(out, "shot_x")
    out["shot_y"] = _as_numeric(out, "shot_y")
    out["abs_shot_y"] = out["shot_y"].abs()

    goal_x = PITCH_X / 2.0
    dx = goal_x - out["shot_x"]
    dy = -out["shot_y"]
    out["distance_to_goal"] = np.hypot(dx, dy)
    out["distance_to_goal_sq"] = out["distance_to_goal"] ** 2
    out["shot_angle"] = shot_angle_to_goal(out["shot_x"], out["shot_y"])
    out["shot_angle_deg"] = np.degrees(out["shot_angle"])
    out["shot_angle_sin"] = np.sin(out["shot_angle"])
    out["shot_angle_cos"] = np.cos(out["shot_angle"])

    start_x = _as_numeric(out, "possession_start_x")
    start_y = _as_numeric(out, "possession_start_y")
    out["possession_distance_to_shot"] = np.hypot(out["shot_x"] - start_x, out["shot_y"] - start_y)

    out["in_box"] = (out["shot_x"] >= goal_x - PENALTY_AREA_DEPTH) & (out["abs_shot_y"] <= PENALTY_AREA_HALF_WIDTH)
    out["in_six_yard_box"] = (out["shot_x"] >= goal_x - SIX_YARD_DEPTH) & (out["abs_shot_y"] <= SIX_YARD_HALF_WIDTH)
    out["central_shot"] = out["abs_shot_y"] <= 8.5
    out["wide_shot"] = out["abs_shot_y"] >= 20.16
    out["very_close_shot"] = out["distance_to_goal"] <= 8.0
    out["long_shot"] = out["distance_to_goal"] >= 25.0

    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in BINARY_FEATURES:
        out[col] = _as_bool(out, col)

    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = "missing"
        out[col] = out[col].astype("string").fillna("missing")

    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = _as_bool(out, TARGET_COLUMN)

    return out


def get_model_feature_columns() -> tuple[list[str], list[str], list[str]]:
    """Return numeric, binary, and categorical feature column names."""
    return NUMERIC_FEATURES.copy(), BINARY_FEATURES.copy(), CATEGORICAL_FEATURES.copy()


def build_feature_matrix(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Convenience helper for scripts/notebooks."""
    featured = add_xg_features(shots)
    numeric, binary, categorical = get_model_feature_columns()
    feature_cols = numeric + binary + categorical
    X = featured[feature_cols].copy()
    y = featured[TARGET_COLUMN].astype(bool) if TARGET_COLUMN in featured.columns else None
    return X, y


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add xG model features to a SkillCorner shot table.")
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    shots = pd.read_parquet(args.shots)
    featured = add_xg_features(shots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(args.out, index=False)
    print(f"[features] wrote {len(featured):,} rows to {args.out}")


if __name__ == "__main__":
    main()
