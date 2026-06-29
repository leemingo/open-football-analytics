"""Feature engineering for SkillCorner xPass completion models."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from football_cdf.constants import PITCH_X, PITCH_Y


SHORT_PASS_MAX_M = 10.0
LONG_PASS_MIN_M = 30.0


NUMERIC_FEATURES = [
    "passer_x",
    "passer_y",
    "target_x",
    "target_y",
    "abs_passer_y",
    "abs_target_y",
    "passer_distance_to_goal",
    "target_distance_to_goal",
    "passer_distance_to_sideline",
    "target_distance_to_sideline",
    "pass_distance_feature",
    "pass_progression",
    "target_goal_distance_gain",
    "pass_lateral_distance",
    "pass_angle_model",
    "pass_angle_model_deg",
    "pass_angle_sin",
    "pass_angle_cos",
    "possession_start_to_pass_distance",
    "duration",
    "distance_covered",
    "speed_avg",
    "n_passing_options",
    "n_off_ball_runs",
    "n_passing_options_line_break",
    "n_passing_options_first_line_break",
    "n_passing_options_second_last_line_break",
    "n_passing_options_last_line_break",
    "n_passing_options_ahead",
    "n_teammates_ahead_start",
    "n_opponents_ahead_player_in_possession_pass_moment",
    "n_opponents_bypassed",
    "n_opponents_ahead_start",
    "last_defensive_line_x_start",
    "delta_to_last_defensive_line_start",
    "last_defensive_line_height_start",
]


BINARY_FEATURES = [
    "high_pass",
    "one_touch",
    "quick_pass",
    "carry",
    "forward_momentum",
    "hand_pass",
    "is_header",
    "is_short_pass",
    "is_medium_pass",
    "is_long_pass",
    "is_set_piece_start",
    "is_corner_start",
    "is_free_kick_start",
    "is_throw_in_start",
    "is_goal_kick_start",
    "first_player_possession_in_team_possession",
    "inside_defensive_shape_start",
    "organised_defense",
    "goal_side_start",
]


CATEGORICAL_FEATURES = [
    "start_type",
    "game_state",
    "team_in_possession_phase_type",
    "team_out_of_possession_phase_type",
    "player_position",
    "player_targeted_position",
    "speed_avg_band",
    "player_targeted_speed_avg_band",
    "player_targeted_channel_pass",
    "player_targeted_third_pass",
    "player_targeted_penalty_area_pass",
    "home_away",
]


TARGET_COLUMN = "pass_completed"
OUTCOME_COLUMN = "pass_outcome"
SKILLCORNER_XPASS_COLUMN = "skillcorner_xpass"


def _empty_series(index: pd.Index, value=np.nan, dtype: str | None = None) -> pd.Series:
    return pd.Series(value, index=index, dtype=dtype)


def _as_numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return _empty_series(df.index, default, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _as_bool(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    bool_alias = f"{column}_bool"
    source = bool_alias if bool_alias in df.columns else column
    if source not in df.columns:
        return _empty_series(df.index, default, dtype="bool")
    s = df[source]
    if s.dtype == bool:
        return s.fillna(default).astype(bool)
    mapped = (
        s.astype("string")
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    return mapped.fillna(default).astype(bool)


def _coalesce(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.where(left.notna(), right)


def add_xpass_features(passes: pd.DataFrame) -> pd.DataFrame:
    """Return ``passes`` with model-ready xPass feature columns added.

    Coordinates are expected to be centre-origin and attacker left-to-right, so
    the opponent goal is centred at ``(+52.5, 0)``. The feature set deliberately
    excludes SkillCorner's supplied xPass/xT/danger flags so that a custom
    model can be compared against them rather than learning from them.
    """
    out = passes.copy()

    out["passer_x"] = _as_numeric(out, "passer_x")
    out["passer_y"] = _as_numeric(out, "passer_y")
    out["target_x"] = _as_numeric(out, "target_x")
    out["target_y"] = _as_numeric(out, "target_y")
    out["possession_start_x"] = _as_numeric(out, "possession_start_x")
    out["possession_start_y"] = _as_numeric(out, "possession_start_y")

    out["abs_passer_y"] = out["passer_y"].abs()
    out["abs_target_y"] = out["target_y"].abs()

    goal_x = PITCH_X / 2.0
    sideline_y = PITCH_Y / 2.0

    out["passer_distance_to_goal"] = np.hypot(goal_x - out["passer_x"], -out["passer_y"])
    out["target_distance_to_goal"] = np.hypot(goal_x - out["target_x"], -out["target_y"])
    out["passer_distance_to_sideline"] = sideline_y - out["abs_passer_y"]
    out["target_distance_to_sideline"] = sideline_y - out["abs_target_y"]

    dx = out["target_x"] - out["passer_x"]
    dy = out["target_y"] - out["passer_y"]
    out["pass_distance_model"] = np.hypot(dx, dy)
    out["pass_distance_skillcorner"] = _as_numeric(out, "pass_distance")
    out["pass_distance_feature"] = out["pass_distance_model"]
    out["pass_progression"] = dx
    out["target_goal_distance_gain"] = out["passer_distance_to_goal"] - out["target_distance_to_goal"]
    out["pass_lateral_distance"] = dy.abs()
    out["pass_angle_model"] = np.arctan2(dy.abs(), dx)
    out["pass_angle_model_deg"] = np.degrees(out["pass_angle_model"])
    out["pass_angle_sin"] = np.sin(out["pass_angle_model"])
    out["pass_angle_cos"] = np.cos(out["pass_angle_model"])
    out["pass_angle_skillcorner"] = _as_numeric(out, "pass_angle")

    out["possession_start_to_pass_distance"] = np.hypot(
        out["passer_x"] - out["possession_start_x"],
        out["passer_y"] - out["possession_start_y"],
    )

    out["is_short_pass"] = out["pass_distance_feature"] < SHORT_PASS_MAX_M
    out["is_medium_pass"] = (
        out["pass_distance_feature"].ge(SHORT_PASS_MAX_M)
        & out["pass_distance_feature"].le(LONG_PASS_MIN_M)
    )
    out["is_long_pass"] = out["pass_distance_feature"] > LONG_PASS_MIN_M

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

    if SKILLCORNER_XPASS_COLUMN in out.columns:
        out[SKILLCORNER_XPASS_COLUMN] = pd.to_numeric(out[SKILLCORNER_XPASS_COLUMN], errors="coerce")

    return out


def get_model_feature_columns() -> tuple[list[str], list[str], list[str]]:
    """Return numeric, binary, and categorical feature column names."""
    return NUMERIC_FEATURES.copy(), BINARY_FEATURES.copy(), CATEGORICAL_FEATURES.copy()


def build_feature_matrix(passes: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Convenience helper for scripts and notebooks."""
    featured = add_xpass_features(passes)
    numeric, binary, categorical = get_model_feature_columns()
    feature_cols = numeric + binary + categorical
    X = featured[feature_cols].copy()
    y = featured[TARGET_COLUMN].astype(bool) if TARGET_COLUMN in featured.columns else None
    return X, y


def filter_modelled_passes(
    passes: pd.DataFrame,
    *,
    include_offside: bool = False,
) -> pd.DataFrame:
    """Keep rows with a clear binary completion outcome."""
    if OUTCOME_COLUMN not in passes.columns:
        raise ValueError(f"Missing required outcome column: {OUTCOME_COLUMN}")
    allowed = {"successful", "unsuccessful"}
    if include_offside:
        allowed.add("offside")
    return passes[passes[OUTCOME_COLUMN].astype("string").isin(allowed)].copy()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add xPass model features to a SkillCorner pass table.")
    parser.add_argument("--passes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-offside", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    passes = pd.read_parquet(args.passes)
    passes = filter_modelled_passes(passes, include_offside=args.include_offside)
    featured = add_xpass_features(passes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(args.out, index=False)
    print(f"[features] wrote {len(featured):,} rows to {args.out}")


if __name__ == "__main__":
    main()
