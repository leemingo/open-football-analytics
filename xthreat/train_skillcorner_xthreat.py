"""Fit and score a SkillCorner action-level expected threat model."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xthreat.skillcorner_actions import DEFAULT_ACTIONS_PATH, DEFAULT_OUTPUT_DIR
from xthreat.xthreat_model import CenterOriginExpectedThreat


DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "skillcorner_xthreat_model.joblib"
DEFAULT_SCORED_ACTIONS_PATH = DEFAULT_OUTPUT_DIR / "scored_actions.parquet"
DEFAULT_SURFACE_PATH = DEFAULT_OUTPUT_DIR / "surface.json"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "metrics.json"
DEFAULT_TEAM_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "team_xthreat.csv"
DEFAULT_PLAYER_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "player_xthreat.csv"

MOVE_ACTIONS = {"pass", "carry", "drive", "dribble", "cross"}


def fit_skillcorner_xthreat(
    actions: pd.DataFrame,
    *,
    l: int = 16,
    w: int = 12,
    eps: float = 1e-5,
    max_iter: int = 200,
    unsuccessful: str = "nan",
) -> tuple[CenterOriginExpectedThreat, pd.DataFrame, dict]:
    """Fit a center-origin xT surface and score every action."""
    model = CenterOriginExpectedThreat(l=l, w=w, eps=eps, max_iter=max_iter).fit(actions)
    scored = model.rate(actions, unsuccessful=unsuccessful)
    metrics = build_xthreat_metrics(scored, model, unsuccessful=unsuccessful)
    return model, scored, metrics


def build_xthreat_metrics(
    scored: pd.DataFrame,
    model: CenterOriginExpectedThreat,
    *,
    unsuccessful: str,
) -> dict:
    """Return compact metrics and diagnostics for a fitted xT run."""
    action_type = scored["action_type"].astype("string") if "action_type" in scored.columns else pd.Series(dtype="string")
    move_mask = action_type.isin(MOVE_ACTIONS)
    value = pd.to_numeric(scored.get("custom_xT_added", pd.Series(np.nan, index=scored.index)), errors="coerce")
    rated_moves = int((move_mask & value.notna()).sum())
    diagnostics = asdict(model.diagnostics) if model.diagnostics is not None else {}
    metrics = {
        "model": "center_origin_expected_threat",
        "grid": {"l": model.l, "w": model.w},
        "eps": model.eps,
        "max_iter": model.max_iter,
        "unsuccessful": unsuccessful,
        "diagnostics": diagnostics,
        "actions": {
            "rows": int(len(scored)),
            "matches": int(scored["match_id"].astype("string").nunique()) if "match_id" in scored.columns else 0,
            "passes": int(action_type.eq("pass").sum()),
            "carries": int(action_type.eq("carry").sum()),
            "shots": int(action_type.eq("shot").sum()),
            "rated_moves": rated_moves,
            "custom_xT_added_sum": float(value.sum(skipna=True)),
            "custom_xT_added_per_100_rated_moves": float(value.sum(skipna=True) / max(rated_moves, 1) * 100.0),
        },
        "surface": {
            "min": float(np.nanmin(model.xT)) if model.xT.size else None,
            "max": float(np.nanmax(model.xT)) if model.xT.size else None,
            "mean": float(np.nanmean(model.xT)) if model.xT.size else None,
        },
    }
    comparison = compare_with_skillcorner_xthreat(scored)
    if comparison:
        metrics["skillcorner_pass_xthreat_comparison"] = comparison
    return metrics


def compare_with_skillcorner_xthreat(scored: pd.DataFrame) -> dict:
    """Compare custom target-cell xT with SkillCorner's supplied pass target xT."""
    required = {"action_type", "custom_xT_end", "skillcorner_target_xthreat"}
    if not required.issubset(scored.columns):
        return {}
    frame = scored[
        scored["action_type"].astype("string").eq("pass")
        & scored["custom_xT_end"].notna()
        & scored["skillcorner_target_xthreat"].notna()
    ].copy()
    if frame.empty:
        return {"rows": 0}
    custom = pd.to_numeric(frame["custom_xT_end"], errors="coerce")
    skillcorner = pd.to_numeric(frame["skillcorner_target_xthreat"], errors="coerce")
    diff = custom - skillcorner
    return {
        "rows": int(len(frame)),
        "pearson_corr": float(custom.corr(skillcorner, method="pearson")),
        "spearman_corr": float(custom.corr(skillcorner, method="spearman")),
        "mean_custom_minus_skillcorner": float(diff.mean()),
        "mae_custom_vs_skillcorner": float(diff.abs().mean()),
        "rmse_custom_vs_skillcorner": float(np.sqrt(np.mean(diff**2))),
    }


def summarize_xthreat(
    scored: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str = "custom_xT_added",
    min_actions: int = 0,
) -> pd.DataFrame:
    """Aggregate xT-added by team, player, or any grouping columns."""
    existing_group_cols = [col for col in group_cols if col in scored.columns]
    if not existing_group_cols:
        return pd.DataFrame()
    frame = scored.copy()
    frame[value_col] = pd.to_numeric(frame.get(value_col, pd.Series(np.nan, index=frame.index)), errors="coerce")
    frame["is_rated_move"] = frame["action_type"].astype("string").isin(MOVE_ACTIONS) & frame[value_col].notna()
    frame["positive_xT_added"] = frame[value_col].clip(lower=0.0)
    summary = (
        frame.groupby(existing_group_cols, dropna=False)
        .agg(
            actions=("event_id", "size"),
            matches=("match_id", "nunique") if "match_id" in frame.columns else ("event_id", "size"),
            rated_moves=("is_rated_move", "sum"),
            custom_xT_added=(value_col, "sum"),
            positive_xT_added=("positive_xT_added", "sum"),
            avg_custom_xT_added=(value_col, "mean"),
        )
        .reset_index()
    )
    counts = (
        frame.groupby(existing_group_cols + ["action_type"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .rename(columns=lambda col: f"n_{col}")
        .reset_index()
    )
    summary = summary.merge(counts, on=existing_group_cols, how="left")
    summary["custom_xT_added_per_100_moves"] = (
        summary["custom_xT_added"] / summary["rated_moves"].replace(0, np.nan) * 100.0
    )
    summary["positive_xT_added_per_100_moves"] = (
        summary["positive_xT_added"] / summary["rated_moves"].replace(0, np.nan) * 100.0
    )
    if min_actions:
        summary = summary[summary["actions"] >= int(min_actions)].copy()
    return summary.sort_values("custom_xT_added_per_100_moves", ascending=False, ignore_index=True)


def write_xthreat_outputs(
    model: CenterOriginExpectedThreat,
    scored: pd.DataFrame,
    metrics: dict,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scored_path: str | Path = DEFAULT_SCORED_ACTIONS_PATH,
    surface_path: str | Path = DEFAULT_SURFACE_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    team_summary_path: str | Path = DEFAULT_TEAM_SUMMARY_PATH,
    player_summary_path: str | Path = DEFAULT_PLAYER_SUMMARY_PATH,
    min_actions: int = 0,
) -> dict[str, Path]:
    """Persist model, scored actions, metrics, and aggregate summaries."""
    paths = {
        "model": Path(model_path),
        "scored": Path(scored_path),
        "surface": Path(surface_path),
        "metrics": Path(metrics_path),
        "team_summary": Path(team_summary_path),
        "player_summary": Path(player_summary_path),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths["model"])
    scored.to_parquet(paths["scored"], index=False)
    model.save_surface(paths["surface"])
    paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    team_group_cols = [col for col in ["season_name", "team_id", "team_shortname"] if col in scored.columns]
    player_group_cols = [col for col in ["season_name", "team_id", "team_shortname", "player_id", "player_name"] if col in scored.columns]
    summarize_xthreat(scored, group_cols=team_group_cols, min_actions=min_actions).to_csv(paths["team_summary"], index=False)
    summarize_xthreat(scored, group_cols=player_group_cols, min_actions=min_actions).to_csv(paths["player_summary"], index=False)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and score SkillCorner xThreat.")
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--l", type=int, default=16, help="Number of pitch cells along the x axis.")
    parser.add_argument("--w", type=int, default=12, help="Number of pitch cells along the y axis.")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--unsuccessful", choices=["nan", "zero", "negative_start"], default="nan")
    parser.add_argument("--min-actions", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    actions = pd.read_parquet(args.actions)
    model, scored, metrics = fit_skillcorner_xthreat(
        actions,
        l=args.l,
        w=args.w,
        eps=args.eps,
        max_iter=args.max_iter,
        unsuccessful=args.unsuccessful,
    )
    paths = write_xthreat_outputs(
        model,
        scored,
        metrics,
        model_path=args.out_dir / DEFAULT_MODEL_PATH.name,
        scored_path=args.out_dir / DEFAULT_SCORED_ACTIONS_PATH.name,
        surface_path=args.out_dir / DEFAULT_SURFACE_PATH.name,
        metrics_path=args.out_dir / DEFAULT_METRICS_PATH.name,
        team_summary_path=args.out_dir / DEFAULT_TEAM_SUMMARY_PATH.name,
        player_summary_path=args.out_dir / DEFAULT_PLAYER_SUMMARY_PATH.name,
        min_actions=args.min_actions,
    )
    print(f"[xT] model: {paths['model']}")
    print(f"[xT] scored actions: {paths['scored']} ({len(scored):,} rows)")
    print(f"[xT] surface: {paths['surface']}")
    print(f"[xT] metrics: {paths['metrics']}")
    print(f"[xT] team summary: {paths['team_summary']}")
    print(f"[xT] player summary: {paths['player_summary']}")


if __name__ == "__main__":
    main()
