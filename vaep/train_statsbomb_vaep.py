"""Train VAEP on StatsBomb Open Data and export player ratings.

Builds the SPADL action table for one StatsBomb Open Data competition+season
(default: the full FIFA World Cup 2022, 64 matches), fits the two
P(scores)/P(concedes) heads, scores every action, applies the VAEP formula, and
aggregates to per-player ratings. This is a transparent, reproducible example of the
*method* rather than a league-strength model -- the same caveat the xg/xpass/xthreat
tutorials carry. The split holds out the last N matches (by StatsBomb match order).

Outputs (under ``tmp/data/vaep_statsbomb`` by default):

* ``scored_actions.parquet`` -- action-level VAEP (ids + values + probs + split)
* ``player_ratings.parquet`` -- per-player totals, phase split, goals
* ``metrics.json``           -- model quality per head/split + run config
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vaep.statsbomb_actions import (
    DEFAULT_COMPETITION_ID,
    DEFAULT_SEASON_ID,
    SHOT_TYPES,
    build_actions,
    player_directory,
)
from vaep.vaep_features import NB_PREV_ACTIONS, compute_feature_matrix
from vaep.vaep_formula import compute_values
from vaep.vaep_labels import NR_ACTIONS, add_labels
from vaep.vaep_model import VAEPModel

DEFAULT_OUTPUT_DIR = Path("tmp/data/vaep_statsbomb")
DEFAULT_CACHE_DIR = Path("tmp/data/statsbomb_open")

ACTION_PHASE = {
    "pass": "pass", "cross": "pass", "throw_in": "pass",
    "freekick_short": "pass", "freekick_crossed": "pass",
    "corner_short": "pass", "corner_crossed": "pass", "goalkick": "pass",
    "dribble": "carry", "take_on": "carry",
    "shot": "shot", "shot_penalty": "shot", "shot_freekick": "shot",
    "tackle": "defend", "interception": "defend", "clearance": "defend",
    "ball_recovery": "defend", "dispossessed": "defend", "bad_touch": "defend",
    "foul": "defend", "shot_block": "defend",
    "keeper_save": "defend", "keeper_claim": "defend", "keeper_punch": "defend",
}


def _phase_pivot(scored: pd.DataFrame, key: str) -> pd.DataFrame:
    phase = scored["type_name"].map(ACTION_PHASE).fillna("other")
    tmp = scored[[key]].copy()
    tmp["phase"] = phase.to_numpy()
    tmp["vaep_value"] = scored["vaep_value"].to_numpy()
    return (tmp.pivot_table(index=key, columns="phase", values="vaep_value", aggfunc="sum", fill_value=0.0)
            .rename(columns=lambda c: f"vaep_{c}").reset_index())


def aggregate_players(scored: pd.DataFrame, directory: pd.DataFrame) -> pd.DataFrame:
    is_goal = scored["type_name"].isin(SHOT_TYPES) & (scored["result_name"] == "success")
    base = scored.assign(is_goal=is_goal.to_numpy())
    agg = (base.groupby("player_id", dropna=False)
           .agg(team_id=("team_id", "first"), matches=("match_id", "nunique"),
                actions=("vaep_value", "size"), vaep=("vaep_value", "sum"),
                offensive=("offensive_value", "sum"), defensive=("defensive_value", "sum"),
                goals=("is_goal", "sum"))
           .reset_index())
    agg = agg.merge(_phase_pivot(base, "player_id"), on="player_id", how="left")
    agg["vaep_per_action"] = agg["vaep"] / agg["actions"].clip(lower=1)
    dir_small = directory.drop_duplicates("player_id")[[c for c in ["player_id", "player_name", "playing_position"] if c in directory.columns]]
    agg = agg.merge(dir_small, on="player_id", how="left")
    agg["player_name"] = agg.get("player_name").fillna(agg["player_id"].astype(str)) if "player_name" in agg else agg["player_id"].astype(str)
    return agg.sort_values("vaep", ascending=False, ignore_index=True)


def run(*, match_ids: list[str] | None, competition_id: int, season_id: int, test_matches: int,
        model_name: str, out_dir: Path, cache_dir: Path, nb_prev: int, nr_actions: int) -> None:
    actions = build_actions(match_ids, competition_id=competition_id, season_id=season_id, cache_dir=cache_dir)
    print(f"[vaep] built {len(actions):,} actions / {actions['match_id'].nunique()} matches", flush=True)

    X = compute_feature_matrix(actions, nb_prev=nb_prev)
    y_scores, y_concedes = add_labels(actions, nr_actions=nr_actions)

    ordered = list(dict.fromkeys(actions["match_id"]))  # match order
    test_ids = set(ordered[-max(1, test_matches):])
    test_mask = actions["match_id"].isin(test_ids).to_numpy()
    train_mask = ~test_mask

    model = VAEPModel(model_name, n_train=int(train_mask.sum()))
    print(f"[vaep] fit {model.model_name} on {int(train_mask.sum()):,} actions ({X.shape[1]} feats); "
          f"scores base={y_scores[train_mask].mean():.4f} concedes base={y_concedes[train_mask].mean():.4f}", flush=True)
    model.fit(X.iloc[train_mask], y_scores[train_mask], y_concedes[train_mask])

    p_scores, p_concedes = model.predict(X)
    values = compute_values(actions, p_scores, p_concedes)
    scored = actions.copy()
    for col in ["offensive_value", "defensive_value", "vaep_value"]:
        scored[col] = values[col].to_numpy()
    scored["p_scores"] = p_scores
    scored["p_concedes"] = p_concedes
    scored["split"] = np.where(train_mask, "train", "test")

    metrics = {"model": model.model_name, "matches": actions["match_id"].nunique(),
               "test_matches": sorted(test_ids), "n_actions": int(len(actions)),
               "n_features": int(X.shape[1]), "nb_prev_actions": nb_prev, "nr_actions": nr_actions}
    metrics.update(model.evaluate(X.iloc[train_mask], y_scores[train_mask], y_concedes[train_mask], split="train"))
    if test_mask.any():
        metrics.update(model.evaluate(X.iloc[test_mask], y_scores[test_mask], y_concedes[test_mask], split="test"))

    directory = player_directory(match_ids, competition_id=competition_id, season_id=season_id, cache_dir=cache_dir)
    players = aggregate_players(scored, directory)

    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out_dir / "scored_actions.parquet", index=False)
    players.to_parquet(out_dir / "player_ratings.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[vaep] wrote outputs to {out_dir}")
    print(f"[vaep] scores AUC train={metrics.get('train_scores',{}).get('auc')} "
          f"test={metrics.get('test_scores',{}).get('auc')}")
    cols = ["player_name", "playing_position", "matches", "actions", "vaep", "offensive", "defensive", "goals"]
    cols = [c for c in cols if c in players.columns]
    print("\n[vaep] Top 15 players by total VAEP (all matches):")
    print(players.head(15)[cols].to_string(index=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAEP on StatsBomb Open Data.")
    parser.add_argument("--match-ids", nargs="+", default=None, help="StatsBomb match ids (default: every match in --competition-id/--season-id)")
    parser.add_argument("--competition-id", type=int, default=DEFAULT_COMPETITION_ID, help="StatsBomb competition_id (default: 43 = FIFA World Cup)")
    parser.add_argument("--season-id", type=int, default=DEFAULT_SEASON_ID, help="StatsBomb season_id (default: 106 = 2022)")
    parser.add_argument("--test-matches", type=int, default=8, help="hold out the last N matches")
    parser.add_argument("--model", default="xgboost", help="xgboost | logistic")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--nb-prev", type=int, default=NB_PREV_ACTIONS)
    parser.add_argument("--nr-actions", type=int, default=NR_ACTIONS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(match_ids=args.match_ids, competition_id=args.competition_id, season_id=args.season_id,
        test_matches=args.test_matches, model_name=args.model, out_dir=args.out_dir,
        cache_dir=args.cache_dir, nb_prev=args.nb_prev, nr_actions=args.nr_actions)


if __name__ == "__main__":
    main()
