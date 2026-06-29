"""Train and compare SkillCorner pass-level xPass completion models."""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xpass.skillcorner_passes import DEFAULT_OUTPUT_DIR, DEFAULT_PASSES_PATH
from xpass.xpass_features import (
    SKILLCORNER_XPASS_COLUMN,
    TARGET_COLUMN,
    add_xpass_features,
    filter_modelled_passes,
    get_model_feature_columns,
)


DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "skillcorner_xpass_best.joblib"
DEFAULT_SCORED_PASSES_PATH = DEFAULT_OUTPUT_DIR / "scored_passes.parquet"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "metrics.json"
DEFAULT_COMPARISON_PATH = DEFAULT_OUTPUT_DIR / "model_comparison.csv"
DEFAULT_SKILLCORNER_COMPARISON_PATH = DEFAULT_OUTPUT_DIR / "skillcorner_comparison.csv"
DEFAULT_TEAM_PAX_PATH = DEFAULT_OUTPUT_DIR / "team_pax.csv"
DEFAULT_PLAYER_PAX_PATH = DEFAULT_OUTPUT_DIR / "player_pax.csv"


MODEL_ALIASES = {
    "logistic": "logistic",
    "logistic_regression": "logistic",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
}


@dataclass
class XPassModelResult:
    """Container returned by ``train_xpass_models``."""

    model_name: str
    pipeline: Pipeline
    scored_passes: pd.DataFrame
    metrics: dict
    skillcorner_comparison: pd.DataFrame


def _make_preprocessor(
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)
    binary_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("bin", binary_pipe, binary_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def _canonical_model_name(model_name: str) -> str:
    key = str(model_name).lower().strip()
    if key not in MODEL_ALIASES:
        supported = ", ".join(sorted(set(MODEL_ALIASES.values())))
        raise ValueError(f"Unsupported model={model_name!r}. Supported: {supported}")
    return MODEL_ALIASES[key]


def _make_logistic_model() -> LogisticRegression:
    # xPass is a probability model, so calibration is more important than
    # class-balanced classification accuracy.
    return LogisticRegression(
        max_iter=3000,
        C=0.7,
        solver="lbfgs",
    )


def _make_xgboost_model():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("xgboost is not installed in this environment.") from exc

    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=10,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=42,
        tree_method="hist",
    )


def _make_lightgbm_model():
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("lightgbm is not installed in this environment.") from exc

    return LGBMClassifier(
        n_estimators=550,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        objective="binary",
        n_jobs=4,
        random_state=42,
        verbose=-1,
    )


def make_xpass_pipeline(model_name: str = "logistic") -> Pipeline:
    """Build one xPass model pipeline by name."""
    model_name = _canonical_model_name(model_name)
    numeric, binary, categorical = get_model_feature_columns()
    preprocessor = _make_preprocessor(
        numeric,
        binary,
        categorical,
        scale_numeric=(model_name == "logistic"),
    )
    if model_name == "logistic":
        model = _make_logistic_model()
    elif model_name == "xgboost":
        model = _make_xgboost_model()
    elif model_name == "lightgbm":
        model = _make_lightgbm_model()
    else:  # pragma: no cover
        raise AssertionError(model_name)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def _prepare_training_frame(
    passes: pd.DataFrame,
    *,
    include_offside: bool = False,
    require_target_coordinates: bool = True,
) -> pd.DataFrame:
    modelled = filter_modelled_passes(passes, include_offside=include_offside)
    featured = add_xpass_features(modelled)
    if require_target_coordinates:
        required_xy = ["passer_x", "passer_y", "target_x", "target_y"]
        featured = featured.dropna(subset=required_xy).copy()
    numeric, binary, categorical = get_model_feature_columns()
    for col in binary:
        featured[col] = featured[col].astype(float)
    for col in categorical:
        featured[col] = featured[col].astype("string").fillna("missing")
    featured[TARGET_COLUMN] = featured[TARGET_COLUMN].astype(bool)
    return featured


def _split_by_season(
    featured: pd.DataFrame,
    *,
    train_seasons: list[str],
    test_seasons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = featured["season_name"].astype(str)
    train = featured[seasons.isin([str(s) for s in train_seasons])].copy()
    test = featured[seasons.isin([str(s) for s in test_seasons])].copy()
    if train.empty:
        raise ValueError(f"No training rows matched train_seasons={train_seasons}")
    if test.empty:
        raise ValueError(f"No test rows matched test_seasons={test_seasons}")
    return train, test


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric, binary, categorical = get_model_feature_columns()
    return frame[numeric + binary + categorical].copy()


def predict_xpass(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Predict pass completion probabilities for a featured frame."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return model.predict_proba(_feature_frame(frame))[:, 1]


def _clip_prob(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, 1e-6, 1 - 1e-6)


def evaluate_predictions(name: str, y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> dict:
    """Evaluate probabilistic pass completion predictions."""
    y_bool = y_true.astype(bool).to_numpy()
    pred = _clip_prob(y_pred)
    metrics = {
        "rows": int(len(y_bool)),
        "completed": int(y_bool.sum()),
        "completion_rate": float(y_bool.mean()) if len(y_bool) else None,
        "expected_completions": float(np.sum(pred)),
        "avg_prediction": float(np.mean(pred)) if len(pred) else None,
        "log_loss": float(log_loss(y_bool, pred, labels=[False, True])),
        "brier": float(brier_score_loss(y_bool, pred)),
    }
    metrics["auc"] = (
        float(roc_auc_score(y_bool, pred))
        if len(np.unique(y_bool)) == 2
        else None
    )

    n_bins = min(10, max(1, len(pred) // 1000))
    bins = pd.qcut(pd.Series(pred), q=n_bins, duplicates="drop")
    calibration = (
        pd.DataFrame({"bin": bins, "completed": y_bool, "prediction": pred})
        .groupby("bin", observed=True)
        .agg(
            passes=("completed", "size"),
            completed=("completed", "sum"),
            expected=("prediction", "sum"),
            avg_prediction=("prediction", "mean"),
        )
        .reset_index()
    )
    calibration["completion_rate"] = calibration["completed"] / calibration["passes"]
    metrics["calibration"] = [
        {
            "bin": str(row["bin"]),
            "passes": int(row["passes"]),
            "completed": int(row["completed"]),
            "expected": float(row["expected"]),
            "avg_prediction": float(row["avg_prediction"]),
            "completion_rate": float(row["completion_rate"]),
        }
        for _, row in calibration.iterrows()
    ]
    return {name: metrics}


def compare_with_skillcorner(
    scored: pd.DataFrame,
    *,
    prediction_col: str = "model_xpass",
    split_name: str = "test",
) -> tuple[dict, pd.DataFrame]:
    """Compare a custom xPass prediction with SkillCorner's supplied xPass."""
    required = [TARGET_COLUMN, prediction_col, SKILLCORNER_XPASS_COLUMN]
    missing = [col for col in required if col not in scored.columns]
    if missing:
        raise ValueError(f"Missing required columns for comparison: {missing}")

    comparison = scored[
        scored[prediction_col].notna()
        & scored[SKILLCORNER_XPASS_COLUMN].notna()
    ].copy()
    if comparison.empty:
        return {
            "rows": 0,
            "note": "No rows had both custom and SkillCorner xPass predictions.",
        }, pd.DataFrame()

    y = comparison[TARGET_COLUMN].astype(bool)
    custom = comparison[prediction_col].astype(float)
    skillcorner = comparison[SKILLCORNER_XPASS_COLUMN].astype(float)

    custom_metrics = evaluate_predictions("custom", y, custom)["custom"]
    skillcorner_metrics = evaluate_predictions("skillcorner", y, skillcorner)["skillcorner"]
    diff = custom - skillcorner
    summary = {
        "split": split_name,
        "rows": int(len(comparison)),
        "custom": custom_metrics,
        "skillcorner": skillcorner_metrics,
        "pearson_corr": float(custom.corr(skillcorner, method="pearson")),
        "spearman_corr": float(custom.corr(skillcorner, method="spearman")),
        "mean_custom_minus_skillcorner": float(diff.mean()),
        "mae_custom_vs_skillcorner": float(diff.abs().mean()),
        "rmse_custom_vs_skillcorner": float(np.sqrt(np.mean(diff**2))),
    }

    deciles = pd.qcut(skillcorner, q=min(10, max(1, len(comparison) // 1000)), duplicates="drop")
    by_decile = (
        comparison.assign(skillcorner_xpass_bin=deciles, custom_minus_skillcorner=diff)
        .groupby("skillcorner_xpass_bin", observed=True)
        .agg(
            passes=("event_id", "size"),
            completion_rate=(TARGET_COLUMN, "mean"),
            custom_xpass=(prediction_col, "mean"),
            skillcorner_xpass=(SKILLCORNER_XPASS_COLUMN, "mean"),
            custom_minus_skillcorner=("custom_minus_skillcorner", "mean"),
            pass_distance=("pass_distance_feature", "mean"),
        )
        .reset_index()
    )
    by_decile["skillcorner_xpass_bin"] = by_decile["skillcorner_xpass_bin"].astype(str)
    return summary, by_decile


def add_pax_columns(
    scored: pd.DataFrame,
    *,
    xpass_col: str,
    prefix: str = "custom",
) -> pd.DataFrame:
    """Add Passes Completed Above Expected columns.

    PAx follows the Opta Analyst definition:

    ``PAx = observed pass outcome - expected pass completion probability``.

    At an aggregate level this is equivalent to completed passes minus expected
    completed passes. The per-pass and per-100-pass rates are computed in
    :func:`summarize_pax`.
    """
    if TARGET_COLUMN not in scored.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    if xpass_col not in scored.columns:
        raise ValueError(f"Missing xPass column: {xpass_col}")
    out = scored.copy()
    outcome = out[TARGET_COLUMN].astype(float)
    xpass = pd.to_numeric(out[xpass_col], errors="coerce")
    out[f"{prefix}_xpass"] = xpass
    out[f"{prefix}_pax"] = outcome - xpass
    return out


def summarize_pax(
    scored: pd.DataFrame,
    *,
    group_cols: list[str],
    xpass_col: str,
    prefix: str = "custom",
    min_passes: int = 0,
) -> pd.DataFrame:
    """Aggregate xPass and PAx by team, player, or any other grouping."""
    required = [TARGET_COLUMN, "event_id", xpass_col]
    missing = [col for col in required if col not in scored.columns]
    if missing:
        raise ValueError(f"Missing required columns for PAx summary: {missing}")
    frame = add_pax_columns(scored, xpass_col=xpass_col, prefix=prefix)
    frame = frame[frame[f"{prefix}_xpass"].notna()].copy()
    agg = {
        "passes": ("event_id", "size"),
        "completed": (TARGET_COLUMN, "sum"),
        f"{prefix}_expected_completed": (f"{prefix}_xpass", "sum"),
        f"avg_{prefix}_xpass": (f"{prefix}_xpass", "mean"),
        f"{prefix}_pax": (f"{prefix}_pax", "sum"),
    }
    optional_numeric = {
        "avg_pass_distance": "pass_distance_feature",
        "avg_pass_progression": "pass_progression",
    }
    for out_col, source_col in optional_numeric.items():
        if source_col in frame.columns:
            agg[out_col] = (source_col, "mean")
    if "player_position" in frame.columns and "player_position" not in group_cols:
        agg["player_position"] = (
            "player_position",
            lambda s: s.astype("string").dropna().mode().iloc[0]
            if not s.astype("string").dropna().empty
            else pd.NA,
        )

    summary = (
        frame.groupby(group_cols, dropna=False)
        .agg(**agg)
        .reset_index()
    )
    summary["completion_rate"] = summary["completed"] / summary["passes"]
    summary[f"{prefix}_pax_per_pass"] = summary[f"{prefix}_pax"] / summary["passes"]
    summary[f"{prefix}_pax_per_100"] = summary[f"{prefix}_pax_per_pass"] * 100.0
    if min_passes:
        summary = summary[summary["passes"] >= int(min_passes)].copy()
    return summary.sort_values(f"{prefix}_pax_per_100", ascending=False, ignore_index=True)


def summarize_custom_and_skillcorner_pax(
    scored: pd.DataFrame,
    *,
    group_cols: list[str],
    custom_xpass_col: str,
    min_passes: int = 0,
) -> pd.DataFrame:
    """Return one table containing custom and SkillCorner PAx aggregates."""
    custom = summarize_pax(
        scored,
        group_cols=group_cols,
        xpass_col=custom_xpass_col,
        prefix="custom",
        min_passes=min_passes,
    )
    if SKILLCORNER_XPASS_COLUMN not in scored.columns:
        return custom
    skillcorner = summarize_pax(
        scored,
        group_cols=group_cols,
        xpass_col=SKILLCORNER_XPASS_COLUMN,
        prefix="skillcorner",
        min_passes=min_passes,
    )
    keep_cols = group_cols + [
        "skillcorner_expected_completed",
        "avg_skillcorner_xpass",
        "skillcorner_pax",
        "skillcorner_pax_per_pass",
        "skillcorner_pax_per_100",
    ]
    return custom.merge(skillcorner[keep_cols], on=group_cols, how="left")


def _coefficient_table(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefs = estimator.coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False, ignore_index=True)


def _tree_importance_table(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    else:
        return pd.DataFrame()
    table = pd.DataFrame({"feature": names, "importance": importances})
    return table.sort_values("importance", ascending=False, ignore_index=True)


def feature_importance_table(model: Pipeline, model_name: str) -> pd.DataFrame:
    """Return a compact feature importance table for fitted models."""
    model_name = _canonical_model_name(model_name)
    if model_name == "logistic":
        return _coefficient_table(model)
    return _tree_importance_table(model)


def train_single_xpass_model(
    passes: pd.DataFrame,
    *,
    train_seasons: list[str],
    test_seasons: list[str],
    model_name: str = "logistic",
    include_offside: bool = False,
    require_target_coordinates: bool = True,
) -> XPassModelResult:
    """Train one model and score train/test rows."""
    model_name = _canonical_model_name(model_name)
    featured = _prepare_training_frame(
        passes,
        include_offside=include_offside,
        require_target_coordinates=require_target_coordinates,
    )
    train, test = _split_by_season(
        featured,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
    )

    model = make_xpass_pipeline(model_name)
    model.fit(_feature_frame(train), train[TARGET_COLUMN])

    train_pred = predict_xpass(model, train)
    test_pred = predict_xpass(model, test)
    pred_col = f"xpass_{model_name}"

    train_scored = train.copy()
    train_scored["split"] = "train"
    train_scored[pred_col] = train_pred
    train_scored["model_xpass"] = train_pred

    test_scored = test.copy()
    test_scored["split"] = "test"
    test_scored[pred_col] = test_pred
    test_scored["model_xpass"] = test_pred

    scored = pd.concat([train_scored, test_scored], ignore_index=True, sort=False)
    train_metrics = evaluate_predictions("train", train[TARGET_COLUMN], train_pred)["train"]
    test_metrics = evaluate_predictions("test", test[TARGET_COLUMN], test_pred)["test"]
    sc_summary, sc_deciles = compare_with_skillcorner(
        test_scored,
        prediction_col=pred_col,
        split_name="test",
    )
    metrics = {
        "model_name": model_name,
        "train_seasons": [str(s) for s in train_seasons],
        "test_seasons": [str(s) for s in test_seasons],
        "train": train_metrics,
        "test": test_metrics,
        "skillcorner_comparison": sc_summary,
    }
    return XPassModelResult(
        model_name=model_name,
        pipeline=model,
        scored_passes=scored,
        metrics=metrics,
        skillcorner_comparison=sc_deciles,
    )


def train_xpass_models(
    passes: pd.DataFrame,
    *,
    train_seasons: list[str],
    test_seasons: list[str],
    models: list[str] | tuple[str, ...] = ("logistic", "xgboost"),
    include_offside: bool = False,
    require_target_coordinates: bool = True,
) -> tuple[dict[str, XPassModelResult], pd.DataFrame]:
    """Train multiple xPass models and return results plus a comparison table."""
    results: dict[str, XPassModelResult] = {}
    rows = []
    for model_name in models:
        canonical = _canonical_model_name(model_name)
        result = train_single_xpass_model(
            passes,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
            model_name=canonical,
            include_offside=include_offside,
            require_target_coordinates=require_target_coordinates,
        )
        results[canonical] = result
        rows.append(
            {
                "model_name": canonical,
                "train_rows": result.metrics["train"]["rows"],
                "test_rows": result.metrics["test"]["rows"],
                "test_auc": result.metrics["test"]["auc"],
                "test_log_loss": result.metrics["test"]["log_loss"],
                "test_brier": result.metrics["test"]["brier"],
                "test_completion_rate": result.metrics["test"]["completion_rate"],
                "test_avg_prediction": result.metrics["test"]["avg_prediction"],
                "skillcorner_auc": result.metrics["skillcorner_comparison"]["skillcorner"]["auc"],
                "skillcorner_log_loss": result.metrics["skillcorner_comparison"]["skillcorner"]["log_loss"],
                "skillcorner_brier": result.metrics["skillcorner_comparison"]["skillcorner"]["brier"],
                "pearson_corr_with_skillcorner": result.metrics["skillcorner_comparison"]["pearson_corr"],
                "mae_vs_skillcorner": result.metrics["skillcorner_comparison"]["mae_custom_vs_skillcorner"],
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["test_log_loss", "test_brier"],
        ascending=[True, True],
        ignore_index=True,
    )
    return results, comparison


def _json_ready_metrics(results: dict[str, XPassModelResult]) -> dict:
    return {name: result.metrics for name, result in results.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare SkillCorner xPass models.")
    parser.add_argument("--passes", type=Path, default=DEFAULT_PASSES_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-seasons", nargs="+", default=["2023", "2024"])
    parser.add_argument("--test-seasons", nargs="+", default=["2025"])
    parser.add_argument("--models", nargs="+", default=["logistic", "xgboost"])
    parser.add_argument("--include-offside", action="store_true")
    parser.add_argument("--allow-missing-target-coordinates", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    passes = pd.read_parquet(args.passes)
    results, comparison = train_xpass_models(
        passes,
        train_seasons=[str(s) for s in args.train_seasons],
        test_seasons=[str(s) for s in args.test_seasons],
        models=args.models,
        include_offside=args.include_offside,
        require_target_coordinates=not args.allow_missing_target_coordinates,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.out_dir / DEFAULT_COMPARISON_PATH.name, index=False)
    (args.out_dir / DEFAULT_METRICS_PATH.name).write_text(
        json.dumps(_json_ready_metrics(results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    best_name = comparison.iloc[0]["model_name"]
    best = results[best_name]
    joblib.dump(best.pipeline, args.out_dir / DEFAULT_MODEL_PATH.name)
    best.scored_passes.to_parquet(args.out_dir / DEFAULT_SCORED_PASSES_PATH.name, index=False)
    best.skillcorner_comparison.to_csv(
        args.out_dir / DEFAULT_SKILLCORNER_COMPARISON_PATH.name,
        index=False,
    )
    best_pred_col = f"xpass_{best_name}"
    test_scored = best.scored_passes[best.scored_passes["split"].eq("test")].copy()
    comparable_test = test_scored[test_scored[SKILLCORNER_XPASS_COLUMN].notna()].copy()
    if not comparable_test.empty:
        summarize_custom_and_skillcorner_pax(
            comparable_test,
            group_cols=["season_name", "team_shortname"],
            custom_xpass_col=best_pred_col,
        ).to_csv(args.out_dir / DEFAULT_TEAM_PAX_PATH.name, index=False)
        summarize_custom_and_skillcorner_pax(
            comparable_test,
            group_cols=["season_name", "team_shortname", "player_name"],
            custom_xpass_col=best_pred_col,
            min_passes=300,
        ).to_csv(args.out_dir / DEFAULT_PLAYER_PAX_PATH.name, index=False)

    for model_name, result in results.items():
        joblib.dump(result.pipeline, args.out_dir / f"skillcorner_xpass_{model_name}.joblib")
        result.scored_passes.to_parquet(args.out_dir / f"scored_passes_{model_name}.parquet", index=False)
        importance = feature_importance_table(result.pipeline, model_name)
        if not importance.empty:
            importance.head(100).to_csv(args.out_dir / f"feature_importance_{model_name}.csv", index=False)

    print(comparison.to_string(index=False))
    print(f"[xpass] best model={best_name}; wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
