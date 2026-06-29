"""Train and compare shot-level xG models for SkillCorner Dynamic Events."""
from __future__ import annotations

import argparse
import json
import warnings
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

from xg.skillcorner_shots import DEFAULT_OUTPUT_DIR, DEFAULT_SHOTS_PATH
from xg.xg_features import (
    TARGET_COLUMN,
    add_xg_features,
    get_model_feature_columns,
)


DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "skillcorner_xg_best.joblib"
DEFAULT_SCORED_SHOTS_PATH = DEFAULT_OUTPUT_DIR / "scored_shots.parquet"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "metrics.json"
DEFAULT_COMPARISON_PATH = DEFAULT_OUTPUT_DIR / "model_comparison.csv"

MODEL_ALIASES = {
    "logistic": "logistic",
    "logistic_regression": "logistic",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
}


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
    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
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
    # We intentionally do not use class weights: for xG, probability calibration
    # matters more than balanced classification accuracy.
    return LogisticRegression(
        max_iter=3000,
        C=0.5,
        solver="lbfgs",
    )


def _make_xgboost_model():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("xgboost is not installed in this environment.") from exc

    return XGBClassifier(
        n_estimators=450,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=8,
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
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=24,
        max_depth=5,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        objective="binary",
        n_jobs=4,
        random_state=42,
        verbose=-1,
    )


def make_xg_pipeline(model_name: str = "logistic") -> Pipeline:
    """Build one xG model pipeline by name."""
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


def make_logistic_xg_pipeline() -> Pipeline:
    """Backward-compatible baseline constructor."""
    return make_xg_pipeline("logistic")


def _prepare_training_frame(shots: pd.DataFrame) -> pd.DataFrame:
    featured = add_xg_features(shots)
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


def _predict_xg(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    numeric, binary, categorical = get_model_feature_columns()
    X = frame[numeric + binary + categorical].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return model.predict_proba(X)[:, 1]


def _evaluate_split(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    y_bool = y_true.astype(bool).to_numpy()
    metrics = {
        "rows": int(len(y_bool)),
        "goals": int(y_bool.sum()),
        "goal_rate": float(y_bool.mean()) if len(y_bool) else None,
        "xg_sum": float(np.sum(y_pred)),
        "xg_per_shot": float(np.mean(y_pred)) if len(y_pred) else None,
        "log_loss": float(log_loss(y_bool, y_pred, labels=[False, True])),
        "brier": float(brier_score_loss(y_bool, y_pred)),
    }
    metrics["auc"] = (
        float(roc_auc_score(y_bool, y_pred))
        if len(np.unique(y_bool)) == 2
        else None
    )

    bins = pd.qcut(pd.Series(y_pred), q=min(10, max(1, len(y_pred) // 50)), duplicates="drop")
    calibration = (
        pd.DataFrame({"bin": bins, "goal": y_bool, "xg": y_pred})
        .groupby("bin", observed=True)
        .agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"), avg_xg=("xg", "mean"))
        .reset_index()
    )
    metrics["calibration"] = [
        {
            "bin": str(row["bin"]),
            "shots": int(row["shots"]),
            "goals": int(row["goals"]),
            "xg": float(row["xg"]),
            "avg_xg": float(row["avg_xg"]),
            "goal_rate": float(row["goals"] / row["shots"]) if row["shots"] else None,
        }
        for _, row in calibration.iterrows()
    ]
    return {name: metrics}


def _coefficient_table(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefs = estimator.coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False, ignore_index=True)


def train_xg_model(
    shots: pd.DataFrame,
    *,
    train_seasons: list[str],
    test_seasons: list[str],
) -> tuple[Pipeline, pd.DataFrame, dict, pd.DataFrame]:
    """Train the logistic baseline for backward compatibility."""
    return train_single_xg_model(
        shots,
        model_name="logistic",
        train_seasons=train_seasons,
        test_seasons=test_seasons,
    )


def train_single_xg_model(
    shots: pd.DataFrame,
    *,
    model_name: str,
    train_seasons: list[str],
    test_seasons: list[str],
) -> tuple[Pipeline, pd.DataFrame, dict, pd.DataFrame]:
    model_name = _canonical_model_name(model_name)
    featured = _prepare_training_frame(shots)
    train, test = _split_by_season(
        featured,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
    )
    numeric, binary, categorical = get_model_feature_columns()
    feature_cols = numeric + binary + categorical

    model = make_xg_pipeline(model_name)
    y_train = train[TARGET_COLUMN].astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(train[feature_cols], y_train)

    train_pred = _predict_xg(model, train)
    test_pred = _predict_xg(model, test)

    metrics: dict = {
        "train_seasons": [str(s) for s in train_seasons],
        "test_seasons": [str(s) for s in test_seasons],
        "model": model_name,
        "target": TARGET_COLUMN,
    }
    metrics.update(_evaluate_split("train", train[TARGET_COLUMN], train_pred))
    metrics.update(_evaluate_split("test", test[TARGET_COLUMN], test_pred))

    scored = featured.copy()
    scored["xg"] = _predict_xg(model, scored)
    scored["split"] = "unused"
    scored.loc[train.index, "split"] = "train"
    scored.loc[test.index, "split"] = "test"

    coefficients = (
        _coefficient_table(model)
        if model_name == "logistic"
        else pd.DataFrame()
    )
    return model, scored, metrics, coefficients


def _comparison_row(metrics: dict) -> dict:
    test = metrics["test"]
    train = metrics["train"]
    return {
        "model": metrics["model"],
        "train_rows": train["rows"],
        "train_goals": train["goals"],
        "train_xg": train["xg_sum"],
        "train_log_loss": train["log_loss"],
        "train_brier": train["brier"],
        "train_auc": train["auc"],
        "test_rows": test["rows"],
        "test_goals": test["goals"],
        "test_xg": test["xg_sum"],
        "test_log_loss": test["log_loss"],
        "test_brier": test["brier"],
        "test_auc": test["auc"],
        "test_xg_minus_goals": test["xg_sum"] - test["goals"],
    }


def train_and_compare_xg_models(
    shots: pd.DataFrame,
    *,
    model_names: list[str],
    train_seasons: list[str],
    test_seasons: list[str],
) -> tuple[dict[str, Pipeline], dict[str, pd.DataFrame], dict, pd.DataFrame, str]:
    """Train several xG models and return comparison artefacts."""
    models: dict[str, Pipeline] = {}
    scored_frames: dict[str, pd.DataFrame] = {}
    metrics_by_model: dict = {
        "train_seasons": [str(s) for s in train_seasons],
        "test_seasons": [str(s) for s in test_seasons],
        "target": TARGET_COLUMN,
        "models": {},
    }
    comparison_rows = []

    for requested_name in model_names:
        model_name = _canonical_model_name(requested_name)
        print(f"[xg] training {model_name}")
        model, scored, metrics, _ = train_single_xg_model(
            shots,
            model_name=model_name,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
        )
        models[model_name] = model
        scored_frames[model_name] = scored
        metrics_by_model["models"][model_name] = metrics
        comparison_rows.append(_comparison_row(metrics))

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["test_log_loss", "test_brier"],
        ascending=[True, True],
        ignore_index=True,
    )
    best_model_name = str(comparison.iloc[0]["model"])
    metrics_by_model["best_model"] = best_model_name
    metrics_by_model["comparison"] = comparison.to_dict(orient="records")
    return models, scored_frames, metrics_by_model, comparison, best_model_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare SkillCorner xG models.")
    parser.add_argument("--shots", type=Path, default=DEFAULT_SHOTS_PATH)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scored-out", type=Path, default=DEFAULT_SCORED_SHOTS_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--comparison-out", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--coefficients-out", type=Path, default=DEFAULT_OUTPUT_DIR / "model_coefficients.csv")
    parser.add_argument("--train-seasons", nargs="+", default=["2023", "2024"])
    parser.add_argument("--test-seasons", nargs="+", default=["2025"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "xgboost", "lightgbm"],
        help="Models to train: logistic, xgboost, lightgbm",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    shots = pd.read_parquet(args.shots)
    models, scored_frames, metrics, comparison, best_model_name = train_and_compare_xg_models(
        shots,
        model_names=args.models,
        train_seasons=args.train_seasons,
        test_seasons=args.test_seasons,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.scored_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_out.parent.mkdir(parents=True, exist_ok=True)
    args.coefficients_out.parent.mkdir(parents=True, exist_ok=True)

    best_model = models[best_model_name]
    best_scored = scored_frames[best_model_name]
    joblib.dump(best_model, args.model_out)
    best_scored.to_parquet(args.scored_out, index=False)
    args.metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparison.to_csv(args.comparison_out, index=False)

    for model_name, model in models.items():
        model_path = args.model_out.with_name(f"skillcorner_xg_{model_name}.joblib")
        scored_path = args.scored_out.with_name(f"scored_shots_{model_name}.parquet")
        joblib.dump(model, model_path)
        scored_frames[model_name].to_parquet(scored_path, index=False)

    if "logistic" in models:
        coefficients = _coefficient_table(models["logistic"])
        coefficients.to_csv(args.coefficients_out, index=False)

    print(f"[xg] model: {args.model_out}")
    print(f"[xg] best model: {best_model_name}")
    print(f"[xg] scored shots: {args.scored_out} ({len(best_scored):,} rows)")
    print(f"[xg] metrics: {args.metrics_out}")
    print(f"[xg] comparison: {args.comparison_out}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
