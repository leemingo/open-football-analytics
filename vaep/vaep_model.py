"""The two-head VAEP probability model: P(scores) and P(concedes).

Both heads share the same game-state feature matrix. XGBoost is the default; the
settings relax automatically on small samples (the open dataset is only a handful
of matches), mirroring the ``make_xgb`` helper in the xg tutorial. A logistic
baseline is available for a quick, interpretable comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

MODEL_ALIASES = {"xgboost": "xgboost", "xgb": "xgboost", "logistic": "logistic", "logistic_regression": "logistic"}


def _canonical(model_name: str) -> str:
    key = str(model_name).lower().strip()
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model={model_name!r}. Supported: xgboost, logistic")
    return MODEL_ALIASES[key]


def make_estimator(model_name: str, n_train: int):
    """One probability head. Relaxes XGBoost on small data so trees can still split."""
    model_name = _canonical(model_name)
    if model_name == "logistic":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([("scale", StandardScaler(with_mean=False)),
                         ("model", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"))])
    from xgboost import XGBClassifier
    # Goals are very rare (~1% base rate) and the open dataset is tiny, so on small
    # samples we regularize hard (shallow trees, large leaves) to avoid memorizing.
    small = n_train < 50_000
    return XGBClassifier(
        n_estimators=200 if small else 400,
        max_depth=3 if small else 5,
        learning_rate=0.05,
        subsample=0.8 if small else 0.85,
        colsample_bytree=0.8 if small else 0.85,
        min_child_weight=10 if small else 10,
        reg_lambda=5.0 if small else 2.0,
        objective="binary:logistic", eval_metric="logloss",
        n_jobs=4, random_state=42, tree_method="hist",
    )


def _evaluate_head(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y = np.asarray(y_true).astype(int)
    two = len(np.unique(y)) == 2
    return {name: {
        "rows": int(len(y)), "positives": int(y.sum()),
        "base_rate": float(y.mean()) if len(y) else None,
        "pred_mean": float(np.mean(y_pred)) if len(y_pred) else None,
        "log_loss": float(log_loss(y, y_pred, labels=[0, 1])) if two else None,
        "brier": float(brier_score_loss(y, y_pred)) if len(y) else None,
        "auc": float(roc_auc_score(y, y_pred)) if two else None,
    }}


class VAEPModel:
    """Wraps the P(scores) and P(concedes) estimators."""

    def __init__(self, model_name: str = "xgboost", n_train: int = 0):
        self.model_name = _canonical(model_name)
        self.scores_model = make_estimator(self.model_name, n_train)
        self.concedes_model = make_estimator(self.model_name, n_train)

    def fit(self, X: pd.DataFrame, y_scores: np.ndarray, y_concedes: np.ndarray) -> "VAEPModel":
        self.scores_model.fit(X, np.asarray(y_scores).astype(int))
        self.concedes_model.fit(X, np.asarray(y_concedes).astype(int))
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (self.scores_model.predict_proba(X)[:, 1],
                self.concedes_model.predict_proba(X)[:, 1])

    def evaluate(self, X: pd.DataFrame, y_scores: np.ndarray, y_concedes: np.ndarray, *, split: str) -> dict:
        p_scores, p_concedes = self.predict(X)
        out: dict = {}
        out.update(_evaluate_head(f"{split}_scores", y_scores, p_scores))
        out.update(_evaluate_head(f"{split}_concedes", y_concedes, p_concedes))
        return out
