from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ProjectConfig


@dataclass
class TrainingResult:
    model: ClassifierMixin
    metrics: Dict
    feature_columns: List[str]


def build_model(model_cfg: Dict, class_weight: str | Dict | None = None) -> ClassifierMixin:
    """Instantiate the classifier specified in the config."""
    model_type = model_cfg.get("type", "logistic_regression").lower()
    params = model_cfg.get("params", {})

    if model_type == "logistic_regression":
        estimator = LogisticRegression(class_weight=class_weight, **params)
        return Pipeline(
            steps=[
                ("preprocess", ColumnTransformer(remainder="drop", transformers=[])),
                ("classifier", estimator),
            ]
        )

    if model_type == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(**params)
        return Pipeline(
            steps=[
                ("preprocess", ColumnTransformer(remainder="drop", transformers=[])),
                ("classifier", estimator),
            ]
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def prepare_feature_matrix(
    features: pd.DataFrame,
    cfg: ProjectConfig,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """Select feature columns and build design matrix."""
    target_column = cfg.training["target_column"]
    include_columns = cfg.features.get("include_columns")

    base_exclusions = {"match_date", "player_a", "player_b", "winner", "player_a_wins", "player_b_wins"}
    candidate_columns = [col for col in features.columns if col not in base_exclusions]

    if include_columns:
        missing = [col for col in include_columns if col not in features.columns]
        if missing:
            raise ValueError(f"Requested feature columns not found in dataset: {missing}")
        feature_columns = include_columns
    else:
        feature_columns = candidate_columns

    X = features.loc[:, feature_columns].copy()
    y = features[target_column].astype(int)

    numeric_columns = [col for col in feature_columns if np.issubdtype(X[col].dtype, np.number)]
    categorical_columns = [col for col in feature_columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return X, y, feature_columns, preprocessor


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: ProjectConfig,
    preprocessor: ColumnTransformer,
) -> TrainingResult:
    """Train and evaluate the chosen model with time-based cross-validation."""
    training_cfg = cfg.training
    validation_cfg = training_cfg.get("validation_strategy", {})
    model_cfg = cfg.model
    class_weight = training_cfg.get("class_weight")

    pipeline = build_model(model_cfg, class_weight=class_weight)
    pipeline.set_params(preprocess=preprocessor)

    n_splits = int(validation_cfg.get("n_splits", 5))
    test_size = validation_cfg.get("test_size")
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    feature_array = X.values
    metrics_folds: List[Dict] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(feature_array, y), start=1):
        estimator = clone(pipeline)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        probas = estimator.predict_proba(X.iloc[val_idx])[:, 1]
        preds = (probas >= 0.5).astype(int)

        fold_metrics = {
            "fold": fold,
            "accuracy": accuracy_score(y.iloc[val_idx], preds),
            "roc_auc": roc_auc_score(y.iloc[val_idx], probas),
            "log_loss": log_loss(y.iloc[val_idx], probas),
            "brier_score": brier_score_loss(y.iloc[val_idx], probas),
        }
        metrics_folds.append(fold_metrics)

    metrics_summary = _summarize_metrics(metrics_folds)

    pipeline.fit(X, y)
    pipeline.feature_columns = list(X.columns)
    return TrainingResult(model=pipeline, metrics={"folds": metrics_folds, "summary": metrics_summary}, feature_columns=list(X.columns))


def _summarize_metrics(metrics_folds: List[Dict]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    keys = [key for key in metrics_folds[0].keys() if key != "fold"]
    for key in keys:
        values = [fold[key] for fold in metrics_folds]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary
