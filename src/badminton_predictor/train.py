from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib

from .config import ProjectConfig, load_config
from .data import load_matches, load_player_attributes
from .features import engineer_features
from .models import TrainingResult, prepare_feature_matrix, train_model


def train_pipeline(config_path: str | Path) -> Dict:
    """Entry point for the end-to-end training workflow."""
    cfg = load_config(config_path)

    matches = load_matches(cfg)
    player_attributes = load_player_attributes(cfg)
    features = engineer_features(matches, cfg, player_attributes)

    _ensure_parent(cfg.output["processed_matches"])
    _ensure_parent(cfg.output["features_matrix"])
    _ensure_parent(cfg.output["model_artifact"])
    _ensure_parent(cfg.output.get("metrics_report", "models/metrics.json"))

    matches.to_parquet(cfg.output["processed_matches"])
    features.to_parquet(cfg.output["features_matrix"])

    X, y, feature_columns, preprocessor = prepare_feature_matrix(features, cfg)
    result = train_model(X, y, cfg, preprocessor)

    joblib.dump(result.model, cfg.output["model_artifact"])

    metrics_report = {
        "metrics": result.metrics,
        "feature_columns": feature_columns,
        "config": cfg.raw,
    }
    if cfg.output.get("metrics_report"):
        metrics_path = Path(cfg.output["metrics_report"])
        metrics_path.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")

    return metrics_report


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train badminton match outcome model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    report = train_pipeline(args.config)
    print(json.dumps(report["metrics"]["summary"], indent=2))


if __name__ == "__main__":
    main()
