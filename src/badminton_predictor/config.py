from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ProjectConfig:
    """Simple container for the entire YAML configuration."""

    raw: Dict[str, Any]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def features(self) -> Dict[str, Any]:
        return self.raw.get("features", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def output(self) -> Dict[str, Any]:
        return self.raw.get("output", {})


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load a YAML config file into a ProjectConfig instance."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    validate_config(config)
    return ProjectConfig(raw=config)


def validate_config(config: Dict[str, Any]) -> None:
    """Perform minimal validation to ensure critical settings exist."""
    required_sections = ["data", "features", "model", "training", "output"]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Config missing sections: {', '.join(missing_sections)}")

    matches = config["data"].get("matches", [])
    if not matches:
        raise ValueError("At least one dataset must be defined under data.matches.")

    for dataset in matches:
        for key in ("path", "player_a_column", "player_b_column", "winner_column", "date_column"):
            if key not in dataset:
                raise ValueError(f"data.matches entry missing '{key}' field: {dataset}")

    target = config["training"].get("target_column")
    if not target:
        raise ValueError("training.target_column must be set.")

    outputs = config["output"]
    for key in ("processed_matches", "features_matrix", "model_artifact"):
        if key not in outputs:
            raise ValueError(f"output.{key} must be defined.")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dictionaries."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
