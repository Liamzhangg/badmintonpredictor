from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import ProjectConfig


def load_matches(cfg: ProjectConfig) -> pd.DataFrame:
    """Load and concatenate match datasets defined in the config."""
    frames: List[pd.DataFrame] = []
    for dataset in cfg.data.get("matches", []):
        df = _read_table(dataset)
        df = _standardize_match_columns(df, dataset)
        frames.append(df)

    if not frames:
        raise ValueError("No match datasets were loaded")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("match_date").reset_index(drop=True)
    return combined


def load_player_attributes(cfg: ProjectConfig) -> pd.DataFrame:
    """Load player-level metadata such as height, country, or handedness."""
    config = cfg.data.get("player_attributes")
    if not config:
        return pd.DataFrame()

    df = _read_table(config)
    id_column = config.get("id_column", "player_name")
    all_columns = [id_column] + config.get("columns", [])
    missing = [column for column in all_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Player attributes file missing columns: {', '.join(missing)}")

    df = df.loc[:, all_columns].drop_duplicates(id_column)
    return df


def _read_table(dataset: Dict) -> pd.DataFrame:
    """Read a CSV or Excel dataset based on the provided metadata."""
    path = Path(dataset["path"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    file_format = dataset.get("format", path.suffix.lstrip(".").lower())

    if file_format == "csv":
        df = pd.read_csv(path)
    elif file_format in {"xlsx", "xls"}:
        df = pd.read_excel(path, sheet_name=dataset.get("sheet_name"))
    elif file_format == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported dataset format '{file_format}' for {path}")

    return df


def _standardize_match_columns(df: pd.DataFrame, dataset: Dict) -> pd.DataFrame:
    """Rename key columns and coerce data types for downstream processing."""
    df = df.copy()

    rename_map = {
        dataset["player_a_column"]: "player_a",
        dataset["player_b_column"]: "player_b",
        dataset["winner_column"]: "winner",
        dataset["date_column"]: "match_date",
    }
    df = df.rename(columns=rename_map)

    if "match_date" not in df.columns:
        raise KeyError("match_date column missing after rename")

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    if df["match_date"].isna().any():
        raise ValueError("Some match_date values could not be parsed; please clean the source file.")

    df["player_a"] = df["player_a"].str.strip()
    df["player_b"] = df["player_b"].str.strip()
    df["winner"] = df["winner"].str.strip()

    # Add optional metadata columns
    event_columns: Iterable[str] = dataset.get("event_columns", [])
    for column in event_columns:
        if column not in df.columns:
            raise ValueError(f"Expected column '{column}' not found in dataset {dataset['path']}")

    # Keep only relevant columns plus optional metadata
    keep_columns = ["match_date", "player_a", "player_b", "winner"]
    keep_columns.extend(event_columns)
    if dataset.get("score_column") and dataset["score_column"] in df.columns:
        df = df.rename(columns={dataset["score_column"]: "score"})
        keep_columns.append("score")

    df = df[keep_columns]
    df["player_a_wins"] = (df["winner"] == df["player_a"]).astype(float)
    df["player_b_wins"] = (df["winner"] == df["player_b"]).astype(float)

    missing_winner = df["winner"].isna() | (df["winner"] == "")
    df.loc[missing_winner, ["player_a_wins", "player_b_wins"]] = np.nan

    return df
