from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import ProjectConfig


@dataclass
class PlayerMatchRecord:
    date: pd.Timestamp
    win: int


def engineer_features(
    matches: pd.DataFrame,
    cfg: ProjectConfig,
    player_attributes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate player-centric features for each match row."""
    matches = matches.sort_values("match_date").reset_index(drop=True)
    windows = _parse_windows(cfg.features.get("rolling_windows_days"))
    elo_cfg = cfg.features.get("elo", {})
    k_factor = elo_cfg.get("k_factor", 32)
    floor_rating = elo_cfg.get("floor_rating", 1500)
    decay = elo_cfg.get("decay", 1.0)

    attr_id = cfg.data.get("player_attributes", {}).get("id_column", "player_name")
    attr_frame = (
        player_attributes.set_index(attr_id) if player_attributes is not None and not player_attributes.empty else None
    )
    if attr_frame is not None and "birth_date" in attr_frame.columns:
        attr_frame["birth_date"] = pd.to_datetime(attr_frame["birth_date"], errors="coerce")

    elo_ratings: Dict[str, float] = defaultdict(lambda: float(floor_rating))
    last_played: Dict[str, pd.Timestamp] = defaultdict(lambda: pd.NaT)  # Tracks recency
    recent_history: Dict[str, Deque[PlayerMatchRecord]] = defaultdict(deque)
    head_to_head: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0})

    feature_rows: List[Dict] = []
    metadata_columns = [col for col in matches.columns if col not in {"match_date", "player_a", "player_b", "winner", "player_a_wins", "player_b_wins"}]

    for row in matches.itertuples(index=False):
        player_a = row.player_a
        player_b = row.player_b
        match_date = row.match_date

        features = {
            "match_date": match_date,
            "player_a": player_a,
            "player_b": player_b,
            "player_a_elo": elo_ratings[player_a],
            "player_b_elo": elo_ratings[player_b],
            "days_since_player_a_last_match": _days_since(last_played.get(player_a), match_date),
            "days_since_player_b_last_match": _days_since(last_played.get(player_b), match_date),
            "head_to_head_wins_a": head_to_head[(player_a, player_b)]["wins"],
            "head_to_head_wins_b": head_to_head[(player_b, player_a)]["wins"],
        }
        features["head_to_head_win_pct_a"] = _win_pct(head_to_head[(player_a, player_b)])
        features["head_to_head_win_pct_b"] = _win_pct(head_to_head[(player_b, player_a)])

        # Rolling performance windows
        for window in windows:
            player_a_stats = _rolling_stats(recent_history[player_a], match_date, window)
            player_b_stats = _rolling_stats(recent_history[player_b], match_date, window)
            features[f"player_a_recent_win_rate_{window}d"] = player_a_stats["win_rate"]
            features[f"player_b_recent_win_rate_{window}d"] = player_b_stats["win_rate"]
            features[f"player_a_matches_played_{window}d"] = player_a_stats["matches_played"]
            features[f"player_b_matches_played_{window}d"] = player_b_stats["matches_played"]

        # Player attributes (age, height, etc.)
        features.update(_extract_attributes(attr_frame, player_a, match_date, prefix="player_a"))
        features.update(_extract_attributes(attr_frame, player_b, match_date, prefix="player_b"))

        # Add raw metadata columns (e.g., tournament, round)
        for column in metadata_columns:
            features[column] = getattr(row, column)

        features["player_a_wins"] = row.player_a_wins
        features["player_b_wins"] = row.player_b_wins
        feature_rows.append(features)

        # Update trackers after using existing stats
        elo_a = elo_ratings[player_a]
        elo_b = elo_ratings[player_b]
        expected_a = _elo_expected(elo_a, elo_b)
        expected_b = _elo_expected(elo_b, elo_a)
        actual_a = row.player_a_wins if not pd.isna(row.player_a_wins) else None
        actual_b = row.player_b_wins if not pd.isna(row.player_b_wins) else None

        if actual_a is not None and actual_b is not None:
            elo_ratings[player_a] = _elo_update(elo_a, actual_a, expected_a, k_factor, floor_rating, decay)
            elo_ratings[player_b] = _elo_update(elo_b, actual_b, expected_b, k_factor, floor_rating, decay)

            head_to_head[(player_a, player_b)]["wins"] += int(actual_a)
            head_to_head[(player_a, player_b)]["losses"] += int(actual_b)
            head_to_head[(player_b, player_a)]["wins"] += int(actual_b)
            head_to_head[(player_b, player_a)]["losses"] += int(actual_a)

            recent_history[player_a].append(PlayerMatchRecord(date=match_date, win=int(actual_a)))
            recent_history[player_b].append(PlayerMatchRecord(date=match_date, win=int(actual_b)))

            _prune_history(recent_history[player_a], match_date, max(windows) if windows else 0)
            _prune_history(recent_history[player_b], match_date, max(windows) if windows else 0)

            last_played[player_a] = match_date
            last_played[player_b] = match_date

    return pd.DataFrame(feature_rows)


def _parse_windows(window_config: Iterable[int] | None) -> List[int]:
    if not window_config:
        return [30, 90, 180]
    windows = sorted({int(window) for window in window_config})
    return [window for window in windows if window > 0]


def _days_since(last_date: pd.Timestamp | None, current_date: pd.Timestamp) -> float:
    if not isinstance(last_date, pd.Timestamp) or pd.isna(last_date):
        return np.nan
    return (current_date - last_date).days


def _rolling_stats(history: Deque[PlayerMatchRecord], current_date: pd.Timestamp, window: int) -> Dict[str, float]:
    if window <= 0 or not history:
        return {"win_rate": np.nan, "matches_played": 0}

    cutoff = current_date - timedelta(days=window)
    wins = 0
    matches = 0
    for record in history:
        if record.date <= cutoff:
            continue
        wins += record.win
        matches += 1

    if matches == 0:
        return {"win_rate": np.nan, "matches_played": 0}

    return {"win_rate": wins / matches, "matches_played": matches}


def _extract_attributes(
    attr_frame: pd.DataFrame | None,
    player: str,
    match_date: pd.Timestamp,
    prefix: str,
) -> Dict[str, float]:
    if attr_frame is None or player not in attr_frame.index:
        return {}

    row = attr_frame.loc[player]
    features: Dict[str, float] = {}

    if "birth_date" in row and not pd.isna(row["birth_date"]):
        features[f"{prefix}_age"] = _calculate_age(row["birth_date"], match_date)

    for column in row.index:
        if column == "birth_date":
            continue
        value = row[column]
        features[f"{prefix}_{column}"] = value

    return features


def _calculate_age(birth_date: pd.Timestamp, match_date: pd.Timestamp) -> float:
    years = (match_date - birth_date).days / 365.25
    return round(years, 2)


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _elo_update(
    rating: float,
    actual: float,
    expected: float,
    k_factor: float,
    floor_rating: float,
    decay: float,
) -> float:
    rating = max(rating * decay, floor_rating)
    return rating + k_factor * (actual - expected)


def _prune_history(history: Deque[PlayerMatchRecord], current_date: pd.Timestamp, max_window: int) -> None:
    if max_window <= 0:
        history.clear()
        return

    cutoff = current_date - timedelta(days=max_window)
    while history and history[0].date <= cutoff:
        history.popleft()


def _win_pct(record: Dict[str, int]) -> float:
    total = record["wins"] + record["losses"]
    if total == 0:
        return 0.0
    return record["wins"] / total
