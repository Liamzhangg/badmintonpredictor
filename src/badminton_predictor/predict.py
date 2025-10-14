from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from .config import load_config
from .data import load_matches, load_player_attributes
from .features import engineer_features


def predict_matchup(
    config_path: str | Path,
    player_a: str,
    player_b: str,
    match_date: Optional[str] = None,
    tournament: Optional[str] = None,
    round_name: Optional[str] = None,
) -> Dict:
    """Generate a win probability estimate for an upcoming matchup."""
    cfg = load_config(config_path)
    matches = load_matches(cfg)
    player_attributes = load_player_attributes(cfg)

    match_ts = pd.Timestamp(match_date) if match_date else pd.Timestamp.today()
    placeholder = _build_placeholder_row(matches, player_a, player_b, match_ts, tournament, round_name)
    matches_with_future = pd.concat([matches, placeholder], ignore_index=True)

    features = engineer_features(matches_with_future, cfg, player_attributes)
    pending_mask = features["player_a_wins"].isna()
    if not pending_mask.any():
        raise ValueError("Unable to generate features for the requested matchup. Check player names and input data.")

    sample = features.loc[pending_mask].iloc[-1:]

    model_artifact = cfg.output["model_artifact"]
    model = joblib.load(model_artifact)
    feature_columns = getattr(model, "feature_columns", None)
    if feature_columns is None:
        metrics_path = cfg.output.get("metrics_report")
        if metrics_path and Path(metrics_path).exists():
            metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
            feature_columns = metrics.get("feature_columns")
    if not feature_columns:
        raise ValueError("Feature column list not found. Retrain the model to regenerate metadata.")

    X_sample = sample.loc[:, feature_columns]
    probability = float(model.predict_proba(X_sample)[0, 1])

    return {
        "player_a": player_a,
        "player_b": player_b,
        "match_date": match_ts.isoformat(),
        "probability_player_a_wins": probability,
        "feature_snapshot": sample.loc[:, feature_columns].to_dict(orient="records")[0],
    }


def _build_placeholder_row(
    matches: pd.DataFrame,
    player_a: str,
    player_b: str,
    match_date: pd.Timestamp,
    tournament: Optional[str],
    round_name: Optional[str],
) -> pd.DataFrame:
    """Create a dataframe row aligning with the matches schema for the future match."""
    columns = matches.columns.tolist()
    data = {col: None for col in columns}
    if "match_date" in data:
        data["match_date"] = match_date
    if "player_a" in data:
        data["player_a"] = player_a
    if "player_b" in data:
        data["player_b"] = player_b
    if "winner" in data:
        data["winner"] = None
    if "score" in data:
        data["score"] = None

    if tournament and "tournament" in data:
        data["tournament"] = tournament
    if round_name and "round" in data:
        data["round"] = round_name

    return pd.DataFrame([data])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict the outcome of a badminton match.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--player-a", required=True, help="Name of player A (probability output is for this player).")
    parser.add_argument("--player-b", required=True, help="Name of player B.")
    parser.add_argument("--date", help="Match date in YYYY-MM-DD format.")
    parser.add_argument("--tournament", help="Tournament name (optional).")
    parser.add_argument("--round", dest="round_name", help="Round name (e.g., Quarterfinal).")
    args = parser.parse_args()

    result = predict_matchup(
        config_path=args.config,
        player_a=args.player_a,
        player_b=args.player_b,
        match_date=args.date,
        tournament=args.tournament,
        round_name=args.round_name,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
