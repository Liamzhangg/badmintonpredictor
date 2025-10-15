from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scrape import TournamentData

_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tournaments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_guid TEXT NOT NULL UNIQUE,
    source_id TEXT,
    name TEXT,
    season_year TEXT,
    start_date TEXT,
    end_date TEXT,
    location TEXT,
    tournament_category TEXT,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER NOT NULL REFERENCES tournaments(id) ON DELETE CASCADE,
    category_code TEXT,
    category_label TEXT,
    category_raw TEXT,
    round TEXT,
    player_a TEXT,
    player_a_rank TEXT,
    player_b TEXT,
    player_b_rank TEXT,
    score TEXT,
    duration_minutes INTEGER,
    UNIQUE(tournament_id, category_raw, round, player_a, player_b, score)
);
"""


def get_connection(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    return conn


def store_tournament(conn: sqlite3.Connection, tournament: "TournamentData") -> int:
    """Persist a scraped tournament and its matches into the SQLite database."""

    metadata = tournament.metadata
    tournament_guid = metadata.get("tournament_id") or metadata.get("source_id")
    if not tournament_guid:
        raise ValueError("Tournament metadata missing 'tournament_id'")

    metadata_json = json.dumps(metadata, ensure_ascii=False)

    conn.execute(
        """
        INSERT INTO tournaments (
            tournament_guid,
            source_id,
            name,
            season_year,
            start_date,
            end_date,
            location,
            tournament_category,
            metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tournament_guid) DO UPDATE SET
            source_id=excluded.source_id,
            name=excluded.name,
            season_year=excluded.season_year,
            start_date=excluded.start_date,
            end_date=excluded.end_date,
            location=excluded.location,
            tournament_category=excluded.tournament_category,
            metadata_json=excluded.metadata_json
        """,
        (
            tournament_guid,
            metadata.get("source_id"),
            metadata.get("tournament_name") or metadata.get("page_title"),
            metadata.get("season_year"),
            metadata.get("start_date"),
            metadata.get("end_date"),
            metadata.get("location"),
            metadata.get("tournament_category"),
            metadata_json,
        ),
    )

    row = conn.execute(
        "SELECT id FROM tournaments WHERE tournament_guid = ?", (tournament_guid,)
    ).fetchone()
    if not row:
        raise RuntimeError("Failed to resolve tournament primary key after insert.")

    tournament_db_id = row["id"]
    conn.execute("DELETE FROM matches WHERE tournament_id = ?", (tournament_db_id,))

    match_records = []
    for record in tournament.matches.to_dict("records"):
        duration = record.get("duration_minutes")
        if duration in ("", None):
            duration = None
        match_records.append(
            (
                tournament_db_id,
                record.get("category_code") or "",
                record.get("category_label") or "",
                record.get("category") or "",
                record.get("round") or "",
                record.get("player_a") or "",
                record.get("player_a_rank") or "",
                record.get("player_b") or "",
                record.get("player_b_rank") or "",
                record.get("score") or "",
                duration,
            )
        )

    if match_records:
        conn.executemany(
            """
            INSERT INTO matches (
                tournament_id,
                category_code,
                category_label,
                category_raw,
                round,
                player_a,
                player_a_rank,
                player_b,
                player_b_rank,
                score,
                duration_minutes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tournament_id, category_raw, round, player_a, player_b, score)
            DO UPDATE SET
                player_a_rank=excluded.player_a_rank,
                player_b_rank=excluded.player_b_rank,
                duration_minutes=excluded.duration_minutes
            """,
            match_records,
        )

    conn.commit()
    return len(match_records)
