from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, parse_qs, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

_DEFAULT_BASE_URL = "https://www.badmintonstatistics.net/"
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/118.0.5993.70 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_DEFAULT_TIMEOUT = 20


@dataclass
class TournamentData:
    """Container for the scraped tournament content."""

    metadata: Dict[str, object]
    statistics: Dict[str, pd.DataFrame]
    history: pd.DataFrame
    draws: Dict[str, pd.DataFrame]


class BadmintonStatisticsClient:
    """Thin client around badmintonstatistics.net tournament pages."""

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(_DEFAULT_HEADERS)

    # -- High level public API -------------------------------------------------

    def scrape_tournament(
        self,
        tournament_id: str,
        *,
        include_draws: bool = True,
    ) -> TournamentData:
        soup = self._get_tournament_soup(tournament_id)

        metadata, draw_options = self._parse_metadata(soup)
        metadata["source_id"] = tournament_id
        canonical_id = metadata.get("tournament_id") or tournament_id

        statistics = self._parse_statistics(soup)
        history = self._parse_history(soup)

        draws: Dict[str, pd.DataFrame] = {}
        if include_draws and draw_options:
            for label, query in draw_options.items():
                try:
                    draws[label] = self._fetch_draw_dataframe(canonical_id, query)
                except Exception as exc:  # pragma: no cover - network edge cases
                    raise RuntimeError(f"Failed to fetch draw '{label}': {exc}") from exc

        return TournamentData(metadata=metadata, statistics=statistics, history=history, draws=draws)

    # -- Internal helpers ------------------------------------------------------

    def _get_tournament_soup(self, tournament_id: str) -> BeautifulSoup:
        url = urljoin(self.base_url, "Tournament")
        response = self.session.get(
            url,
            params={"tournamentid": tournament_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def _parse_metadata(self, soup: BeautifulSoup) -> Tuple[Dict[str, object], Dict[str, str]]:
        metadata: Dict[str, object] = {}

        def _value(selector: str, attr: str = "value") -> Optional[str]:
            node = soup.select_one(selector)
            return node.get(attr) if node else None

        metadata["tournament_id"] = _value("#tournamentid")
        metadata["tournament_family"] = _value("#tournamentFamily")
        metadata["start_date"] = _value("#mindate")
        metadata["end_date"] = _value("#maxdate")

        title_tag = soup.find("title")
        metadata["page_title"] = title_tag.get_text(strip=True) if title_tag else None

        heading = soup.select_one("span > b")
        metadata["tournament_name"] = heading.get_text(strip=True) if heading else None

        selected_year = soup.select_one("#yearSelect option[selected]")
        metadata["season_year"] = selected_year.get_text(strip=True) if selected_year else None

        year_options: List[Dict[str, str]] = []
        for opt in soup.select("#yearSelect option"):
            value = opt.get("value")
            label = opt.get_text(strip=True)
            if not value:
                continue
            year_options.append(
                {
                    "label": label,
                    "tournament_id": value,
                    "selected": bool(opt.has_attr("selected")),
                }
            )

        metadata["available_years"] = [item["label"] for item in year_options]
        metadata["available_year_ids"] = {item["label"]: item["tournament_id"] for item in year_options}
        metadata["year_options"] = year_options

        metadata["stat_categories"] = [
            opt.get("value", "").strip()
            for opt in soup.select("#statCategories option")
            if opt.get("value")
        ]
        metadata["round_filters"] = [
            opt.get("value", "").strip() for opt in soup.select("#roundselect option") if opt.get("value")
        ]

        draw_options: Dict[str, str] = {}
        for opt in soup.select("#drawSelect option"):
            raw_value = (opt.get("value") or "").strip()
            label = opt.get_text(strip=True)
            if not raw_value or opt.has_attr("disabled"):
                continue
            draw_options[label] = raw_value

        metadata["draw_categories"] = list(draw_options.keys())
        return metadata, draw_options

    def _parse_statistics(self, soup: BeautifulSoup) -> Dict[str, pd.DataFrame]:
        container = soup.select_one("#tournamentstatisticspartial")
        if not container:
            return {}

        statistics: Dict[str, pd.DataFrame] = {}
        for heading in container.find_all("h4"):
            title = heading.get_text(strip=True)
            table = heading.find_next("table")
            if not table:
                continue
            statistics[title] = _html_table_to_dataframe(table)
        return statistics

    def _parse_history(self, soup: BeautifulSoup) -> pd.DataFrame:
        table = soup.select_one("#HistoricalWinners table")
        if not table:
            return pd.DataFrame()
        return _html_table_to_dataframe(table)

    def _fetch_draw_dataframe(self, tournament_id: str, query: str) -> pd.DataFrame:
        params = {"tournamentid": tournament_id}
        params.update(dict(parse_qsl(query, keep_blank_values=True)))

        url = urljoin(self.base_url, "home/TournamentDrawPartial")
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", class_="drawtable")
        if table is None:
            raise ValueError("draw table not found in partial response")

        df = _html_table_to_dataframe(table)
        header_values = df.columns.tolist()
        # Drop duplicated header rows that appear at the bottom of the draw table
        df = df.loc[~(df.apply(lambda row: row.tolist() == header_values, axis=1))]
        return df.reset_index(drop=True)


# -- Utility functions ---------------------------------------------------------


def _normalize_cell_text(cell: Tag) -> str:
    text = cell.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("\xa0", " ")


def _html_table_to_dataframe(table: Tag) -> pd.DataFrame:
    header_cells = [
        _normalize_cell_text(th) for th in table.select("thead tr th")
    ]

    if not header_cells:
        first_row = table.find("tr")
        if first_row:
            header_cells = [_normalize_cell_text(cell) for cell in first_row.find_all(["th", "td"])]

    body_rows: Iterable[Tag]
    if table.find("tbody"):
        body_rows = table.find_all("tbody")
        row_tags = []
        for body in body_rows:
            row_tags.extend(body.find_all("tr"))
    else:
        row_tags = table.find_all("tr")[1:] if header_cells else table.find_all("tr")

    max_columns = len(header_cells)
    for row in row_tags:
        cells = row.find_all(["td", "th"])
        if len(cells) > max_columns:
            max_columns = len(cells)

    rows: List[List[str]] = []
    for row in row_tags:
        cells = [_normalize_cell_text(cell) for cell in row.find_all(["td", "th"])]
        if not cells:
            continue
        if len(cells) < max_columns:
            cells.extend([""] * (max_columns - len(cells)))
        rows.append(cells[:max_columns])

    columns = header_cells[:]
    if len(columns) < max_columns:
        for idx in range(len(columns), max_columns):
            columns.append(f"col_{idx + 1}")
    if not any(columns):
        columns = None

    dataframe = pd.DataFrame(rows, columns=columns)
    return dataframe


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return slug.strip("_") or "section"


# -- CLI -----------------------------------------------------------------------


def write_outputs(
    payload: TournamentData,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(payload.metadata, fp, indent=2, ensure_ascii=False)

    history_path = output_dir / "history.csv"
    payload.history.to_csv(history_path, index=False)

    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(exist_ok=True)
    for title, frame in payload.statistics.items():
        slug = slugify(title)
        frame.to_csv(stats_dir / f"{slug}.csv", index=False)

    if payload.draws:
        draws_dir = output_dir / "draws"
        draws_dir.mkdir(exist_ok=True)
        for label, frame in payload.draws.items():
            slug = slugify(label)
            frame.to_csv(draws_dir / f"{slug}.csv", index=False)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Scrape tournament data from badmintonstatistics.net")
    parser.add_argument("--tournament-id", help="Tournament identifier from badmintonstatistics.net")
    parser.add_argument("--tournament-url", help="Full tournament page URL; the tournament id will be parsed from the query string.")
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL, help="Base URL of badmintonstatistics.net")
    parser.add_argument("--years", nargs="+", help="Specific season years to scrape (e.g., 2025 2024).")
    parser.add_argument("--all-years", action="store_true", help="Scrape every season listed in the year dropdown.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory for scraped outputs (used when scraping multiple seasons).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where scraped data will be written (single-season mode only, default: data/raw/<slug>).",
    )
    parser.add_argument(
        "--skip-draws",
        action="store_true",
        help="Skip fetching draw brackets (they require extra requests).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print a short preview of the scraped data instead of writing files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.tournament_id and not args.tournament_url:
        parser.error("Provide either --tournament-id or --tournament-url.")

    client = BadmintonStatisticsClient(base_url=args.base_url)

    def _resolve_tournament_id() -> str:
        if args.tournament_id:
            return args.tournament_id
        parsed = urlparse(args.tournament_url)
        query = parse_qs(parsed.query)
        tournament_id = query.get("tournamentid")
        if not tournament_id:
            parser.error("Could not find 'tournamentid' parameter in the provided URL.")
        return tournament_id[0]

    tournament_id = _resolve_tournament_id()

    if args.output_dir and (args.all_years or args.years):
        parser.error("--output-dir cannot be combined with --all-years/--years.")

    multi_year = args.all_years or bool(args.years)

    def _handle_output(dataset: TournamentData, year_label: Optional[str] = None) -> None:
        if args.preview:
            print(json.dumps(dataset.metadata, indent=2, ensure_ascii=False))
            if not dataset.history.empty:
                print("\nHistory head:")
                print(dataset.history.head())
            if dataset.statistics:
                print("\nStatistics sections:")
                for title, frame in dataset.statistics.items():
                    print(f"- {title}: {len(frame)} rows, columns={list(frame.columns)}")
            if dataset.draws:
                print("\nDraw sections:")
                for label, frame in dataset.draws.items():
                    print(f"- {label}: {len(frame)} rows")
            print("-" * 60)
            return

        folder_title = dataset.metadata.get("tournament_name") or dataset.metadata.get("page_title")
        folder_slug = slugify(folder_title or dataset.metadata.get("season_year") or dataset.metadata.get("source_id"))
        if year_label:
            folder_slug = f"{folder_slug}_{year_label}"

        output_dir = args.output_dir or args.output_root / folder_slug
        write_outputs(dataset, output_dir)
        print(f"Wrote scraped data to {output_dir}")

    if not multi_year:
        tournament = client.scrape_tournament(tournament_id, include_draws=not args.skip_draws)
        _handle_output(tournament, year_label=tournament.metadata.get("season_year"))
        if args.preview:
            return
        if args.output_dir:
            return
        return

    first_dataset = client.scrape_tournament(tournament_id, include_draws=not args.skip_draws)
    year_options = first_dataset.metadata.get("year_options", [])
    if not year_options:
        parser.error("No year dropdown information found; cannot scrape multiple seasons automatically.")

    available_order = [item["label"] for item in year_options]
    year_id_map = {item["label"]: item["tournament_id"] for item in year_options}

    requested_years = available_order if args.all_years else args.years
    if not requested_years:
        parser.error("Specify --years or --all-years when automating multiple seasons.")

    missing_years = [year for year in requested_years if year not in year_id_map]
    if missing_years:
        parser.error(f"Years not available on the tournament page: {', '.join(missing_years)}")

    results: List[Tuple[str, TournamentData]] = []
    for year in requested_years:
        year_identifier = year_id_map[year]
        if year_identifier == first_dataset.metadata.get("tournament_id"):
            dataset = first_dataset
        else:
            dataset = client.scrape_tournament(year_identifier, include_draws=not args.skip_draws)
        results.append((year, dataset))

    for year, dataset in results:
        _handle_output(dataset, year_label=year)


if __name__ == "__main__":  # pragma: no cover
    main()
