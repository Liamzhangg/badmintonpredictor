# Badminton Match Predictor

This project scaffolds a Python workflow for forecasting outcomes of elite badminton matches. Because the Badminton World Federation (BWF) site does not offer a public download feed, you will need to export match history data manually (CSV / Excel) and place it under the `data/` directory. The pipeline will then ingest the data, engineer structured features (e.g., head-to-head results, player form, demographics), and train a model to estimate each player's win probability.

## Getting Started

1. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Download data from the BWF website**
   - Visit [https://bwfbadminton.com](https://bwfbadminton.com) → `Match Centre` → choose a tournament/event.
   - Use the site export or copy the results table (player names, score, round, date, etc.) into a spreadsheet.
   - Enrich the file with additional attributes you care about (age, height, handedness, ranking, etc.).
   - Save the file as `data/raw/matches_<year>.csv` (or `.xlsx`).

3. **Configure the project**
   - Update `config/example_config.yml` with the paths to the files you created.
   - Copy it to `config/config.yml` and adjust settings as needed: data locations, features to include, model type, and training parameters.

4. **Run the training pipeline**
   ```bash
   python -m badminton_predictor.train --config config/config.yml
   ```

5. **Generate predictions for a future matchup**
   ```bash
   python -m badminton_predictor.predict --config config/config.yml \
       --player-a "Viktor Axelsen" --player-b "Shi Yu Qi" --tournament "France Open 2024"
   ```

## Scraping Tournament Data

You can pull structured match statistics, historical winners, and draw brackets directly from [badmintonstatistics.net](https://www.badmintonstatistics.net/):

1. Activate your virtual environment (see _Getting Started_ step 1).
2. Run the scraper module, passing the `tournamentid` from the site URL:
   ```bash
   PYTHONPATH=src python3 -m badminton_predictor.scrape \
       --tournament-id YONEXGermanOpe20257cf \
       --output-dir data/raw/german_open_2025
   ```
   This writes `metadata.json`, `history.csv`, `matches.csv` (one row per match with players/scores), a `statistics/` folder, and optional `draws/` CSVs under the output directory.
3. Use `--preview` to inspect the payload without writing files, or `--skip-draws` if you only need the statistics tables.

To automate every season listed in the tournament dropdown, let the CLI crawl the embedded links:

```bash
PYTHONPATH=src python3 -m badminton_predictor.scrape \
    --tournament-url "https://www.badmintonstatistics.net/Tournament?tournamentid=YONEXGermanOpe20257cf" \
    --all-years \
    --output-root data/raw
```

You can limit the scrape to specific seasons with `--years 2025 2024`, and the tool will create one folder per season (e.g., `data/raw/german_open_2025/`).

To persist everything in a structured SQLite database (keyed by tournament → year → category), add:

```bash
PYTHONPATH=src python3 -m badminton_predictor.scrape \
    --tournament-url "https://www.badmintonstatistics.net/Tournament?tournamentid=YONEXGermanOpe20257cf" \
    --years 2025 2024 \
    --database data/bwf_matches.sqlite
```

Each run prints the tournament name and match count as it scrapes, then upserts the data into:

- `tournaments` (tournament GUID, year, location, category metadata)
- `matches` (round, players, scores, duration, and normalized category labels: Men's Singles, Women's Singles, etc.)

The scraper normalizes HTML tables into pandas-ready CSVs so you can plug them into the existing ingestion pipeline (e.g., add the new files to `config/config.yml` under `data.matches` or load them in notebooks for exploratory analysis).

## Project Layout

```
badmintonpredictor/
├── config/                # YAML configs for data and models
├── data/                  # Place your raw/processed datasets here (gitignored)
├── notebooks/             # Optional exploratory analysis
├── src/badminton_predictor
│   ├── __init__.py
│   ├── config.py          # Load/validate configuration files
│   ├── data.py            # Read raw match/player data into dataframes
│   ├── features.py        # Feature engineering utilities
│   ├── models.py          # Model training & evaluation
│   ├── train.py           # Orchestrate the end-to-end training pipeline
│   └── predict.py         # CLI entry point for inference on upcoming matches
└── requirements.txt
```

## Next Steps

- Begin by building a historical dataset for two or three tournaments and iterate on feature engineering.
- Use the `notebooks/` directory for exploratory data analysis (EDA) to understand which features matter most.
- Once the pipeline works on a small dataset, automate the extraction step (e.g., a scraper) and add tests.

If you need help exporting or structuring the data, let me know which tournaments you are focusing on and I can provide a template.
