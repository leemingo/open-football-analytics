# xG (expected Goals) Pipeline

Shot-level **xG** for SkillCorner Dynamic Events and Bepro K League events. Estimates the
probability a shot becomes a goal from its location, geometry, and context.

See the repo-level [README](../README.md) for data sources (open vs licensed) and the
tutorial-vs-analysis split.

## Directory Layout

```text
xg/
  xg_features.py             # shot-level feature engineering (distance, open angle, in-box, ...)
  train_skillcorner_xg.py    # logistic / XGBoost / LightGBM training + evaluation + calibration
  xg_surface.py              # location xG surface plotting (mplsoccer)
  skillcorner_shots.py       # build the shot table from SkillCorner Dynamic Events (licensed K League)
  bepro_drive_shots.py       # build the shot table from Bepro events on Google Drive (rclone)
  bepro_drive_players.py     # player-name/position lookup for Bepro shots
  week1_report.py      # regenerate the published week1-xg post figures (Seoul/Ulsan story)
  notebooks/                               # two notebooks:
    00_xg_from_scratch.ipynb               # (1) TUTORIAL + simple analysis — build xG inline on open data (DATA_SOURCE toggle)
    week1_xg_analysis.ipynb                # (2) reproduce the published week1-xg analyses (runs week1_report.py)
```

Generated data/models (`tmp/data/`) and report outputs (`reports/`, written by `week1_report.py`)
are gitignored; the published week1-xg figures live on the KAISport site.

## Shot definition

Both vendors converge to a centre-origin, attacker-left-to-right frame (goal at `(+52.5, 0)`).

- **SkillCorner**: `event_type == "player_possession"` and `end_type == "shot"`;
  goal `= game_interruption_after == "goal_for"`.
- **Bepro**: an event whose `event_types` contains a `Shot`; goal `= outcome == "Goal"`.

## Tutorial (open data — reproducible by anyone)

`notebooks/00_xg_from_scratch.ipynb` implements xG inline (distance/angle features → XGBoost →
log loss / Brier / calibration → xG surface) and then verifies it matches the modules below.
A `DATA_SOURCE` toggle switches between public SkillCorner Open Data (`"opendata"`, default —
auto-downloaded, fully reproducible) and licensed K League (`"kleague"`).

## Analysis (licensed data — post reproduction)

```bash
# SkillCorner K League shot table
python -m xg.skillcorner_shots --skillcorner-root /data2/MHL/data/skillcorner/kleague \
  --season-names 2023 2024 2025 --out tmp/data/skillcorner_xg/shots.parquet

# Train + compare models
python -m xg.train_skillcorner_xg --shots tmp/data/skillcorner_xg/shots.parquet \
  --train-seasons 2023 2024 --test-seasons 2025 --models logistic xgboost lightgbm

# Regenerate the published week1-xg figures (Bepro data)
python -m xg.week1_report   # reads tmp/data/bepro_drive_xg_k1 -> reports/bepro_week1
```

`train_skillcorner_xg` writes `scored_shots.parquet` (best model by validation log loss),
`metrics.json`, `model_comparison.csv`, and per-model artefacts under the output dir.
