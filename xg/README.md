# xG

Shot-level **expected Goals (xG)** for SkillCorner Dynamic Events.

This directory contains a self-contained tutorial and reusable model-building
utilities. The tutorial uses SkillCorner Open Data by default, while the Python
modules can be pointed at a local SkillCorner match-bundle root.

## Tutorial And Analysis

| Resource | Link |
|---|---|
| Tutorial notebook | `xg/notebooks/xg_tutorial.ipynb` |
| Example analysis | [Week 1: xG analysis](https://kaisport.github.io/posts/week1-xg-en.html) |

The notebook is intentionally self-contained: it builds the shot table, creates
core features, trains compact models, compares a module-rich feature pipeline,
and visualizes both smooth logistic and compact XGBoost xG surfaces.

## Input Data

The default tutorial input is the public
[SkillCorner Open Data](https://github.com/SkillCorner/opendata) sample.

For local data, point the scripts or notebook to a SkillCorner match-bundle root:

```bash
export SKILLCORNER_ROOT=/path/to/skillcorner/matches
```

## Shot Definition

SkillCorner Dynamic Events are converted into shots with:

```text
event_type == "player_possession" and end_type == "shot"
```

The target is:

```text
goal = game_interruption_after == "goal_for"
```

Coordinates are represented on a center-origin 105 x 68 meter pitch, with event
locations treated as attacking left to right.

## Feature And Model Overview

The tutorial shows three levels of xG modelling:

| Model | Purpose |
|---|---|
| `distance + angle` compact model | The minimum geometry baseline. |
| `distance + angle + in_box + header` compact model | A readable tree-based extension used for surface comparison. |
| module-rich pipeline | The reusable feature set from `xg_features.py`, with geometry, movement, possession context, defensive-line context, phase/game-state fields, and categorical descriptors where available. |

For visualization, the notebook compares:

- a smooth logistic surface based on `distance_to_goal` and `shot_angle`
- a compact XGBoost surface that can reveal blocky/non-monotonic artifacts on
  small public samples

The richer reusable pipeline is implemented in:

```text
xg_features.py
train_skillcorner_xg.py
xg_surface.py
```

## Build A Shot Table

```bash
python -m xg.skillcorner_shots \
  --skillcorner-root /path/to/skillcorner/matches \
  --out /path/to/output/skillcorner_xg/shots.parquet
```

Optional filters:

```bash
python -m xg.skillcorner_shots \
  --skillcorner-root /path/to/skillcorner/matches \
  --season-names 2023 2024 \
  --limit-matches 20 \
  --out /path/to/output/skillcorner_xg/shots.parquet
```

## Train Models

```bash
python -m xg.train_skillcorner_xg \
  --shots /path/to/output/skillcorner_xg/shots.parquet \
  --train-seasons 2023 \
  --test-seasons 2024 \
  --models logistic xgboost lightgbm
```

Model outputs are written to the configured output directory:

```text
skillcorner_xg_best.joblib
scored_shots.parquet
metrics.json
model_comparison.csv
model_coefficients.csv
```

## Caveats

SkillCorner Open Data is a small public sample. It is excellent for inspecting
the workflow, but a stable production xG model should be trained and validated
on a larger local dataset with carefully chosen train/test splits.
