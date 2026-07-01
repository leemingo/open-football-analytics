# xPass

Pass-level **expected Pass (xPass)** for SkillCorner Dynamic Events.

xPass estimates pass completion probability from pass location, target location,
pass geometry, and pre-pass context. SkillCorner's provided xPass value is kept
as a benchmark when it exists in the source table.

## Tutorial And Analysis

| Resource | Link |
|---|---|
| Tutorial notebook | `xpass/notebooks/xpass_tutorial.ipynb` |
| Example analysis | [Week 2: xPass analysis](https://kaisport.github.io/posts/week2-xpass-en.html) |

The notebook builds the pass table from Dynamic Events, trains compact and
module-rich xPass models, compares against SkillCorner's benchmark, and computes
PAx (Pass Above Expected).

## Input Data

The default tutorial input is the public
[SkillCorner Open Data](https://github.com/SkillCorner/opendata) sample.

For local data, point the scripts or notebook to a SkillCorner match-bundle root:

```bash
export SKILLCORNER_ROOT=/path/to/skillcorner/matches
```

## Pass Definition

SkillCorner Dynamic Events are converted into passes with:

```text
event_type == "player_possession" and end_type == "pass"
```

The binary target is:

```text
pass_completed = pass_outcome == "successful"
```

Offside pass outcomes are retained in the raw table but excluded from model
training by default.

## Feature And Model Overview

The custom model uses pre-pass information only. It deliberately excludes
SkillCorner's own xPass/xThreat benchmark fields and post-outcome matching
fields so the model does not learn from information that would only be known
after the pass.

The reusable feature pipeline in `xpass_features.py` includes:

- passer and target coordinates
- pass distance, angle, progression, and target goal-distance gain
- sideline and goal-distance geometry
- possession context
- passing-option and defensive context fields when available
- categorical player/zone descriptors when available

Training and comparison utilities live in:

```text
skillcorner_passes.py
xpass_features.py
train_skillcorner_xpass.py
xpass_plots.py
```

## Build A Pass Table

```bash
python -m xpass.skillcorner_passes \
  --skillcorner-root /path/to/skillcorner/matches \
  --out /path/to/output/skillcorner_xpass/passes.parquet
```

## Add Features

Feature engineering is usually called by the training script, but it can also be
run directly:

```bash
python -m xpass.xpass_features \
  --passes /path/to/output/skillcorner_xpass/passes.parquet \
  --out /path/to/output/skillcorner_xpass/passes_with_features.parquet
```

## Train And Compare Models

```bash
python -m xpass.train_skillcorner_xpass \
  --passes /path/to/output/skillcorner_xpass/passes.parquet \
  --train-seasons 2023 \
  --test-seasons 2024 \
  --models logistic xgboost
```

Model outputs are written to the configured output directory:

```text
skillcorner_xpass_best.joblib
scored_passes.parquet
metrics.json
model_comparison.csv
skillcorner_comparison.csv
team_pax.csv
player_pax.csv
```

## PAx

PAx follows the Opta Analyst framing:

```text
PAx = completed passes - expected completed passes
PAx per 100 = PAx / passes * 100
```

Positive PAx means a player or team completed more passes than expected given
the modelled difficulty of those passes.

## Caveats

SkillCorner Open Data is a small public sample. It is useful for demonstrating
the workflow and validating code paths, but robust xPass evaluation needs a
larger local dataset and careful calibration checks.
