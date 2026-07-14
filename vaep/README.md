# VAEP

Action-level **Valuing Actions by Estimating Probabilities (VAEP)** for
StatsBomb Open Data.

VAEP assigns a value to every on-the-ball action by measuring how it changes the
acting team's probability of scoring and conceding in the near future. This
directory contains a self-contained tutorial and reusable utilities for building
the action table, training the two probability models, scoring actions, and
aggregating player ratings.

## Tutorial And Analysis

| Resource | Link |
|---|---|
| Tutorial notebook | `vaep/notebooks/vaep_tutorial.ipynb` |
| Example analysis | [Week 4: VAEP analysis](https://kaisport.github.io/posts/week4-vaep-en.html) |

The notebook implements VAEP from scratch and compares each step with the
reusable modules in this directory. Unlike the xG, xPass, and xThreat tutorials,
it uses StatsBomb Open Data because VAEP needs a typed action stream with action
results, end locations, and defensive actions.

## Input Data

The default input is the public
[StatsBomb Open Data](https://github.com/statsbomb/open-data) dataset. The
workflow downloads individual event, lineup, competition, and match files on
demand and caches them locally, so cloning the full data repository is not
required.

The default sample is the full FIFA World Cup 2022:

```text
competition_id = 43
season_id = 106
matches = 64
```

Downloaded files are cached under:

```text
tmp/data/statsbomb_open
```

Other competitions and seasons in StatsBomb Open Data can be selected with the
`--competition-id` and `--season-id` arguments.

## Action Definition

Raw StatsBomb events are converted to a SPADL-style action table with
`football_cdf.statsbomb_preprocessing.StatsbombDataPreprocessor`. The canonical
VAEP table includes:

- action type, result, and body part
- acting team and player
- provider-recorded start and end coordinates
- period-relative action time
- opponent, score-state, and goal bookkeeping

StatsBomb provides real end coordinates and explicit `Carry` events. Carries are
therefore retained as `dribble` actions without reconstructing an endpoint from
the next event.

A goal is defined as:

```text
type_name in ["shot", "shot_penalty", "shot_freekick"]
and result_name == "success"
```

Coordinates use a center-origin 105 x 68 meter pitch. Game-state features are
normalized so the acting team always attacks toward `+x`, with the goal at
`(+52.5, 0)`.

## Feature And Model Overview

The VAEP workflow:

1. Convert StatsBomb events into a chronological SPADL-style action stream.
2. Represent each state with the current action and the previous two actions.
3. Label whether the acting team scores or concedes within the next 10 actions.
4. Train separate probability models for `P(scores)` and `P(concedes)`.
5. Convert probability changes into offensive, defensive, and total VAEP.
6. Aggregate action values into player ratings and action-phase contributions.

The feature matrix in `vaep_features.py` includes:

- start and end locations
- distance and angle to goal
- action movement
- action type, result, and body-part indicators
- period, time, and score context
- time and space differences between consecutive actions

XGBoost is the default estimator for both probability heads. A logistic
regression baseline is also available. The default training command holds out
the last eight matches, so no match appears in both the training and test sets.

Reusable code lives in:

```text
statsbomb_actions.py
vaep_features.py
vaep_labels.py
vaep_model.py
vaep_formula.py
train_statsbomb_vaep.py
```

## VAEP Formula

For action `a_i`:

```text
offensive(a_i) = P_scores(S_i) - P_scores(S_i-1)
defensive(a_i) = -(P_concedes(S_i) - P_concedes(S_i-1))
VAEP(a_i)      = offensive(a_i) + defensive(a_i)
```

When possession changes, the previous state's scoring and conceding
probabilities are swapped into the new acting team's perspective. The previous
state is reset after goals and at dead-ball restarts so a restart does not
inherit danger from the preceding phase.

## Build And Inspect An Action Table

Download and process every match in the default competition and season:

```bash
python -m vaep.statsbomb_actions
```

Select another StatsBomb competition and season, or specific matches:

```bash
python -m vaep.statsbomb_actions \
  --competition-id 43 \
  --season-id 106 \
  --match-ids 3869685
```

The command prints the number of actions and matches plus the most common action
types. Downloaded raw files remain in the configured cache directory.

## Train Models And Score Actions

Run the full World Cup 2022 workflow with the default XGBoost models:

```bash
python -m vaep.train_statsbomb_vaep \
  --competition-id 43 \
  --season-id 106 \
  --test-matches 8 \
  --model xgboost \
  --out-dir /path/to/output/vaep_statsbomb
```

For a faster baseline, use:

```bash
python -m vaep.train_statsbomb_vaep --model logistic
```

Model outputs are written to the configured output directory:

```text
scored_actions.parquet
player_ratings.parquet
metrics.json
```

`scored_actions.parquet` contains the two predicted probabilities, offensive
value, defensive value, total VAEP, and train/test split for every action.
`player_ratings.parquet` contains player totals, per-action VAEP, goals, and
contributions grouped into pass, carry, shot, defensive, and other phases.

## Caveats

StatsBomb Open Data is provided for research and non-commercial use with
attribution; review its licence before redistribution or commercial use.

The World Cup sample is useful for reproducing and inspecting the complete VAEP
method, but it is not a league-strength production model. Robust player
evaluation should use a larger competition-specific dataset, carefully chosen
match-level splits, calibration checks for both probability heads, and reviewed
action and restart definitions.
