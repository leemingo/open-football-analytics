# xThreat

Action-based **expected Threat (xT)** utilities for SkillCorner Dynamic Events.

xT values a pass or carry by the change in expected possession value between the
start and end locations. The model in this directory learns a pitch grid through
value iteration in the Karun Singh style.

## Tutorial And Analysis

| Resource | Link |
|---|---|
| Tutorial notebook | `xthreat/notebooks/xthreat_tutorial.ipynb` |
| Example analysis | [Week 3: xT analysis](https://kaisport.github.io/posts/week3-xt-en.html) |

The notebook builds pass/carry/shot actions directly from SkillCorner Dynamic
Events, learns an xT grid, scores successful moves, compares pass xT vs carry
xT, inspects the top creator's routes, and animates one long carry using
tracking data.

## Input Data

The default tutorial input is the public
[SkillCorner Open Data](https://github.com/SkillCorner/opendata) sample.

For local data, point the scripts or notebook to a SkillCorner match-bundle root:

```bash
export SKILLCORNER_ROOT=/path/to/skillcorner/matches
```

## Action Definition

SkillCorner Dynamic Events are converted into:

| xT action | Source |
|---|---|
| pass | `event_type == "player_possession"` and `end_type == "pass"` |
| carry | `carry == true`, or the same `player_possession` row moves at least `carry_min_distance` meters from start coordinates to end coordinates |
| shot | `event_type == "player_possession"` and `end_type == "shot"` |

The carry fallback is intentionally row-local: it compares `x_start, y_start`
and `x_end, y_end` from the same player-possession row, so it does not join two
different players' actions.

Coordinates are represented on a center-origin 105 x 68 meter pitch, with event
locations treated as attacking left to right.

## Model Overview

The xT workflow:

1. Build an action table with pass, carry, and shot rows.
2. Estimate per-cell shot probability, move probability, scoring probability,
   and successful-move transition probabilities.
3. Learn the xT grid through value iteration.
4. Score successful passes and carries with:

```text
xT added = xT(destination) - xT(origin)
```

Reusable code lives in:

```text
skillcorner_actions.py
xthreat_model.py
train_skillcorner_xthreat.py
xthreat_plots.py
```

## Build An Action Table

Download the public SkillCorner Open Data files needed by this workflow and
write the action table:

```bash
python -m xthreat.skillcorner_actions \
  --download-opendata \
  --opendata-out-dir /path/to/output/skillcorner_opendata \
  --out /path/to/output/skillcorner_xthreat/actions.parquet
```

If you already have the Open Data repo locally, pass the match root instead:

```bash
python -m xthreat.skillcorner_actions \
  --skillcorner-root /path/to/opendata/data/matches \
  --out /path/to/output/skillcorner_xthreat/actions.parquet
```

The same command also works with a local SkillCorner match-bundle root containing
`matches_index.csv`, or folders such as `{match_id}/dynamic_events.csv` plus
`match_meta.json`.

## Fit And Score xT

```bash
python -m xthreat.train_skillcorner_xthreat \
  --actions /path/to/output/skillcorner_xthreat/actions.parquet \
  --out-dir /path/to/output/skillcorner_xthreat
```

Model outputs are written to the configured output directory:

```text
actions.parquet
actions.summary.json
skillcorner_xthreat_model.joblib
scored_actions.parquet
surface.json
metrics.json
team_xthreat.csv
player_xthreat.csv
```

## Tracking Animation Note

The tutorial can animate the longest carry for the selected top xT creator.
SkillCorner Open Data tracking files are Git LFS objects, so the notebook uses
GitHub's media endpoint and validates that the downloaded file is real JSONL
rather than a small LFS pointer file.

If you cloned SkillCorner Open Data locally, run `git lfs pull` before pointing
the notebook at that local data.

## Caveats

SkillCorner Open Data is a small public sample. The learned surface is useful as
a transparent tutorial model, but robust xT evaluation should use a larger local
dataset and carefully reviewed action definitions.
