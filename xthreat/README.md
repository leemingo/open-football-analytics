# xT (expected Threat) Pipeline

Action-based **expected Threat (xT)** for SkillCorner Dynamic Events, in the Karun Singh
style: a pitch grid where each cell holds the probability that possession there eventually
leads to a goal, so a pass/carry's value is the change in cell value between its start and end.

See the repo-level [README](../README.md) for data sources (open vs licensed) and the
tutorial-vs-analysis split.

## Directory Layout

```text
xthreat/
  skillcorner_actions.py        # build the action table (pass / carry / shot) from Dynamic Events
  bepro_actions.py              # build a Bepro action table from the canonical SPADL store
  xthreat_model.py              # CenterOriginExpectedThreat (learned grid) + PrecomputedXThreat (load a published map)
  xthreat_analysis.py           # routes, sequences, credit assignment, team/player summaries
  xthreat_plots.py              # xT surface, heatmaps, sequence/route plots (mplsoccer)
  run_xthreat.py    # end-to-end run + comparison against SkillCorner-provided xT
  scripts/run_xthreat_experiment.sh
  notebooks/
    xthreat_analysis.ipynb   # pipeline walkthrough on licensed K League data
    00_xthreat_from_scratch.ipynb           # open-data tutorial: build the xT surface inline (value iteration)
```

Generated data and surfaces are kept outside the package under `tmp/data/` (gitignored).

## What it does

1. **Actions** — `skillcorner_actions.build_skillcorner_xthreat_actions` turns Dynamic Events
   into a pass/carry/shot action table with start/end coordinates (centre-origin, 105×68 m).
   `bepro_actions.build_bepro_xthreat_actions` does the same for the Bepro K League SPADL
   store built by `pipelines/bepro_ingest.py`; SPADL `dribble` actions are mapped to xT
   `carry` rows so xG, xPass, and xT share the same football-cdf-derived source of truth.
2. **Surface** — `CenterOriginExpectedThreat` learns the xT grid by value iteration; a
   published map can also be loaded via `PrecomputedXThreat` for comparison.
3. **Rating** — each action is scored by `xT_end − xT_start`; aggregated to team/player level
   and compared against SkillCorner's **provided** xT as a benchmark.

## Tutorials vs analysis

- **Analysis** (`xthreat_analysis.ipynb`, `run_xthreat.py`) runs on
  the licensed SkillCorner K League data and is the path used for internal analysis.
- **Tutorial** (`00_xthreat_from_scratch.ipynb`) builds the xT surface from scratch on the
  public SkillCorner Open Data (`DATA_SOURCE` toggle, default `"opendata"`) via inline value
  iteration, then bridges to the modules above (verifies it matches `CenterOriginExpectedThreat`).
