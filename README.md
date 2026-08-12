# open-football-analytics

Open, reproducible football analytics tutorials and utilities built on public
event and tracking data. The current examples cover **expected Goals (xG)**,
**expected Pass (xPass)**, **expected Threat (xT)**, **VAEP** (valuing every
on-the-ball action), and **physical metrics** (speed bands, sustained efforts, and
ball-in-play-normalised running), with the project designed to grow into a broader
collection of football metrics.

The repository is built around a simple workflow:

1. Use `football-cdf` to normalize raw provider event/tracking data into the
   Common Data Format (CDF), following
   [Anzer et al., "Common Data Format (CDF): A Standardized Format for
   Match-Data in Football (Soccer)"](https://arxiv.org/abs/2505.15820).
   When provider events are converted into action-level tables, the action
   type/result/body-part conventions follow
   [SPADL](https://socceraction.readthedocs.io/en/latest/documentation/spadl/SPADL_definitions.html).
   Sportec / DFL tracking is loaded and normalized with
   [Kloppy](https://kloppy.pysport.org/), while SPADL-style action tables keep
   the event layer close to [socceraction](https://socceraction.readthedocs.io/).
2. Use the public [SkillCorner Open Data](https://github.com/SkillCorner/opendata)
   sample as the default reproducible dataset (VAEP is the exception — it needs a
   typed action stream, so it uses the public DFL/IDSSE open data; see **Data**).
3. Implement each metric in tutorial notebooks and reusable Python modules.

The notebooks are written as self-contained walkthroughs: the core metric logic
is visible in the notebook, while the package modules provide reusable versions
for larger local datasets.

## Quick Start

Clone the repository with the `football-cdf` submodule and install the package in
an isolated Python 3.10+ environment.

```bash
git clone --recursive <repo-url>
cd open-football-analytics
```

If you cloned without submodules:

```bash
git submodule update --init --recursive
```

Using `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[models,notebooks]"
```

Using conda:

```bash
conda create -n open-football-analytics python=3.11 -y
conda activate open-football-analytics
pip install -e ".[models,notebooks]"
```

## Tutorials

| Topic | Notebook | What it shows |
|---|---|---|
| CDF preprocessing | `football-cdf/notebooks/provider_to_cdf.ipynb` | Convert provider raw data into the common tracking/event shape used downstream, including SPADL-style action tables where available. |
| xG | `xg/notebooks/xg_tutorial.ipynb` | Build a shot table, train compact and richer xG models, and compare smooth vs tree-based xG surfaces. |
| xPass | `xpass/notebooks/xpass_tutorial.ipynb` | Build a pass table, train xPass models, compare against SkillCorner's benchmark, and compute PAx. |
| xT | `xthreat/notebooks/xthreat_tutorial.ipynb` | Build pass/carry/shot actions, learn an xT grid, compare pass vs carry xT, and animate a carry. |
| VAEP | `vaep/notebooks/vaep_tutorial.ipynb` | Build a SPADL action table from StatsBomb open data, train P(scores)/P(concedes), and value every action (offensive + defensive). |
| Physical | `physical/notebooks/physical_tutorial.ipynb` | Turn raw tracking coordinates into speed bands, sustained efforts, PSV-99, and ball-in-play-normalised distance, then rank players. |

## Example Analyses

These public posts show the same metric ideas in analysis form:

| Metric | Analysis |
|---|---|
| xG | [Week 1: xG analysis](https://kaisport.github.io/posts/week1-xg-en.html) |
| xPass | [Week 2: xPass analysis](https://kaisport.github.io/posts/week2-xpass-en.html) |
| xT | [Week 3: xT analysis](https://kaisport.github.io/posts/week3-xt-en.html) |
| VAEP | [Week 4: VAEP analysis](https://kaisport.github.io/posts/week4-vaep-en.html) |
| Physical | [Week 5: physical metrics analysis](https://kaisport.github.io/posts/week5-physical-en.html) |

## Repository Map

| Path | Purpose |
|---|---|
| `football-cdf/` | Provider preprocessing utilities, CDF tracking/event conversion, SPADL-style action conversion, and the CDF tutorial notebook. |
| `xg/` | Shot table construction, xG features, model training, and xG surface plotting. |
| `xpass/` | Pass table construction, xPass features, model training, SkillCorner benchmark comparison, and PAx summaries. |
| `xthreat/` | Action table helpers, xT grid/value-iteration model, route plots, summaries, and animation examples. |
| `vaep/` | SPADL action table from StatsBomb open data, VAEP features/labels, the two-head P(scores)/P(concedes) model, the VAEP formula, and per-player action-value ratings. |
| `physical/` | Tracking download/streaming, speed and acceleration estimation, speed bands and sustained efforts, PSV-99, TIP/OTIP splits, ball-in-play normalisation, and player leaderboards. |
| `animations/` | Lightweight pitch animation helpers for exploratory review. |

## Data

The default reproducible path uses SkillCorner Open Data:

<https://github.com/SkillCorner/opendata>

The Open Data sample is small, so tutorial models are best treated as
transparent, reproducible examples rather than final league-strength models. If
you have your own licensed SkillCorner data, the same scripts and notebooks can
be pointed at your local match-bundle root by changing the path or setting an
environment variable.

```bash
export SKILLCORNER_ROOT=/path/to/skillcorner/matches
```

### VAEP data (StatsBomb open data)

VAEP needs a typed **action stream**, which SkillCorner Open Data does not provide,
so the VAEP tutorial uses the public **StatsBomb Open Data** (FIFA World Cup 2022,
64 matches), converted to SPADL-style actions with `football-cdf`'s StatsBomb
preprocessor and downloaded automatically by `vaep.statsbomb_actions`.

### Physical data (SkillCorner Open Data tracking)

Physical metrics need **tracking**, not events, so the physical tutorial reads the
`*_tracking_extrapolated.jsonl` files from the same SkillCorner Open Data sample
(10 A-League 2024/25 matches at 10 Hz). Those files are stored with **Git LFS**
(~90 MB per match, ~915 MB for all ten) and must be fetched from the
`media.githubusercontent.com` endpoint — `raw.githubusercontent.com` returns a
~133-byte pointer file. `physical.skillcorner_tracking` handles this and verifies
each download against the remote `Content-Length`:

```bash
python -m physical.build_skillcorner_physical --download
```


## Metric Workflows

### xG

`xg.skillcorner_shots` builds a shot table from SkillCorner Dynamic Events.
`xg.xg_features` adds geometry and context features, and
`xg.train_skillcorner_xg` trains logistic, XGBoost, or LightGBM models.

### xPass

`xpass.skillcorner_passes` builds a pass table from player-possession events.
`xpass.xpass_features` adds pass geometry and context features, and
`xpass.train_skillcorner_xpass` trains completion models and compares them with
SkillCorner's provided xPass benchmark when the column is available.

### xT

`xthreat.skillcorner_actions` prepares pass/carry/shot action rows, and
`xthreat.xthreat_model` contains a center-origin expected-threat model that
learns a grid through value iteration. `xthreat.train_skillcorner_xthreat`
scores actions and exports team/player summaries.

### VAEP

`vaep.statsbomb_actions` downloads the StatsBomb open data and builds a SPADL action
table via the `football_cdf` StatsBomb chain. `vaep.vaep_features` builds
socceraction-style game-state features, `vaep.vaep_labels` marks whether the team
scores/concedes within the next actions, and `vaep.vaep_model` trains the two
P(scores)/P(concedes) heads. `vaep.vaep_formula` combines them into offensive +
defensive action value, and `vaep.train_statsbomb_vaep` scores actions and exports
per-player ratings.

### Physical

`physical.skillcorner_tracking` downloads the Open Data tracking files and streams
each match's JSONL into a canonical long tracking frame plus a lineup table.
`physical.kinematics` segments the signal, estimates velocity and acceleration, and
flags physiologically impossible samples without clipping them.
`physical.physical_features` turns that into speed-band distances, sustained efforts,
PSV-99, TIP/OTIP splits, and per-60-ball-in-play-minute rates.
`physical.normalize` holds the eligibility and aggregation rules,
`physical.explosiveness` reconstructs time-to-speed after accelerations and direction
changes, and `physical.build_skillcorner_physical` runs the match loop and writes
season profiles. Every threshold is defined in `physical.definitions`.
