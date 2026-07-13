# open-football-analytics

Open, reproducible football analytics tutorials and utilities built on public
event and tracking data. The current examples cover **expected Goals (xG)**,
**expected Pass (xPass)**, **expected Threat (xT)**, and **VAEP** (valuing every
on-the-ball action), with the project designed to grow into a broader collection
of football metrics.

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
| VAEP | `vaep/notebooks/vaep_tutorial.ipynb` | Build a SPADL action table from DFL/Sportec open data, train P(scores)/P(concedes), and value every action (offensive + defensive). |

## Example Analyses

These public posts show the same metric ideas in analysis form:

| Metric | Analysis |
|---|---|
| xG | [Week 1: xG analysis](https://kaisport.github.io/posts/week1-xg-en.html) |
| xPass | [Week 2: xPass analysis](https://kaisport.github.io/posts/week2-xpass-en.html) |
| xT | [Week 3: xT analysis](https://kaisport.github.io/posts/week3-xt-en.html) |

## Repository Map

| Path | Purpose |
|---|---|
| `football-cdf/` | Provider preprocessing utilities, CDF tracking/event conversion, SPADL-style action conversion, and the CDF tutorial notebook. |
| `xg/` | Shot table construction, xG features, model training, and xG surface plotting. |
| `xpass/` | Pass table construction, xPass features, model training, SkillCorner benchmark comparison, and PAx summaries. |
| `xthreat/` | Action table helpers, xT grid/value-iteration model, route plots, summaries, and animation examples. |
| `vaep/` | SPADL action table from DFL/Sportec open data, VAEP features/labels, the two-head P(scores)/P(concedes) model, the VAEP formula, and per-player action-value ratings. |
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

### VAEP data (DFL/Sportec open data)

VAEP needs a typed **action stream**, which SkillCorner Open Data does not provide,
so the VAEP tutorial uses the public **DFL/IDSSE** open dataset (7 Bundesliga
2022/23 matches, CC-BY) — the same open data as `kloppy.sportec.load_open_*` —
converted to SPADL-style actions with `football-cdf`'s Sportec preprocessor and
downloaded automatically by `vaep.sportec_actions`.

**Limitation — no carries.** DFL open events are *point* events: one location per
event, with no end/receiver coordinate. A moving action's end is therefore
reconstructed as the next recorded on-ball location, and **explicit carry/dribble
actions are not produced** — ball-carrying value is folded into the (linked) pass
end. Feeds that record end locations (e.g. Bepro, StatsBomb) do support separate
carries; the K-League analysis in `kleague-insights`, built on Bepro, keeps them.

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

`vaep.sportec_actions` downloads the DFL/IDSSE open data and builds a SPADL action
table (via the `football_cdf` Sportec chain → typed actions; end coordinates are
reconstructed from the next event because DFL events are point events, so **no
separate carry actions** — see **Data**). `vaep.vaep_features` builds
socceraction-style game-state features, `vaep.vaep_labels` marks whether the team
scores/concedes within the next actions, and `vaep.vaep_model` trains the two
P(scores)/P(concedes) heads. `vaep.vaep_formula` combines them into offensive +
defensive action value, and `vaep.train_sportec_vaep` scores actions and exports
per-player ratings.
