# football-analytics

K League football analytics — **expected Goals (xG)**, **expected Pass (xPass)**, and
**expected Threat (xT)**.

This repo serves two purposes:

1. **Analysis** — reproduce the figures behind the KAISports K League blog posts
   (<https://kaisport.github.io/insights.html>) from licensed SkillCorner / Bepro data.
2. **Open tutorials** — learn and re-implement each metric *from scratch* on the **public
   SkillCorner Open Data**, so anyone can reproduce the methods without licensed data.

## Data sources

| Source | Access | Used for |
|---|---|---|
| SkillCorner **Open Data** (10 matches, A-League) | Public — [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata), auto-downloaded by the tutorials | Tutorials (`00_*_from_scratch.ipynb`) |
| SkillCorner **K League** (full seasons) | Licensed — `/data2/MHL/data/skillcorner/kleague` | Analysis / post reproduction |
| **Bepro** K League | Licensed — Google Drive via `rclone` | Analysis / post reproduction |

Tutorials switch between sources with a `DATA_SOURCE` toggle; the default `"opendata"` is
public and reproducible by everyone. The licensed K League / Bepro data are **not** public —
only the open-data path is fully reproducible outside this machine.

## Layout

Each metric is a Python package — `xg/`, `xpass/`, `xthreat/`. Within each:

| File pattern | Role |
|---|---|
| `*_features.py` | feature engineering |
| `train_*.py` / `*_model.py` | models (logistic / XGBoost / LightGBM, or the xT surface) |
| `*_plots.py` | plotting (mplsoccer pitches) |
| `skillcorner_*.py`, `bepro_*.py` | data builders — **both vendors kept** |
| `*_report.py` | script that regenerates a blog post's figures |
| `notebooks/` | **exactly two per metric**: `00_*_from_scratch.ipynb` (open-data tutorial + simple analysis) and one report notebook that reproduces the published post's analyses |
| `*_report.py` outputs | written under `reports/` — **gitignored** (regenerated on demand; published copies live on the KAISport site) |

Shared: `football-cdf/` (submodule, imported as `football_cdf`), `pipelines/` (data
pipelines that build the canonical stores — e.g. `pipelines/bepro_ingest.py` builds the
Bepro **SPADL action store** the xG/xPass builders derive from), `animations/` (pitch
animation helpers, used by tutorials/analysis as needed), `tmp/` (generated data — gitignored),
`tools/` + `.secrets/` (rclone binary + config — gitignored).

The Bepro pipeline is standardized on SPADL: `python -m pipelines.bepro_ingest` builds
`tmp/data/bepro_spadl_k1/{actions,match_meta}.parquet` once (via the shared
`football_cdf` Bepro→SPADL converter), and `xg/bepro_drive_shots.py` /
`xpass/bepro_drive_passes.py` derive the shot / pass tables from it.

## Blog post → producing code

| Post | Producer | Data |
|---|---|---|
| `week1-xg` (Seoul xG story) | `xg/week1_report.py` (notebook `week1_xg_analysis.ipynb` reproduces it) | `tmp/data/bepro_drive_xg_k1` (Bepro) |
| `week2-xpass` | `xpass/week2_report.py` | `tmp/data/skillcorner_xpass/passes.parquet` (SkillCorner **provided** xPass) |

(For xPass, Bepro keeps only the pass builder `xpass/bepro_drive_passes.py` for future analysis;
a publishable Bepro xPass model is future work — Bepro's realized next-touch leaks the outcome.)

## Setup

Clone with the `football-cdf` submodule, then create an isolated **Python ≥ 3.10**
environment and install the package editable (with the extras you need:
`models` = xgboost/lightgbm, `notebooks` = jupyter). Name the environment whatever
you like — nothing in the repo depends on a specific env name.

```bash
git clone --recursive <repo-url>          # pulls the football-cdf submodule too
cd football-analytics
# (cloned without --recursive?) git submodule update --init --recursive
```

**Option A — uv (recommended):**

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[models,notebooks]"
```

**Option B — conda:**

```bash
conda create -n football-analytics python=3.11 -y
conda activate football-analytics
pip install -e ".[models,notebooks]"
```

**Jupyter kernel** — the notebooks use the generic `python3` kernel, so they run
under whatever environment you launch Jupyter from. Optionally register your env so
it shows up by name in the kernel picker:

```bash
python -m ipykernel install --user --name football-analytics --display-name "football-analytics"
```

## Tutorials (open data — reproducible by anyone)

```text
xg/notebooks/00_xg_from_scratch.ipynb            # available
xpass/notebooks/00_xpass_from_scratch.ipynb      # available
xthreat/notebooks/00_xthreat_from_scratch.ipynb  # available
```

Each builds the metric from raw SkillCorner Dynamic Events, implements the features/model/
evaluation inline, and then shows that it matches the production modules. Set
`DATA_SOURCE = "opendata"` (the default) to run entirely on public data.
