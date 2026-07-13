"""VAEP (Valuing Actions by Estimating Probabilities) on StatsBomb Open Data.

A reproducible, tutorial-oriented implementation of VAEP (Decroos et al.,
"Actions Speak Louder than Goals", KDD 2019). SkillCorner Open Data (used by the
xg/xpass/xthreat tutorials) has no SPADL-style action stream, so VAEP is built on
the public **StatsBomb Open Data** dataset instead, loaded and converted to
SPADL-style actions (with real end coordinates and explicit carries) with
``football_cdf``'s StatsBomb preprocessor.

The method follows socceraction but is implemented from scratch (no socceraction
import), mirroring how the other metric packages build on ``football_cdf``:

* ``statsbomb_actions``   -- download StatsBomb Open Data + build the SPADL action table
* ``vaep_features``       -- game-state features (centre-origin, socceraction style)
* ``vaep_labels``         -- scores / concedes within the next ``nr_actions`` actions
* ``vaep_model``          -- two gradient-boosted P(scores) / P(concedes) heads
* ``vaep_formula``        -- VAEP = dP(scores) - dP(concedes), offensive + defensive
* ``train_statsbomb_vaep``-- fit, score, and aggregate to player ratings

Coordinates are centre-origin (``x`` in [-52.5, 52.5], ``y`` in [-34, 34]); each
game state is normalized so the acting team attacks +x (goal at ``(+52.5, 0)``).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# The StatsBomb preprocessor lives in *this* repo's football-cdf submodule. Some
# environments also have an older/unrelated football-cdf checkout editable-installed
# (without StatsBomb support) -- prepend the local submodule so `import football_cdf`
# always resolves here first, regardless of what else is on the ambient PYTHONPATH.
_football_cdf_dir = _Path(__file__).resolve().parent.parent / "football-cdf"
if _football_cdf_dir.is_dir() and str(_football_cdf_dir) not in _sys.path:
    _sys.path.insert(0, str(_football_cdf_dir))
