"""VAEP labels: does the acting team score / concede within the next actions.

For each action ``i`` by team ``T``, look ahead over the next ``nr_actions`` actions
(inclusive of ``i``) within the same game: ``scores`` = 1 if ``T`` scores in that
window, ``concedes`` = 1 if ``T`` concedes. Both derive from the ``scoring_team_id``
column added by :func:`statsbomb_actions.add_bookkeeping`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NR_ACTIONS = 10


def add_labels(actions: pd.DataFrame, nr_actions: int = NR_ACTIONS) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, concedes)`` int arrays aligned to ``actions`` order."""
    a = actions.reset_index(drop=True)
    gb = a.groupby("match_id", sort=False)
    team = a["team_id"].fillna("").to_numpy()
    opp = a["opp_team_id"].fillna("").to_numpy()

    scores = np.zeros(len(a), dtype=bool)
    concedes = np.zeros(len(a), dtype=bool)
    for k in range(nr_actions):
        fut = gb["scoring_team_id"].shift(-k).fillna("").to_numpy()
        has_goal = fut != ""
        scores |= has_goal & (fut == team)
        concedes |= has_goal & (fut == opp)
    return scores.astype(int), concedes.astype(int)
