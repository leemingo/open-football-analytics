"""The VAEP value formula.

    offensive(a_i) = P_scores(S_i)   - P_scores(S_{i-1})
    defensive(a_i) = -(P_concedes(S_i) - P_concedes(S_{i-1}))
    VAEP(a_i)      = offensive(a_i) + defensive(a_i)

When possession changes between ``a_{i-1}`` and ``a_i`` the previous state's
scoring/conceding probabilities swap (scoring is always from the acting team's
view). The previous state is reset at phase boundaries so a restart never inherits
the prior action's danger through the swap: after a goal (the restart kickoff), and
at any dead-ball restart (goalkick / throw-in / corner / free-kick) -- otherwise a
goalkick after the opponent's off-target shot would inflate goalkeeper value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESTART_TYPES = (
    "goalkick", "throw_in", "corner_crossed", "corner_short",
    "freekick_short", "freekick_crossed",
)


def compute_values(actions: pd.DataFrame, p_scores: np.ndarray, p_concedes: np.ndarray) -> pd.DataFrame:
    """Return ``offensive_value``, ``defensive_value``, ``vaep_value`` (aligned to actions)."""
    a = actions.reset_index(drop=True)
    df = pd.DataFrame(
        {
            "match_id": a["match_id"].to_numpy(),
            "team_id": a["team_id"].fillna("").to_numpy(),
            "is_goal": a["scoring_team_id"].notna().to_numpy(),
            "type_name": a["type_name"].astype("string").fillna("").to_numpy(),
            "ps": np.asarray(p_scores, dtype="float64"),
            "pc": np.asarray(p_concedes, dtype="float64"),
        }
    )
    gb = df.groupby("match_id", sort=False)
    prev_ps = gb["ps"].shift(1).to_numpy()
    prev_pc = gb["pc"].shift(1).to_numpy()
    prev_team = gb["team_id"].shift(1).fillna("").to_numpy()

    same_team = prev_team == df["team_id"].to_numpy()
    prev_scores = np.nan_to_num(np.where(same_team, prev_ps, prev_pc), nan=0.0)
    prev_concedes = np.nan_to_num(np.where(same_team, prev_pc, prev_ps), nan=0.0)

    prev_was_goal = gb["is_goal"].shift(1).fillna(False).to_numpy().astype(bool)
    is_restart = np.isin(df["type_name"].to_numpy().astype(str), RESTART_TYPES)
    reset = prev_was_goal | is_restart
    prev_scores = np.where(reset, 0.0, prev_scores)
    prev_concedes = np.where(reset, 0.0, prev_concedes)

    offensive = df["ps"].to_numpy() - prev_scores
    defensive = -(df["pc"].to_numpy() - prev_concedes)
    return pd.DataFrame(
        {"offensive_value": offensive, "defensive_value": defensive, "vaep_value": offensive + defensive},
        index=a.index,
    )
