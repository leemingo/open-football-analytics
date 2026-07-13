"""Game-state features for VAEP (socceraction-style, on centre-origin SPADL).

For each action we build a *game state* = the current action ``a0`` plus the
``nb_prev - 1`` preceding actions within the same game, then normalize the state so
the acting team attacks +x ("play left to right"). In the centre-origin frame that
mirror is ``x -> -x, y -> -y`` for states whose acting team is the away team
(home already attacks +x).

Features per frame ``a{s}``: start/end location, polar distance+angle to the goal
at ``(+52.5, 0)``, movement, and one-hot action-type / result / body-part; plus
``a0`` period/time/goal-score context and time/space deltas between ``a0`` and each
previous action. All columns are numeric (float32), NaN/inf-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from football_cdf.bepro_actions import actiontypes as _BASE_ACTIONTYPES
from football_cdf.bepro_actions import bodyparts as BODYPARTS
from football_cdf.bepro_actions import results as RESULTS
from football_cdf.constants import PITCH_X

# StatsBomb (like Sportec/Bepro) keeps a few action types outside the official
# SPADL-23 vocabulary; include them so their one-hots exist.
ACTIONTYPES = list(_BASE_ACTIONTYPES) + ["ball_recovery", "dispossessed", "shot_block"]

NB_PREV_ACTIONS = 3
GOAL_X = PITCH_X / 2.0  # +52.5 in the centre-origin frame
GOAL_Y = 0.0


def _san(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def compute_feature_matrix(actions: pd.DataFrame, nb_prev: int = NB_PREV_ACTIONS) -> pd.DataFrame:
    """Return the numeric VAEP feature matrix aligned (positionally) to ``actions``."""
    a = actions.reset_index(drop=True)
    gb = a.groupby("match_id", sort=False)
    away = (a["team_id"] != a["home_team_id"]).fillna(False).to_numpy()

    def shift_num(col: str, s: int) -> np.ndarray:
        if s == 0:
            return a[col].to_numpy(dtype="float64")
        return gb[col].shift(s).fillna(a[col]).to_numpy(dtype="float64")

    def shift_cat(col: str, s: int) -> np.ndarray:
        ser = a[col] if s == 0 else gb[col].shift(s).fillna(a[col])
        return ser.fillna("").astype(object).to_numpy()

    feats: dict[str, np.ndarray] = {}
    times: dict[int, np.ndarray] = {}
    end_x: dict[int, np.ndarray] = {}
    end_y: dict[int, np.ndarray] = {}
    start_x0 = start_y0 = None

    for s in range(nb_prev):
        sx, sy = shift_num("start_x", s), shift_num("start_y", s)
        ex, ey = shift_num("end_x", s), shift_num("end_y", s)
        sx = np.where(away, -sx, sx); sy = np.where(away, -sy, sy)
        ex = np.where(away, -ex, ex); ey = np.where(away, -ey, ey)

        feats[f"start_x_a{s}"] = sx; feats[f"start_y_a{s}"] = sy
        feats[f"end_x_a{s}"] = ex; feats[f"end_y_a{s}"] = ey

        dxs, dys = np.abs(GOAL_X - sx), np.abs(GOAL_Y - sy)
        feats[f"start_dist_goal_a{s}"] = np.hypot(dxs, dys)
        feats[f"start_angle_goal_a{s}"] = np.arctan2(dys, dxs)
        dxe, dye = np.abs(GOAL_X - ex), np.abs(GOAL_Y - ey)
        feats[f"end_dist_goal_a{s}"] = np.hypot(dxe, dye)
        feats[f"end_angle_goal_a{s}"] = np.arctan2(dye, dxe)

        mdx, mdy = ex - sx, ey - sy
        feats[f"dx_a{s}"] = mdx; feats[f"dy_a{s}"] = mdy
        feats[f"movement_a{s}"] = np.hypot(mdx, mdy)

        tn = shift_cat("type_name", s)
        for t in ACTIONTYPES:
            feats[f"type_{_san(t)}_a{s}"] = (tn == t)
        rn = shift_cat("result_name", s)
        for r in RESULTS:
            feats[f"result_{_san(r)}_a{s}"] = (rn == r)
        bp = shift_cat("bodypart_name", s)
        for b in BODYPARTS:
            feats[f"bodypart_{_san(b)}_a{s}"] = (bp == b)

        times[s] = shift_num("time_seconds", s)
        end_x[s], end_y[s] = ex, ey
        if s == 0:
            start_x0, start_y0 = sx, sy

    feats["period_id_a0"] = a["period_id"].to_numpy(dtype="float64")
    feats["time_seconds_a0"] = times[0]
    feats["team_goals_before"] = a["team_goals_before"].to_numpy(dtype="float64")
    feats["opp_goals_before"] = a["opp_goals_before"].to_numpy(dtype="float64")
    feats["goal_diff_before"] = a["goal_diff_before"].to_numpy(dtype="float64")

    for s in range(1, nb_prev):
        feats[f"time_delta_a0a{s}"] = times[0] - times[s]
        sdx, sdy = start_x0 - end_x[s], start_y0 - end_y[s]
        feats[f"space_dx_a0a{s}"] = sdx
        feats[f"space_dy_a0a{s}"] = sdy
        feats[f"space_dist_a0a{s}"] = np.hypot(sdx, sdy)

    X = pd.DataFrame(feats, index=a.index).astype("float32")
    return X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
