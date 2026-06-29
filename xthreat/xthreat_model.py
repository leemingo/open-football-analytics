"""Center-origin Expected Threat models for SkillCorner actions.

The implementation follows the Karun Singh / socceraction xT formulation, but
keeps the project-wide SkillCorner coordinate convention internally:

    x in [-52.5, 52.5], y in [-34, 34], attacking left-to-right.

Precomputed socceraction/Karun maps can still be used through
``PrecomputedXThreat``. That adapter converts center-origin coordinates to the
map's SPADL-style lookup coordinates only at inference time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd


PITCH_X = 105.0
PITCH_Y = 68.0
DEFAULT_L = 16
DEFAULT_W = 12

MOVE_ACTIONS = {"pass", "carry", "drive", "dribble", "cross"}
SHOT_ACTIONS = {"shot"}


def _as_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def get_cell_indexes_center(
    x: pd.Series,
    y: pd.Series,
    *,
    l: int = DEFAULT_L,
    w: int = DEFAULT_W,
    pitch_x: float = PITCH_X,
    pitch_y: float = PITCH_Y,
) -> tuple[pd.Series, pd.Series]:
    """Map center-origin metric coordinates to grid cell indexes.

    ``xi`` increases from own goal to opponent goal. ``yj`` increases from
    negative y to positive y; plotting helpers use ``origin='lower'`` so the
    matrix row order remains intuitive for our center-origin convention.
    """
    x0 = pd.to_numeric(x, errors="coerce") + pitch_x / 2.0
    y0 = pd.to_numeric(y, errors="coerce") + pitch_y / 2.0
    xi = np.floor(x0 / pitch_x * l)
    yj = np.floor(y0 / pitch_y * w)
    xi = pd.Series(xi, index=x.index).astype("float64").clip(0, l - 1).fillna(0)
    yj = pd.Series(yj, index=y.index).astype("float64").clip(0, w - 1).fillna(0)
    return xi.astype("int64"), yj.astype("int64")


def get_flat_indexes_center(
    x: pd.Series,
    y: pd.Series,
    *,
    l: int = DEFAULT_L,
    w: int = DEFAULT_W,
) -> pd.Series:
    xi, yj = get_cell_indexes_center(x, y, l=l, w=w)
    return yj.mul(l).add(xi)


def _count_locations(
    x: pd.Series,
    y: pd.Series,
    *,
    l: int = DEFAULT_L,
    w: int = DEFAULT_W,
) -> npt.NDArray[np.int_]:
    valid = pd.to_numeric(x, errors="coerce").notna() & pd.to_numeric(y, errors="coerce").notna()
    if not valid.any():
        return np.zeros((w, l), dtype=int)
    flat = get_flat_indexes_center(x[valid], y[valid], l=l, w=w)
    counts = flat.value_counts(sort=False)
    vector = np.zeros(w * l, dtype=int)
    vector[counts.index.to_numpy(dtype=int)] = counts.to_numpy(dtype=int)
    return vector.reshape((w, l))


def _safe_divide(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a_arr = np.asarray(a, dtype="float64")
    b_arr = np.asarray(b, dtype="float64")
    return np.divide(a_arr, b_arr, out=np.zeros_like(a_arr, dtype="float64"), where=b_arr != 0)


def _is_move(actions: pd.DataFrame) -> pd.Series:
    return actions["action_type"].astype("string").str.lower().isin(MOVE_ACTIONS)


def _is_shot(actions: pd.DataFrame) -> pd.Series:
    return actions["action_type"].astype("string").str.lower().isin(SHOT_ACTIONS)


def _is_success(actions: pd.DataFrame) -> pd.Series:
    if "move_success" in actions.columns:
        return actions["move_success"].fillna(False).astype(bool)
    if "outcome_flag" in actions.columns:
        return actions["outcome_flag"].fillna(False).astype(bool)
    return pd.Series(False, index=actions.index, dtype="bool")


def _is_goal(actions: pd.DataFrame) -> pd.Series:
    for column in ["goal", "shot_goal_actual", "shot_goal"]:
        if column in actions.columns:
            return actions[column].fillna(False).astype(bool)
    return pd.Series(False, index=actions.index, dtype="bool")


def scoring_prob(
    actions: pd.DataFrame,
    *,
    l: int = DEFAULT_L,
    w: int = DEFAULT_W,
) -> npt.NDArray[np.float64]:
    """Compute P(goal | shot from cell)."""
    shots = actions[_is_shot(actions)].copy()
    goals = shots[_is_goal(shots)].copy()
    shot_counts = _count_locations(shots["start_x"], shots["start_y"], l=l, w=w)
    goal_counts = _count_locations(goals["start_x"], goals["start_y"], l=l, w=w)
    return _safe_divide(goal_counts, shot_counts)


def action_prob(
    actions: pd.DataFrame,
    *,
    l: int = DEFAULT_L,
    w: int = DEFAULT_W,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return P(shot) and P(move) from each grid cell."""
    moves = actions[_is_move(actions)].copy()
    shots = actions[_is_shot(actions)].copy()
    move_counts = _count_locations(moves["start_x"], moves["start_y"], l=l, w=w)
    shot_counts = _count_locations(shots["start_x"], shots["start_y"], l=l, w=w)
    totals = move_counts + shot_counts
    return _safe_divide(shot_counts, totals), _safe_divide(move_counts, totals)


def move_transition_matrix(
    actions: pd.DataFrame,
    *,
    l: int = DEFAULT_L,
    w: int = DEFAULT_W,
) -> npt.NDArray[np.float64]:
    """Compute successful-move transition probabilities.

    The denominator is all attempted moves from a start cell; the numerator is
    successful moves ending in each target cell. Therefore the row sum is the
    empirical probability that an attempted move from the start cell keeps
    possession.
    """
    moves = actions[_is_move(actions)].copy()
    valid = (
        _as_numeric(moves, "start_x").notna()
        & _as_numeric(moves, "start_y").notna()
        & _as_numeric(moves, "end_x").notna()
        & _as_numeric(moves, "end_y").notna()
    )
    moves = moves[valid].copy()
    if moves.empty:
        return np.zeros((w * l, w * l), dtype="float64")

    start_cell = get_flat_indexes_center(moves["start_x"], moves["start_y"], l=l, w=w)
    start_counts = start_cell.value_counts(sort=False)
    denominator = np.zeros(w * l, dtype="float64")
    denominator[start_counts.index.to_numpy(dtype=int)] = start_counts.to_numpy(dtype="float64")

    successful = moves[_is_success(moves)].copy()
    transition = np.zeros((w * l, w * l), dtype="float64")
    if successful.empty:
        return transition

    start_success = get_flat_indexes_center(successful["start_x"], successful["start_y"], l=l, w=w)
    end_success = get_flat_indexes_center(successful["end_x"], successful["end_y"], l=l, w=w)
    pairs = pd.DataFrame({"start": start_success, "end": end_success})
    pair_counts = pairs.groupby(["start", "end"], observed=True).size()
    for (start, end), count in pair_counts.items():
        denom = denominator[int(start)]
        if denom > 0:
            transition[int(start), int(end)] = float(count) / denom
    return transition


@dataclass
class XThreatDiagnostics:
    n_actions: int
    n_moves: int
    n_successful_moves: int
    n_shots: int
    n_goals: int
    iterations: int
    max_delta: float


class CenterOriginExpectedThreat:
    """Karun/socceraction Expected Threat with center-origin coordinates."""

    def __init__(self, *, l: int = DEFAULT_L, w: int = DEFAULT_W, eps: float = 1e-5, max_iter: int = 200):
        self.l = int(l)
        self.w = int(w)
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.xT = np.zeros((self.w, self.l), dtype="float64")
        self.heatmaps: list[npt.NDArray[np.float64]] = []
        self.scoring_prob_matrix: npt.NDArray[np.float64] | None = None
        self.shot_count_matrix: npt.NDArray[np.int_] | None = None
        self.goal_count_matrix: npt.NDArray[np.int_] | None = None
        self.shot_prob_matrix: npt.NDArray[np.float64] | None = None
        self.move_prob_matrix: npt.NDArray[np.float64] | None = None
        self.transition_matrix: npt.NDArray[np.float64] | None = None
        self.diagnostics: XThreatDiagnostics | None = None

    def fit(self, actions: pd.DataFrame) -> "CenterOriginExpectedThreat":
        actions = actions.copy()
        shots_for_counts = actions[_is_shot(actions)].copy()
        goals_for_counts = shots_for_counts[_is_goal(shots_for_counts)].copy()
        self.shot_count_matrix = _count_locations(shots_for_counts["start_x"], shots_for_counts["start_y"], l=self.l, w=self.w)
        self.goal_count_matrix = _count_locations(goals_for_counts["start_x"], goals_for_counts["start_y"], l=self.l, w=self.w)
        self.scoring_prob_matrix = _safe_divide(self.goal_count_matrix, self.shot_count_matrix)
        self.shot_prob_matrix, self.move_prob_matrix = action_prob(actions, l=self.l, w=self.w)
        self.transition_matrix = move_transition_matrix(actions, l=self.l, w=self.w)
        self.xT = np.zeros((self.w, self.l), dtype="float64")
        self.heatmaps = [self.xT.copy()]

        immediate_goal_payoff = self.scoring_prob_matrix * self.shot_prob_matrix
        flat_move = self.move_prob_matrix.reshape(-1)
        flat_goal = immediate_goal_payoff.reshape(-1)
        flat_xt = self.xT.reshape(-1)
        max_delta = np.inf
        iterations = 0
        while iterations < self.max_iter and max_delta > self.eps:
            move_payoff = self.transition_matrix @ flat_xt
            new_flat_xt = flat_goal + flat_move * move_payoff
            max_delta = float(np.nanmax(np.abs(new_flat_xt - flat_xt)))
            flat_xt = new_flat_xt
            self.xT = flat_xt.reshape((self.w, self.l))
            self.heatmaps.append(self.xT.copy())
            iterations += 1

        move_mask = _is_move(actions)
        shot_mask = _is_shot(actions)
        self.diagnostics = XThreatDiagnostics(
            n_actions=int(len(actions)),
            n_moves=int(move_mask.sum()),
            n_successful_moves=int((_is_success(actions) & move_mask).sum()),
            n_shots=int(shot_mask.sum()),
            n_goals=int((_is_goal(actions) & shot_mask).sum()),
            iterations=iterations,
            max_delta=max_delta,
        )
        return self

    def value_at(self, x: pd.Series, y: pd.Series) -> pd.Series:
        xi, yj = get_cell_indexes_center(x, y, l=self.l, w=self.w)
        values = self.xT[yj.to_numpy(dtype=int), xi.to_numpy(dtype=int)]
        values = pd.Series(values, index=x.index, dtype="float64")
        values[pd.to_numeric(x, errors="coerce").isna() | pd.to_numeric(y, errors="coerce").isna()] = np.nan
        return values

    def rate(
        self,
        actions: pd.DataFrame,
        *,
        unsuccessful: Literal["nan", "zero", "negative_start"] = "nan",
    ) -> pd.DataFrame:
        """Add xT start/end/added columns to actions.

        By default, only successful move actions receive an xT-added value,
        matching socceraction. ``unsuccessful='negative_start'`` is useful for
        a sensitivity view that penalizes failed moves by the threat at the
        starting location.
        """
        out = actions.copy()
        out["custom_xT_start"] = self.value_at(out["start_x"], out["start_y"])
        out["custom_xT_end"] = self.value_at(out["end_x"], out["end_y"])
        move_mask = _is_move(out)
        success_mask = _is_success(out)
        out["custom_xT_added"] = np.nan
        out.loc[move_mask & success_mask, "custom_xT_added"] = (
            out.loc[move_mask & success_mask, "custom_xT_end"]
            - out.loc[move_mask & success_mask, "custom_xT_start"]
        )
        if unsuccessful == "zero":
            out.loc[move_mask & ~success_mask, "custom_xT_added"] = 0.0
        elif unsuccessful == "negative_start":
            out.loc[move_mask & ~success_mask, "custom_xT_added"] = -out.loc[move_mask & ~success_mask, "custom_xT_start"]
        elif unsuccessful != "nan":
            raise ValueError("unsuccessful must be 'nan', 'zero', or 'negative_start'")
        return out

    def save_surface(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.xT.tolist(), handle)
        return path

    @classmethod
    def load_surface(cls, path: str | Path, *, eps: float = 1e-5) -> "CenterOriginExpectedThreat":
        grid = pd.read_json(path).values
        model = cls(l=grid.shape[1], w=grid.shape[0], eps=eps)
        model.xT = np.asarray(grid, dtype="float64")
        return model


class PrecomputedXThreat:
    """Adapter for SPADL/Karun-style precomputed xT maps.

    The precomputed surface is assumed to use 0..105 x 0..68 lookup
    coordinates and row 0 at the top of the pitch, matching socceraction's
    load/rate convention. Input actions remain center-origin.
    """

    def __init__(self, surface: npt.ArrayLike, *, row_zero_top: bool = True):
        self.xT = np.asarray(surface, dtype="float64")
        if self.xT.ndim != 2:
            raise ValueError("Precomputed xT surface must be a 2D matrix")
        self.w, self.l = self.xT.shape
        self.row_zero_top = bool(row_zero_top)

    @classmethod
    def load(cls, path_or_url: str | Path, *, row_zero_top: bool = True) -> "PrecomputedXThreat":
        grid = pd.read_json(path_or_url).values
        return cls(grid, row_zero_top=row_zero_top)

    def _cell_indexes(self, x_center: pd.Series, y_center: pd.Series) -> tuple[pd.Series, pd.Series]:
        x = pd.to_numeric(x_center, errors="coerce") + PITCH_X / 2.0
        y = pd.to_numeric(y_center, errors="coerce") + PITCH_Y / 2.0
        xi = pd.Series(np.floor(x / PITCH_X * self.l), index=x_center.index).clip(0, self.l - 1).fillna(0).astype("int64")
        yj = pd.Series(np.floor(y / PITCH_Y * self.w), index=y_center.index).clip(0, self.w - 1).fillna(0).astype("int64")
        if self.row_zero_top:
            yj = (self.w - 1) - yj
        return xi, yj

    def value_at(self, x_center: pd.Series, y_center: pd.Series) -> pd.Series:
        xi, yj = self._cell_indexes(x_center, y_center)
        values = self.xT[yj.to_numpy(dtype=int), xi.to_numpy(dtype=int)]
        values = pd.Series(values, index=x_center.index, dtype="float64")
        values[pd.to_numeric(x_center, errors="coerce").isna() | pd.to_numeric(y_center, errors="coerce").isna()] = np.nan
        return values

    def rate(
        self,
        actions: pd.DataFrame,
        *,
        unsuccessful: Literal["nan", "zero", "negative_start"] = "nan",
    ) -> pd.DataFrame:
        out = actions.copy()
        out["precomputed_xT_start"] = self.value_at(out["start_x"], out["start_y"])
        out["precomputed_xT_end"] = self.value_at(out["end_x"], out["end_y"])
        move_mask = _is_move(out)
        success_mask = _is_success(out)
        out["precomputed_xT_added"] = np.nan
        out.loc[move_mask & success_mask, "precomputed_xT_added"] = (
            out.loc[move_mask & success_mask, "precomputed_xT_end"]
            - out.loc[move_mask & success_mask, "precomputed_xT_start"]
        )
        if unsuccessful == "zero":
            out.loc[move_mask & ~success_mask, "precomputed_xT_added"] = 0.0
        elif unsuccessful == "negative_start":
            out.loc[move_mask & ~success_mask, "precomputed_xT_added"] = -out.loc[move_mask & ~success_mask, "precomputed_xT_start"]
        elif unsuccessful != "nan":
            raise ValueError("unsuccessful must be 'nan', 'zero', or 'negative_start'")
        return out
