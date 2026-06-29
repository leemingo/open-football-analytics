"""Location-based xG surface plots."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mplsoccer import Pitch

from football_cdf.constants import PITCH_X, PITCH_Y

from xg.xg_features import add_xg_features, get_model_feature_columns


def _template_row(reference_shots: pd.DataFrame) -> pd.Series:
    featured = add_xg_features(reference_shots)
    if featured.empty:
        raise ValueError("reference_shots is empty")

    # Prefer a common open-play, non-header template. Fallback to the median/mode
    # of the full data if the filter is empty.
    template_pool = featured.copy()
    for col, value in {
        "is_header": False,
        "is_set_piece_start": False,
        "start_type": "pass_reception",
    }.items():
        if col in template_pool.columns:
            filtered = template_pool[template_pool[col].eq(value)]
            if not filtered.empty:
                template_pool = filtered

    row = template_pool.iloc[0].copy()
    for col in template_pool.columns:
        series = template_pool[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                row[col] = numeric.median()
        elif pd.api.types.is_bool_dtype(series):
            row[col] = bool(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else False
        else:
            mode = series.astype("string").mode(dropna=True)
            row[col] = mode.iloc[0] if not mode.empty else "missing"

    row["goal"] = False
    row["is_header"] = False
    row["is_set_piece_start"] = False
    row["is_corner_start"] = False
    row["is_free_kick_start"] = False
    row["is_throw_in_start"] = False
    row["is_goal_kick_start"] = False
    row["carry"] = False
    row["one_touch"] = False
    row["quick_pass"] = False
    row["start_type"] = "pass_reception"
    row["team_in_possession_phase_type"] = "finish"
    return row


def build_location_xg_surface(
    model,
    reference_shots: pd.DataFrame,
    *,
    x_min: float = 0.0,
    x_max: float = PITCH_X / 2.0,
    y_min: float = -PITCH_Y / 2.0,
    y_max: float = PITCH_Y / 2.0,
    nx: int = 90,
    ny: int = 70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict xG across a grid of shot locations.

    The surface isolates location by holding non-location features at typical
    open-play, non-header values from ``reference_shots``.
    """
    xs = np.linspace(float(x_min), float(x_max), int(nx))
    ys = np.linspace(float(y_min), float(y_max), int(ny))
    xx, yy = np.meshgrid(xs, ys)

    template = _template_row(reference_shots)
    grid = pd.DataFrame([template.to_dict()] * xx.size)
    grid["shot_x"] = xx.ravel()
    grid["shot_y"] = yy.ravel()
    grid["x_end"] = grid["shot_x"]
    grid["y_end"] = grid["shot_y"]
    grid["possession_start_x"] = grid["shot_x"]
    grid["possession_start_y"] = grid["shot_y"]
    grid["distance_covered"] = 0.0
    grid["duration"] = 0.0

    featured = add_xg_features(grid)
    numeric, binary, categorical = get_model_feature_columns()
    for col in binary:
        featured[col] = featured[col].astype(float)
    for col in categorical:
        featured[col] = featured[col].astype("string").fillna("missing")

    X = featured[numeric + binary + categorical]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pred = model.predict_proba(X)[:, 1]
    return xx, yy, pred.reshape(xx.shape)


def draw_attacking_half_pitch(ax, *, line_color: str = "#25316d") -> None:
    """Draw regulation pitch markings on the attacking half."""
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_X,
        pitch_width=PITCH_Y,
        half=True,
        pitch_color="none",
        line_color=line_color,
        linewidth=2.0,
        line_alpha=0.95,
        line_zorder=3,
        goal_type="box",
        goal_alpha=0.95,
        spot_scale=0.012,
        corner_arcs=True,
    )
    pitch.draw(ax=ax)


def plot_location_xg_surface(
    model,
    reference_shots: pd.DataFrame,
    *,
    ax=None,
    cmap: str = "Reds",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot a location-only xG surface similar to a probability-of-goal map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    xx, yy, zz = build_location_xg_surface(model, reference_shots)
    if vmin is None:
        vmin = float(max(0.0, np.nanpercentile(zz, 1)))
    if vmax is None:
        vmax = float(max(0.10, min(0.35, np.nanpercentile(zz, 99.5))))
    image = ax.imshow(
        zz,
        extent=[PITCH_X / 2.0, PITCH_X, 0, PITCH_Y],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.82,
        aspect="equal",
    )
    draw_attacking_half_pitch(ax)
    ax.set_title("Probability of goal by shot location")
    return ax, image
