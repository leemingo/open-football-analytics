"""Adapters for reviewing synchronized SPADL events against tracking data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from football_cdf.constants import PITCH_X, PITCH_Y

from .animator import Animator, MarkerSpec, VectorSpec, save_animation as _save_animation


def _to_pitch_x(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(values, copy=False).astype(float) + (PITCH_X / 2.0)


def _to_pitch_y(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(values, copy=False).astype(float) + (PITCH_Y / 2.0)


def _safe_delta(x1, y1, x2, y2) -> float:
    values = pd.to_numeric(pd.Series([x1, y1, x2, y2]), errors="coerce")
    if values.isna().any():
        return np.nan
    return float(np.sqrt((values.iloc[0] - values.iloc[2]) ** 2 + (values.iloc[1] - values.iloc[3]) ** 2))


def rank_sync_coordinate_deltas(synced_spadl: pd.DataFrame, *, top_n: int = 20) -> pd.DataFrame:
    ranked = synced_spadl.copy()

    ranked["delta_start"] = np.sqrt(
        (pd.to_numeric(ranked["start_x"], errors="coerce") - pd.to_numeric(ranked["start_x_synced"], errors="coerce")) ** 2
        + (pd.to_numeric(ranked["start_y"], errors="coerce") - pd.to_numeric(ranked["start_y_synced"], errors="coerce")) ** 2
    )
    ranked["delta_end"] = np.sqrt(
        (pd.to_numeric(ranked["end_x"], errors="coerce") - pd.to_numeric(ranked["end_x_synced"], errors="coerce")) ** 2
        + (pd.to_numeric(ranked["end_y"], errors="coerce") - pd.to_numeric(ranked["end_y_synced"], errors="coerce")) ** 2
    )
    ranked["delta_max"] = ranked[["delta_start", "delta_end"]].max(axis=1)

    keep_columns = [
        "match_id",
        "original_event_id",
        "period_id",
        "utc_timestamp",
        "player_id",
        "object_id",
        "receiver_id",
        "receiver_object_id",
        "spadl_type",
        "result_name",
        "tracking_frame_id",
        "tracking_frame_id_end",
        "start_x",
        "start_y",
        "start_x_synced",
        "start_y_synced",
        "end_x",
        "end_y",
        "end_x_synced",
        "end_y_synced",
        "delta_start",
        "delta_end",
        "delta_max",
    ]
    keep_columns = [column for column in keep_columns if column in ranked.columns]

    return ranked.loc[:, keep_columns].sort_values(["delta_max", "delta_end", "delta_start"], ascending=False).head(top_n)


def _prepare_tracking_segment(
    tracking: pd.DataFrame,
    *,
    period_id: str,
    start_frame: int,
    end_frame: int,
) -> pd.DataFrame:
    tracking_segment = tracking[
        (tracking["period"] == period_id)
        & (tracking["frame_id"] >= start_frame)
        & (tracking["frame_id"] <= end_frame)
    ].copy()

    if tracking_segment.empty:
        raise ValueError("No tracking rows found for the requested review window.")

    tracking_segment["display_x"] = _to_pitch_x(tracking_segment["x"]).round(2)
    tracking_segment["display_y"] = _to_pitch_y(tracking_segment["y"]).round(2)

    frames = (
        tracking_segment[["frame_id", "timestamp", "utc_timestamp", "period", "match_id"]]
        .drop_duplicates("frame_id")
        .sort_values("frame_id")
        .reset_index(drop=True)
    )

    players = tracking_segment[~tracking_segment["ball"]].copy()
    players_wide = (
        players.pivot_table(index="frame_id", columns="object_id", values=["display_x", "display_y"], aggfunc="first")
        .sort_index(axis=1)
    )
    players_wide.columns = [f"{object_id}_{'x' if coord == 'display_x' else 'y'}" for coord, object_id in players_wide.columns]
    players_wide = players_wide.reset_index()

    ball = tracking_segment[tracking_segment["ball"]][["frame_id", "display_x", "display_y"]].drop_duplicates("frame_id")
    ball = ball.rename(columns={"display_x": "ball_x", "display_y": "ball_y"})

    segment = frames.merge(players_wide, on="frame_id", how="left").merge(ball, on="frame_id", how="left")
    return segment.sort_values("frame_id").reset_index(drop=True)


def prepare_event_animation_data(
    synced_spadl: pd.DataFrame,
    tracking: pd.DataFrame,
    *,
    event_index: int,
    pre_frames: int = 40,
    post_frames: int = 60,
) -> pd.DataFrame:
    if event_index not in synced_spadl.index:
        raise KeyError(f"Event index {event_index} not found in synced_spadl.")

    event = synced_spadl.loc[event_index].copy()
    if pd.isna(event.get("tracking_frame_id")):
        raise ValueError("Selected event has no synchronized tracking_frame_id.")

    start_frame = int(event["tracking_frame_id"])
    end_frame = int(event["tracking_frame_id_end"]) if pd.notna(event.get("tracking_frame_id_end")) else start_frame
    frame_from = max(min(start_frame, end_frame) - int(pre_frames), 0)
    frame_to = max(start_frame, end_frame) + int(post_frames)

    segment = _prepare_tracking_segment(
        tracking,
        period_id=str(event["period_id"]),
        start_frame=frame_from,
        end_frame=frame_to,
    )

    original_start_x = float(_to_pitch_x([event["start_x"]]).iloc[0]) if pd.notna(event.get("start_x")) else np.nan
    original_start_y = float(_to_pitch_y([event["start_y"]]).iloc[0]) if pd.notna(event.get("start_y")) else np.nan
    synced_start_x = float(_to_pitch_x([event["start_x_synced"]]).iloc[0]) if pd.notna(event.get("start_x_synced")) else np.nan
    synced_start_y = float(_to_pitch_y([event["start_y_synced"]]).iloc[0]) if pd.notna(event.get("start_y_synced")) else np.nan
    original_end_x = float(_to_pitch_x([event["end_x"]]).iloc[0]) if pd.notna(event.get("end_x")) else np.nan
    original_end_y = float(_to_pitch_y([event["end_y"]]).iloc[0]) if pd.notna(event.get("end_y")) else np.nan
    synced_end_x = float(_to_pitch_x([event["end_x_synced"]]).iloc[0]) if pd.notna(event.get("end_x_synced")) else np.nan
    synced_end_y = float(_to_pitch_y([event["end_y_synced"]]).iloc[0]) if pd.notna(event.get("end_y_synced")) else np.nan

    segment["event_summary"] = (
        f"{event.get('spadl_type', '')} | {event.get('object_id', '')} -> {event.get('receiver_object_id', '')} | "
        f"{event.get('result_name', '')}"
    )

    delta_start = _safe_delta(
        event.get("start_x", np.nan),
        event.get("start_y", np.nan),
        event.get("start_x_synced", np.nan),
        event.get("start_y_synced", np.nan),
    )
    delta_end = _safe_delta(
        event.get("end_x", np.nan),
        event.get("end_y", np.nan),
        event.get("end_x_synced", np.nan),
        event.get("end_y_synced", np.nan),
    )

    segment["event_id_text"] = f"event_id={event.get('original_event_id', pd.NA)}"
    segment["frame_text"] = f"start_frame={start_frame}, end_frame={end_frame}"
    segment["delta_text"] = f"delta_start={delta_start:.2f}m, delta_end={delta_end:.2f}m"
    segment["receiver_text"] = (
        f"receiver(orig)={event.get('receiver_object_id', pd.NA)}, "
        f"receiver(sync)={event.get('receiver_object_id_synced', pd.NA)}"
    )

    for column, value in {
        "orig_start_x": original_start_x,
        "orig_start_y": original_start_y,
        "sync_start_x": synced_start_x,
        "sync_start_y": synced_start_y,
        "orig_end_x": original_end_x,
        "orig_end_y": original_end_y,
        "sync_end_x": synced_end_x,
        "sync_end_y": synced_end_y,
    }.items():
        segment[column] = value

    return segment


def build_event_animation(
    synced_spadl: pd.DataFrame,
    tracking: pd.DataFrame,
    *,
    event_index: int,
    pre_frames: int = 40,
    post_frames: int = 60,
    fps: int = 10,
    play_speed: int = 1,
    anonymize: bool = False,
):
    segment = prepare_event_animation_data(
        synced_spadl,
        tracking,
        event_index=event_index,
        pre_frames=pre_frames,
        post_frames=post_frames,
    )

    marker_specs = [
        MarkerSpec("orig_start_x", "orig_start_y", color="black", marker="X", label="original start", size=170),
        MarkerSpec("sync_start_x", "sync_start_y", color="orange", marker="*", label="synced start", size=200, edgecolor="black"),
        MarkerSpec("orig_end_x", "orig_end_y", color="black", marker="s", label="original end", size=120),
        MarkerSpec("sync_end_x", "sync_end_y", color="deepskyblue", marker="D", label="synced end", size=120, edgecolor="black"),
    ]
    vector_specs = [
        VectorSpec("orig_start_x", "orig_start_y", "orig_end_x", "orig_end_y", color="black", label="original vector"),
        VectorSpec("sync_start_x", "sync_start_y", "sync_end_x", "sync_end_y", color="orange", label="synced vector", linestyle="-"),
    ]

    animator = Animator(
        segment,
        marker_specs=marker_specs,
        vector_specs=vector_specs,
        show_times=True,
        show_event_text=True,
        text_cols=["event_id_text", "frame_text", "delta_text", "receiver_text"],
        anonymize=anonymize,
        play_speed=play_speed,
    )
    return animator.run(fps=fps)


def save_event_animation(
    synced_spadl: pd.DataFrame,
    tracking: pd.DataFrame,
    path: str | Path,
    *,
    event_index: int,
    pre_frames: int = 40,
    post_frames: int = 60,
    fps: int = 10,
    play_speed: int = 1,
    anonymize: bool = False,
):
    anim = build_event_animation(
        synced_spadl,
        tracking,
        event_index=event_index,
        pre_frames=pre_frames,
        post_frames=post_frames,
        fps=fps,
        play_speed=play_speed,
        anonymize=anonymize,
    )
    return _save_animation(anim, path, fps=fps)


def save_animation(anim, path: str | Path, *, fps: int = 10):
    return _save_animation(anim, path, fps=fps)
