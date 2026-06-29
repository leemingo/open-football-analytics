"""Animation utilities for synchronized event and tracking review."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, axes, collections, lines, text

from .matplotsoccer import field


ANIM_CONFIG = {
    "figsize": (10.8, 7.2),
    "fontsize": 13,
    "player_size": 380,
    "ball_size": 110,
    "player_history": 18,
    "ball_history": 35,
}


@dataclass(frozen=True)
class MarkerSpec:
    x_col: str
    y_col: str
    color: str
    marker: str
    label: str
    size: int = 140
    edgecolor: Optional[str] = "black"
    alpha: float = 1.0
    zorder: int = 100


@dataclass(frozen=True)
class VectorSpec:
    start_x_col: str
    start_y_col: str
    end_x_col: str
    end_y_col: str
    color: str
    label: str
    linestyle: str = "--"
    linewidth: float = 2.0
    alpha: float = 0.85
    zorder: int = 80


class Animator:
    def __init__(
        self,
        trace_df: pd.DataFrame,
        *,
        marker_specs: Optional[Iterable[MarkerSpec]] = None,
        vector_specs: Optional[Iterable[VectorSpec]] = None,
        show_times: bool = True,
        show_event_text: bool = True,
        text_cols: Optional[Iterable[str]] = None,
        anonymize: bool = False,
        play_speed: int = 1,
    ):
        self.trace_df = trace_df.copy().reset_index(drop=True)
        self.marker_specs = list(marker_specs or [])
        self.vector_specs = list(vector_specs or [])
        self.show_times = show_times
        self.show_event_text = show_event_text
        self.text_cols = list(text_cols or [])
        self.anonymize = anonymize
        self.play_speed = max(int(play_speed), 1)
        self.pitch_size = (105.0, 68.0)
        self.arg_dict: dict[str, object] = {}

    @staticmethod
    def plot_players(traces: pd.DataFrame, ax: axes.Axes, sizes=ANIM_CONFIG["player_size"], alpha=1.0, anonymize=False):
        if traces.empty or len(traces.columns) == 0:
            return None

        x_cols = [column for column in traces.columns if column.endswith("_x")]
        players = [column[:-2] for column in x_cols if f"{column[:-2]}_y" in traces.columns]
        x_cols = [f"{player}_x" for player in players]
        y_cols = [f"{player}_y" for player in players]

        if not players:
            return None

        color = "tab:red" if players[0].startswith("home_") else "tab:blue"
        x = traces[x_cols].to_numpy(dtype=float)
        y = traces[y_cols].to_numpy(dtype=float)
        size = sizes[0, 0] if isinstance(sizes, np.ndarray) else sizes
        scatter = ax.scatter(x[0], y[0], s=size, c=color, alpha=alpha, zorder=2)

        player_lookup = dict(zip(players, np.arange(len(players)) + 1))
        plots: Dict[str, lines.Line2D] = {}
        annotations: Dict[str, text.Annotation] = {}

        for player in players:
            (plots[player],) = ax.plot([], [], c=color, alpha=alpha, ls=":", zorder=1)
            player_label = player_lookup[player] if anonymize else int(player.split("_")[-1])
            annotation = ax.annotate(
                player_label,
                xy=traces.loc[0, [f"{player}_x", f"{player}_y"]],
                ha="center",
                va="center",
                color="white",
                fontsize=ANIM_CONFIG["fontsize"] - 2,
                fontweight="bold",
                annotation_clip=False,
                zorder=3,
            )
            annotation.set_animated(True)
            annotations[player] = annotation

        return traces, sizes, scatter, plots, annotations

    @staticmethod
    def animate_players(
        t: int,
        inplay_records: pd.DataFrame,
        traces: pd.DataFrame,
        sizes: np.ndarray,
        scatter: collections.PathCollection,
        plots: Dict[str, lines.Line2D],
        annotations: Dict[str, text.Annotation],
    ):
        x_cols = [f"{player}_x" for player in plots if f"{player}_x" in traces.columns]
        y_cols = [f"{player}_y" for player in plots if f"{player}_y" in traces.columns]
        x = traces[x_cols].to_numpy(dtype=float)
        y = traces[y_cols].to_numpy(dtype=float)
        scatter.set_offsets(np.stack([x[t], y[t]]).T)

        if isinstance(sizes, np.ndarray):
            scatter.set_sizes(sizes[t])

        for player in plots:
            start_index = inplay_records.at[player, "start_index"]
            end_index = inplay_records.at[player, "end_index"]
            if t >= start_index:
                if t <= end_index:
                    t_from = max(t - ANIM_CONFIG["player_history"] + 1, start_index)
                    plots[player].set_data(traces.loc[t_from:t, f"{player}_x"], traces.loc[t_from:t, f"{player}_y"])
                    annotations[player].set_position(traces.loc[t, [f"{player}_x", f"{player}_y"]].to_numpy(dtype=float))
                elif t == end_index + 1:
                    plots[player].set_alpha(0.0)
                    annotations[player].set_alpha(0.0)

    @staticmethod
    def plot_ball(xy: pd.DataFrame, ax: axes.Axes, color="white", edgecolor="black", marker="o", show_path=True):
        x = xy.iloc[:, 0].to_numpy(dtype=float)
        y = xy.iloc[:, 1].to_numpy(dtype=float)
        scatter = ax.scatter(x[0], y[0], s=ANIM_CONFIG["ball_size"], c=color, edgecolors=edgecolor, marker=marker, zorder=4)
        plot = None
        if show_path:
            (plot,) = ax.plot([], [], color="black", zorder=3)
        return x, y, scatter, plot

    @staticmethod
    def animate_ball(t: int, x: np.ndarray, y: np.ndarray, scatter: collections.PathCollection, plot: Optional[lines.Line2D] = None):
        scatter.set_offsets(np.array([x[t], y[t]], dtype=float))
        if plot is not None:
            t_from = max(t - ANIM_CONFIG["ball_history"], 0)
            plot.set_data(x[t_from : t + 1], y[t_from : t + 1])

    @staticmethod
    def plot_marker(traces: pd.DataFrame, spec: MarkerSpec, ax: axes.Axes):
        x = traces[spec.x_col].to_numpy(dtype=float)
        y = traces[spec.y_col].to_numpy(dtype=float)
        scatter = ax.scatter(
            x[0],
            y[0],
            s=spec.size,
            c=spec.color,
            edgecolors=spec.edgecolor,
            marker=spec.marker,
            alpha=spec.alpha,
            zorder=spec.zorder,
            label=spec.label,
        )
        return x, y, scatter

    @staticmethod
    def animate_marker(t: int, x: np.ndarray, y: np.ndarray, scatter: collections.PathCollection):
        scatter.set_offsets(np.array([x[t], y[t]], dtype=float))

    @staticmethod
    def plot_vector(traces: pd.DataFrame, spec: VectorSpec, ax: axes.Axes):
        start_x = traces[spec.start_x_col].to_numpy(dtype=float)
        start_y = traces[spec.start_y_col].to_numpy(dtype=float)
        end_x = traces[spec.end_x_col].to_numpy(dtype=float)
        end_y = traces[spec.end_y_col].to_numpy(dtype=float)
        (line,) = ax.plot(
            [start_x[0], end_x[0]],
            [start_y[0], end_y[0]],
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            alpha=spec.alpha,
            zorder=spec.zorder,
            label=spec.label,
        )
        return start_x, start_y, end_x, end_y, line

    @staticmethod
    def animate_vector(t: int, start_x: np.ndarray, start_y: np.ndarray, end_x: np.ndarray, end_y: np.ndarray, line: lines.Line2D):
        line.set_data([start_x[t], end_x[t]], [start_y[t], end_y[t]])

    def _prepare_trace(self) -> pd.DataFrame:
        traces = self.trace_df.iloc[:: self.play_speed].copy()
        traces = traces.reset_index(drop=True)
        return traces

    def _plot_init(self, ax: axes.Axes, traces: pd.DataFrame):
        xy_cols = [column for column in traces.columns if column.endswith("_x") or column.endswith("_y")]
        player_xy_cols = [
            column for column in xy_cols
            if column.startswith("home_") or column.startswith("away_")
        ]
        player_objects = sorted({column[:-2] for column in player_xy_cols if column.endswith("_x")})

        inplay_records = []
        for player in player_objects:
            x_col = f"{player}_x"
            if x_col not in traces.columns:
                continue
            inplay_index = traces[traces[x_col].notna()].index
            if len(inplay_index) > 0:
                inplay_records.append([player, int(inplay_index[0]), int(inplay_index[-1])])
        inplay_records = pd.DataFrame(inplay_records, columns=["object", "start_index", "end_index"]).set_index("object")

        home_players = sorted([player for player in player_objects if player.startswith("home_")])
        away_players = sorted([player for player in player_objects if player.startswith("away_")])

        home_cols = [coord for player in home_players for coord in (f"{player}_x", f"{player}_y") if coord in traces.columns]
        away_cols = [coord for player in away_players for coord in (f"{player}_x", f"{player}_y") if coord in traces.columns]

        home_traces = traces[home_cols].fillna(-100.0)
        away_traces = traces[away_cols].fillna(-100.0)

        home_args = self.plot_players(home_traces, ax, alpha=1.0, anonymize=self.anonymize)
        away_args = self.plot_players(away_traces, ax, alpha=1.0, anonymize=self.anonymize)

        ball_args = None
        if {"ball_x", "ball_y"}.issubset(traces.columns):
            ball_xy = traces[["ball_x", "ball_y"]].copy()
            ball_args = self.plot_ball(ball_xy, ax)

        marker_args = []
        for spec in self.marker_specs:
            if spec.x_col in traces.columns and spec.y_col in traces.columns:
                marker_args.append((spec, *self.plot_marker(traces, spec, ax)))

        vector_args = []
        for spec in self.vector_specs:
            required = {spec.start_x_col, spec.start_y_col, spec.end_x_col, spec.end_y_col}
            if required.issubset(traces.columns):
                vector_args.append((spec, *self.plot_vector(traces, spec, ax)))

        self.arg_dict = {
            "traces": traces,
            "inplay_records": inplay_records,
            "home": home_args,
            "away": away_args,
            "ball": ball_args,
            "markers": marker_args,
            "vectors": vector_args,
        }

    def run(self, *, fps: int = 10, max_frames: Optional[int] = None):
        fig, ax = plt.subplots(figsize=ANIM_CONFIG["figsize"])
        field("green", self.pitch_size[0], self.pitch_size[1], fig, ax, show=False)

        traces = self._prepare_trace()
        self._plot_init(ax, traces)

        text_y = self.pitch_size[1] + 1

        if self.show_times:
            timestamps = traces["timestamp"]
            if pd.api.types.is_timedelta64_dtype(timestamps):
                seconds = timestamps.dt.total_seconds()
            else:
                seconds = timestamps.astype(float)
            timestamp_labels = seconds.apply(lambda value: f"{int(value // 60):02d}:{value % 60:05.2f}").to_numpy()
            time_text = ax.text(0, text_y, timestamp_labels[0], fontsize=ANIM_CONFIG["fontsize"], ha="left", va="bottom")
            time_text.set_animated(True)
        else:
            timestamp_labels = None
            time_text = None

        if self.show_event_text and "event_summary" in traces.columns:
            event_labels = traces["event_summary"].fillna("").astype(str).to_numpy()
            event_text = ax.text(
                self.pitch_size[0] / 2,
                text_y,
                event_labels[0],
                fontsize=ANIM_CONFIG["fontsize"],
                ha="center",
                va="bottom",
            )
            event_text.set_animated(True)
        else:
            event_labels = None
            event_text = None

        extra_text = {}
        for idx, column in enumerate(self.text_cols):
            if column not in traces.columns:
                continue
            values = traces[column].fillna("").astype(str).to_numpy()
            text_artist = ax.text(
                0,
                -2.5 - (idx * 2.0),
                values[0],
                fontsize=ANIM_CONFIG["fontsize"] - 1,
                ha="left",
                va="top",
            )
            text_artist.set_animated(True)
            extra_text[column] = (values, text_artist)

        if self.marker_specs or self.vector_specs:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=ANIM_CONFIG["fontsize"] - 2)

        def animate(t: int):
            inplay_records = self.arg_dict["inplay_records"]

            home_args = self.arg_dict["home"]
            away_args = self.arg_dict["away"]
            ball_args = self.arg_dict["ball"]

            if home_args is not None:
                self.animate_players(t, inplay_records, *home_args)
            if away_args is not None:
                self.animate_players(t, inplay_records, *away_args)
            if ball_args is not None:
                self.animate_ball(t, *ball_args)

            for _spec, marker_x, marker_y, marker_scatter in self.arg_dict["markers"]:
                self.animate_marker(t, marker_x, marker_y, marker_scatter)

            for _spec, start_x, start_y, end_x, end_y, line in self.arg_dict["vectors"]:
                self.animate_vector(t, start_x, start_y, end_x, end_y, line)

            if time_text is not None and timestamp_labels is not None:
                time_text.set_text(timestamp_labels[t])

            if event_text is not None and event_labels is not None:
                event_text.set_text(event_labels[t])

            for values, text_artist in extra_text.values():
                text_artist.set_text(values[t])

        frames = traces.shape[0] if max_frames is None else min(max_frames, traces.shape[0])
        animation_obj = animation.FuncAnimation(fig, animate, frames=frames, interval=1000 / fps)
        plt.close(fig)
        return animation_obj


def save_animation(anim: animation.FuncAnimation, path: str | Path, *, fps: int = 10):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(output_path.as_posix(), writer=writer)
    return output_path
