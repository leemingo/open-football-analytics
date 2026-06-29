"""Minimal soccer pitch drawing helpers for animation."""

import matplotlib.pyplot as plt
from matplotlib.patches import Arc


SPADL_PITCH_CONFIG = {
    "length": 105.0,
    "width": 68.0,
    "penalty_box_length": 16.5,
    "penalty_box_width": 40.3,
    "six_yard_box_length": 5.5,
    "six_yard_box_width": 18.3,
    "penalty_spot_distance": 11.0,
    "goal_width": 7.3,
    "goal_length": 2.0,
    "origin_x": 0.0,
    "origin_y": 0.0,
    "circle_radius": 9.15,
}

Z_LINE = 0
Z_FIELD = -5000


def _plot_rectangle(x1, y1, x2, y2, ax, color):
    ax.plot([x1, x1], [y1, y2], color=color, zorder=Z_LINE)
    ax.plot([x2, x2], [y1, y2], color=color, zorder=Z_LINE)
    ax.plot([x1, x2], [y1, y1], color=color, zorder=Z_LINE)
    ax.plot([x1, x2], [y2, y2], color=color, zorder=Z_LINE)


def field(color="green", field_length=None, field_width=None, fig=None, ax=None, figsize=None, show=True):
    cfg = SPADL_PITCH_CONFIG.copy()
    if field_length is not None:
        cfg["length"] = float(field_length)
    if field_width is not None:
        cfg["width"] = float(field_width)

    if color == "white":
        return _field(
            fig=fig,
            ax=ax,
            linecolor="black",
            fieldcolor="white",
            alpha=1.0,
            figsize=figsize,
            field_config=cfg,
            show=show,
        )
    if color == "green":
        return _field(
            fig=fig,
            ax=ax,
            linecolor="white",
            fieldcolor="#3e8e41",
            alpha=0.55,
            figsize=figsize,
            field_config=cfg,
            show=show,
        )
    raise ValueError(f"Unsupported pitch color: {color}")


def _field(fig=None, ax=None, linecolor="black", fieldcolor="white", alpha=1.0, figsize=None, field_config=None, show=True):
    cfg = SPADL_PITCH_CONFIG if field_config is None else field_config

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.gca()

    x1 = cfg["origin_x"]
    y1 = cfg["origin_y"]
    x2 = cfg["origin_x"] + cfg["length"]
    y2 = cfg["origin_y"] + cfg["width"]

    ax.set_xlim(x1 - 6, x2 + 6)
    ax.set_ylim(y1 - 4, y2 + 4)

    goal_depth = cfg["goal_length"]
    rectangle = plt.Rectangle(
        (x1 - 2 * goal_depth, y1 - 2 * goal_depth),
        cfg["length"] + 4 * goal_depth,
        cfg["width"] + 4 * goal_depth,
        fc=fieldcolor,
        alpha=alpha,
        zorder=Z_FIELD,
    )
    ax.add_patch(rectangle)

    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)
    ax.plot([(x1 + x2) / 2, (x1 + x2) / 2], [y1, y2], color=linecolor, zorder=Z_LINE)

    mid_y = (cfg["origin_y"] + cfg["width"]) / 2

    _plot_rectangle(
        cfg["origin_x"],
        mid_y - cfg["penalty_box_width"] / 2,
        cfg["origin_x"] + cfg["penalty_box_length"],
        mid_y + cfg["penalty_box_width"] / 2,
        ax=ax,
        color=linecolor,
    )
    _plot_rectangle(
        cfg["origin_x"] + cfg["length"] - cfg["penalty_box_length"],
        mid_y - cfg["penalty_box_width"] / 2,
        cfg["origin_x"] + cfg["length"],
        mid_y + cfg["penalty_box_width"] / 2,
        ax=ax,
        color=linecolor,
    )

    _plot_rectangle(
        cfg["origin_x"],
        mid_y - cfg["six_yard_box_width"] / 2,
        cfg["origin_x"] + cfg["six_yard_box_length"],
        mid_y + cfg["six_yard_box_width"] / 2,
        ax=ax,
        color=linecolor,
    )
    _plot_rectangle(
        cfg["origin_x"] + cfg["length"] - cfg["six_yard_box_length"],
        mid_y - cfg["six_yard_box_width"] / 2,
        cfg["origin_x"] + cfg["length"],
        mid_y + cfg["six_yard_box_width"] / 2,
        ax=ax,
        color=linecolor,
    )

    _plot_rectangle(
        cfg["origin_x"] - cfg["goal_length"],
        mid_y - cfg["goal_width"] / 2,
        cfg["origin_x"],
        mid_y + cfg["goal_width"] / 2,
        ax=ax,
        color=linecolor,
    )
    _plot_rectangle(
        cfg["origin_x"] + cfg["length"],
        mid_y - cfg["goal_width"] / 2,
        cfg["origin_x"] + cfg["length"] + cfg["goal_length"],
        mid_y + cfg["goal_width"] / 2,
        ax=ax,
        color=linecolor,
    )

    center_x = (cfg["origin_x"] + cfg["length"]) / 2
    center_circle = plt.Circle((center_x, mid_y), cfg["circle_radius"], color=linecolor, fill=False, zorder=Z_LINE)
    center_spot = plt.Circle((center_x, mid_y), 0.4, color=linecolor, zorder=Z_LINE)

    left_spot_x = cfg["origin_x"] + cfg["penalty_spot_distance"]
    right_spot_x = cfg["origin_x"] + cfg["length"] - cfg["penalty_spot_distance"]
    left_spot = plt.Circle((left_spot_x, mid_y), 0.4, color=linecolor, zorder=Z_LINE)
    right_spot = plt.Circle((right_spot_x, mid_y), 0.4, color=linecolor, zorder=Z_LINE)

    ax.add_patch(center_circle)
    ax.add_patch(center_spot)
    ax.add_patch(left_spot)
    ax.add_patch(right_spot)

    arc_radius = cfg["circle_radius"] * 2
    left_arc = Arc(
        (left_spot_x, mid_y),
        width=arc_radius,
        height=arc_radius,
        angle=0,
        theta1=307,
        theta2=53,
        color=linecolor,
        zorder=Z_LINE,
    )
    right_arc = Arc(
        (right_spot_x, mid_y),
        width=arc_radius,
        height=arc_radius,
        angle=0,
        theta1=127,
        theta2=233,
        color=linecolor,
        zorder=Z_LINE,
    )
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    plt.axis("off")

    if figsize:
        height, width = fig.get_size_inches()
        fig.set_size_inches(figsize, width / height * figsize, forward=True)

    if show:
        plt.show()

    return fig, ax
