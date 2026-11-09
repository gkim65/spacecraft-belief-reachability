"""Simulation figure (rover-core render style): true and perceived trajectories with heading arrows."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from toy_example.config import SystemConfig
from toy_example.grids.axes import nearest_axis_indices
from toy_example.reachability.grid_value import GridValue
from toy_example.simulation.simulator import Rollout
from toy_example.visualization.common import draw_scene, heading_arrow, setup_xy, theta_label

BRT_COLOR = "tab:purple"
ROLLOUT_COLOR = "#404040"
COLLIDED_COLOR = "tab:red"
PERCEIVED_COLOR = "silver"


def rollout_title(controller: str, player: str, sys: SystemConfig, ro: Rollout) -> str:
    e = sys.error_bound
    error = "perfect estimate" if player == "zero" else f"{player} E=±({e[0]:g},{e[1]:g},{e[2]:g})"
    n_hit = int(ro.collided.sum())
    outcome = "all safe" if n_hit == 0 else f"{n_hit}/{len(ro.collided)} collide"
    if len(ro.collided) == 1 and n_hit:
        outcome = f"collision t={ro.collision_time[0]:.2f}s"
    return f"{controller} | {error} | T={ro.times[-1]:g}s | {outcome}"


def draw_brt(ax: plt.Axes, gv: GridValue, theta: float) -> Patch:
    """Fill the BRT {V(0, x) < 0} at the grid θ nearest `theta`; returns a legend handle."""
    k = int(nearest_axis_indices(gv.axes[2], np.array([theta]))[0])
    v = gv.values[0][:, :, k].T
    ax.contourf(gv.axes[0], gv.axes[1], v, levels=[v.min() - 1, 0], colors=[BRT_COLOR], alpha=0.2, zorder=1)
    ax.contour(gv.axes[0], gv.axes[1], v, levels=[0], colors=[BRT_COLOR], linewidths=1, zorder=1)
    return Patch(facecolor=BRT_COLOR, edgecolor=BRT_COLOR, alpha=0.4, label=f"BRT t=0, {theta_label(gv.axes[2][k])}")


def plot_rollout(sys: SystemConfig, ro: Rollout, title: str, show_estimate: bool = True, step: int | None = None,
                 brt: GridValue | None = None) -> plt.Figure:
    """Plot trajectories up to `step` (default: the end), stopping each at its first collision.

    Rollouts are dark gray, red if they collide; the perceived state is light gray dotted.
    With `brt`, the t=0 BRT (sliced at the first rollout's initial heading) is overlaid and
    initial points with V(0, x0) < 0 are marked red.
    """
    last = ro.states.shape[1] - 1 if step is None else step
    fig, ax = plt.subplots(figsize=(12, 4.5))
    draw_scene(ax, sys)
    x0 = ro.states[:, 0]
    unsafe_start = brt.value(x0, 0.0) < 0 if brt is not None else np.zeros(len(x0), dtype=bool)
    handles = [draw_brt(ax, brt, x0[0, 2])] if brt is not None else []
    for b in range(len(ro.states)):
        hit = int(np.argmax(ro.failure[b] < 0)) if ro.collided[b] else None
        k = last if hit is None else min(last, hit)
        c = COLLIDED_COLOR if hit is not None and hit <= last else ROLLOUT_COLOR
        xs = ro.states[b, : k + 1]
        ax.plot(xs[:, 0], xs[:, 1], "-", lw=2, color=c)
        ax.plot(*xs[-1, :2], "o", color=c, ms=4, zorder=10)
        ax.add_patch(heading_arrow(*xs[-1], color=c, zorder=10))
        if show_estimate and k > 0:
            es = ro.estimates[b, :k]
            ax.plot(es[:, 0], es[:, 1], ":", lw=1.5, color=PERCEIVED_COLOR)
            ax.add_patch(heading_arrow(*es[-1], length=0.6, color=PERCEIVED_COLOR, zorder=9))
        if c == COLLIDED_COLOR:
            ax.plot(*ro.states[b, hit, :2], "x", color=COLLIDED_COLOR, ms=10, mew=2.5, zorder=11)
    start_colors = np.where(unsafe_start, COLLIDED_COLOR, ROLLOUT_COLOR)
    ax.scatter(x0[:, 0], x0[:, 1], c=start_colors, s=40, edgecolors="k", linewidths=0.8, zorder=12)

    handles.append(Line2D([], [], color=ROLLOUT_COLOR, lw=2, label="rollout"))
    if ro.collided.any():
        handles.append(Line2D([], [], color=COLLIDED_COLOR, lw=2, label="collided"))
    if show_estimate:
        handles.append(Line2D([], [], color=PERCEIVED_COLOR, ls=":", lw=1.5, label="perceived"))
    if brt is not None:
        handles.append(Line2D([], [], marker="o", ls="", color=COLLIDED_COLOR, mec="k", label="start, V<0"))
    setup_xy(ax, sys)
    ax.grid(True, zorder=0)
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(title)
    return fig
