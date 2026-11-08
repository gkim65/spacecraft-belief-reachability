"""Simulation figure: true and perceived trajectories with heading triangles."""

import matplotlib.pyplot as plt
import numpy as np

from toy_example.config import SystemConfig
from toy_example.simulation.simulator import Rollout
from toy_example.visualization.common import draw_scene, heading_triangle, setup_xy


def rollout_title(controller: str, player: str, sys: SystemConfig, ro: Rollout) -> str:
    e = sys.error_bound
    error = "perfect estimate" if player == "zero" else f"{player} E=±({e[0]:g},{e[1]:g},{e[2]:g})"
    n_hit = int(ro.collided.sum())
    outcome = "all safe" if n_hit == 0 else f"{n_hit}/{len(ro.collided)} collide"
    if len(ro.collided) == 1 and n_hit:
        outcome = f"collision t={ro.collision_time[0]:.2f}s"
    return f"{controller} | {error} | T={ro.times[-1]:g}s | {outcome}"


def plot_rollout(sys: SystemConfig, ro: Rollout, title: str, show_estimate: bool = True, step: int | None = None) -> plt.Figure:
    """Plot trajectories up to `step` (default: the end), with triangles at the current true/perceived state."""
    k = ro.states.shape[1] - 1 if step is None else step
    colors = plt.cm.tab10(np.arange(len(ro.states)) % 10)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    draw_scene(ax, sys)
    for b, c in enumerate(colors):
        xs = ro.states[b, : k + 1]
        ax.plot(xs[:, 0], xs[:, 1], "-", lw=2, color=c, label="true" if b == 0 else None)
        ax.add_patch(heading_triangle(*xs[-1], color=c, zorder=10))
        if show_estimate and k > 0:
            es = ro.estimates[b, :k]
            ax.plot(es[:, 0], es[:, 1], "--", lw=1.2, color="gray", alpha=0.9, label="perceived" if b == 0 else None)
            ax.add_patch(heading_triangle(*es[-1], facecolor="none", edgecolor="gray", lw=1.2, zorder=9))
        if ro.collided[b]:
            hit = int(np.argmax(ro.failure[b] < 0))
            if hit <= k:
                ax.plot(*ro.states[b, hit, :2], "x", color="k", ms=9, mew=2, zorder=11)
    setup_xy(ax, sys)
    ax.grid(True, zorder=0)
    ax.legend(loc="upper right")
    ax.set_title(title)
    return fig
