"""Shared plotting helpers: scene drawing, heading arrows, figure saving."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import FancyArrowPatch

from toy_example.config import Config, SystemConfig


def save_fig(fig: plt.Figure, cfg: Config, name: str, dpi: int = 150) -> Path:
    path = cfg.output_dir / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def setup_xy(ax: plt.Axes, sys: SystemConfig) -> None:
    ax.set_xlim(sys.state_lo[0], sys.state_hi[0])
    ax.set_ylim(sys.state_lo[1], sys.state_hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$p_x$ (m)")
    ax.set_ylabel(r"$p_y$ (m)")


def draw_scene(ax: plt.Axes, sys: SystemConfig, fill: bool = True) -> None:
    for o in sys.obstacles:
        ax.add_patch(CirclePatch((o.cx, o.cy), o.radius, edgecolor="tab:red",
                                 facecolor="tab:red" if fill else "none", alpha=0.3 if fill else 0.9, zorder=5))
    ax.plot(*sys.goal, marker="*", color="tab:green", markersize=14, zorder=6)


def heading_arrow(x: float, y: float, theta: float, length: float = 0.75, **kwargs) -> FancyArrowPatch:
    """Arrow from (x, y) along theta (rover-core render style)."""
    end = (x + length * np.cos(theta), y + length * np.sin(theta))
    return FancyArrowPatch((x, y), end, mutation_scale=10, **kwargs)


def theta_label(theta: float) -> str:
    return rf"$\theta$={theta:.2f}"
