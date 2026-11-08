"""Shared plotting helpers: scene drawing, heading triangles, figure saving."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Polygon

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
    ax.plot(*sys.goal, marker="*", color="tab:green", markersize=14, zorder=6, label="goal")


def heading_triangle(x: float, y: float, theta: float, size: float = 0.5, **kwargs) -> Polygon:
    """Isosceles triangle centred at (x, y) pointing along theta."""
    local = size * np.array([[1.0, 0.0], [-0.6, 0.5], [-0.6, -0.5]])
    c, s = np.cos(theta), np.sin(theta)
    pts = local @ np.array([[c, s], [-s, c]]) + [x, y]
    return Polygon(pts, closed=True, **kwargs)


def theta_label(theta: float) -> str:
    return rf"$\theta$={theta:.2f}"
