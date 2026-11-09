"""(x, y) slices at fixed θ for the failure function, GridInput and GridSet (rover-core visualize_* style)."""

import matplotlib.pyplot as plt
import numpy as np

from toy_example.config import SystemConfig
from toy_example.grids.axes import nearest_axis_indices
from toy_example.grids.grid_input import GridInput
from toy_example.grids.grid_set import GridSet
from toy_example.reachability.grid_value import GridValue
from toy_example.systems.dubins import failure
from toy_example.visualization.common import draw_scene, setup_xy, theta_label

CONTROL_KW = dict(levels=np.linspace(-1, 1, 21), cmap="RdBu", vmin=-1, vmax=1, extend="both")


def _slice(values: np.ndarray, axes: list[np.ndarray], theta: float) -> tuple[np.ndarray, float]:
    k = int(nearest_axis_indices(axes[2], np.array([theta]))[0])
    return values[:, :, k], float(axes[2][k])


def _panel(ax, sys, axes, field, title, **kw):
    cs = ax.contourf(axes[0], axes[1], field.T, **kw)
    draw_scene(ax, sys, fill=False)
    setup_xy(ax, sys)
    ax.set_title(title)
    plt.colorbar(cs, ax=ax, shrink=0.8)


def plot_failure(sys: SystemConfig, n: int = 400) -> plt.Figure:
    x = np.linspace(sys.state_lo[0], sys.state_hi[0], n)
    y = np.linspace(sys.state_lo[1], sys.state_hi[1], n // 2)
    X, Y = np.meshgrid(x, y, indexing="ij")
    sdf = failure(np.stack([X, Y, np.zeros_like(X)], axis=-1), sys)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    lim = np.abs(sdf).max()
    _panel(ax, sys, [x, y], sdf, "Failure ℓ(x) = signed distance", levels=21, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.contour(x, y, sdf.T, levels=[0], colors="k", linewidths=1)
    return fig


def plot_grid_input(sys: SystemConfig, gi: GridInput, thetas=(-np.pi / 2, 0.0, np.pi / 2, np.pi)) -> plt.Figure:
    fig, axs = plt.subplots(len(thetas), 1, figsize=(10, 3.4 * len(thetas)))
    for ax, th in zip(np.atleast_1d(axs), thetas):
        field, th_k = _slice(gi.values, gi.axes, th)
        _panel(ax, sys, gi.axes, field, f"MPC ω | {theta_label(th_k)}", **CONTROL_KW)
    fig.tight_layout()
    return fig


def plot_grid_set(sys: SystemConfig, gs: GridSet, gi: GridInput, theta: float = 0.0) -> plt.Figure:
    lower, th_k = _slice(gs.lower, gs.axes, theta)
    upper, _ = _slice(gs.upper, gs.axes, theta)
    nominal, _ = _slice(gi.values, gi.axes, theta)
    e = sys.error_bound
    fig, axs = plt.subplots(2, 2, figsize=(18, 7))
    _panel(axs[0, 0], sys, gs.axes, lower, "Lower bound ℓ", **CONTROL_KW)
    _panel(axs[0, 1], sys, gs.axes, upper, "Upper bound h", **CONTROL_KW)
    _panel(axs[1, 0], sys, gs.axes, upper - lower, "Width h − ℓ", levels=np.linspace(0, 2, 21), cmap="inferno", vmin=0, vmax=2, extend="max")
    _panel(axs[1, 1], sys, gs.axes, nominal, "Nominal ω", **CONTROL_KW)
    fig.suptitle(f"GridSet | E=±({e[0]:g},{e[1]:g},{e[2]:g}) | {theta_label(th_k)}")
    fig.tight_layout()
    return fig


def _value_slice(gv: GridValue, theta: float, t: float) -> tuple[np.ndarray, float]:
    k = int(nearest_axis_indices(gv.times, np.array([t]))[0])
    return _slice(gv.values[k], gv.axes, theta)


def plot_brt(sys: SystemConfig, nominal: GridValue, robust: GridValue, thetas=(-np.pi / 2, 0.0, np.pi / 2, np.pi), t: float = 0.0) -> plt.Figure:
    """Robust V(t) with the zero level of the nominal (dashed) and robust (solid) BRT."""
    fig, axs = plt.subplots(len(thetas), 1, figsize=(10, 3.4 * len(thetas)))
    for ax, th in zip(np.atleast_1d(axs), thetas):
        v_rob, th_k = _value_slice(robust, th, t)
        v_nom, _ = _value_slice(nominal, th, t)
        lim = np.abs(v_rob).max()
        _panel(ax, sys, robust.axes, v_rob, f"V robust | {theta_label(th_k)} | t={t:g}s", levels=21, cmap="RdBu", vmin=-lim, vmax=lim)
        ax.contour(nominal.axes[0], nominal.axes[1], v_nom.T, levels=[0], colors="k", linestyles="--", linewidths=1.2)
        ax.contour(robust.axes[0], robust.axes[1], v_rob.T, levels=[0], colors="k", linewidths=1.5)
    fig.tight_layout()
    return fig


def plot_brt_tube(sys: SystemConfig, gv: GridValue, title: str, theta: float = 0.0, n_times: int = 6) -> plt.Figure:
    """Zero level of V(t) at several t: the BRT grows as the remaining horizon T - t grows."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    draw_scene(ax, sys, fill=False)
    ts = np.linspace(gv.times[-1], gv.times[0], n_times)
    colors = plt.cm.viridis(np.linspace(0, 1, n_times))
    for t, c in zip(ts, colors):
        v, th_k = _value_slice(gv, theta, t)
        ax.contour(gv.axes[0], gv.axes[1], v.T, levels=[0], colors=[c], linewidths=1.5)
        ax.plot([], [], color=c, label=f"T−t={gv.times[-1] - t:.1f}s")
    setup_xy(ax, sys)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title(f"{title} | {theta_label(th_k)}")
    return fig
