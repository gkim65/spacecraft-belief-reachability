"""GridSet: per-node control bounds U(x) = [lower, upper] over the error box (rover-core scripts/grid_set).

For each node x, the bounds are the min/max of GridInput over the nodes inside x + E,
and corner_idx stores the flat index of the node attaining each bound (0: lower, 1: upper).
The x/y neighbor ranges follow rover-core (`_compute_neighbor_ranges_np`); θ ranges wrap
across ±π, because the simulator wraps the estimate. Ties pick the first node in C order,
as rover-core's argmin does.

The box min/max is computed axis by axis (θ, then y, then x), which equals the
per-node box reduction and keeps rover-core's C-order tie-breaking.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toy_example import cache
from toy_example.config import Config
from toy_example.grids.axes import nearest_indices
from toy_example.grids.grid_input import GridInput
from toy_example.systems.dubins import error_limits, wrap_angle

PERIODIC = (False, False, True)


@dataclass
class GridSet:
    lower: np.ndarray  # (n0, n1, n2)
    upper: np.ndarray
    corner_idx: np.ndarray  # (n0, n1, n2, 2) flat node indices attaining lower / upper
    axes: list[np.ndarray]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.lower.shape

    def bounds(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = nearest_indices(self.axes, states)
        return self.lower[i[..., 0], i[..., 1], i[..., 2]], self.upper[i[..., 0], i[..., 1], i[..., 2]]

    def extremizer_state(self, states: np.ndarray, use_upper: np.ndarray) -> np.ndarray:
        """Grid node whose control attains the chosen bound at the node nearest each state."""
        i = nearest_indices(self.axes, states)
        flat = self.corner_idx[i[..., 0], i[..., 1], i[..., 2], np.asarray(use_upper, dtype=int)]
        j = np.unravel_index(flat, self.shape)
        return np.stack([self.axes[d][j[d]] for d in range(3)], axis=-1)


def neighbor_table(axis: np.ndarray, half_width: float, periodic: bool) -> np.ndarray:
    """Row i lists the indices of nodes within half_width of node i (padded with -1)."""
    if periodic:
        offset = wrap_angle(axis[None, :] - axis[:, None])
        inside = np.abs(offset) <= half_width
        order = np.argsort(np.where(inside, offset, np.inf), axis=1, kind="stable")
    else:
        lo = np.clip(np.searchsorted(axis, axis - half_width, side="left"), 0, axis.size - 1)
        hi = np.clip(np.searchsorted(axis, axis + half_width, side="right") - 1, lo, axis.size - 1)
        j = np.arange(axis.size)
        inside = (j[None, :] >= lo[:, None]) & (j[None, :] <= hi[:, None])
        order = np.argsort(~inside, axis=1, kind="stable")
    count = inside.sum(axis=1)
    table = np.where(np.arange(axis.size)[None, :] < count[:, None], order, -1)
    return table[:, : count.max()]


def _reduce_axis(vals, idx, table, axis, is_min):
    best_v = np.take(vals, table[:, 0], axis=axis)
    best_i = np.take(idx, table[:, 0], axis=axis)
    for s in range(1, table.shape[1]):
        j = table[:, s]
        valid = np.expand_dims(j >= 0, tuple(a for a in range(vals.ndim) if a != axis))
        cand_v = np.take(vals, np.maximum(j, 0), axis=axis)
        cand_i = np.take(idx, np.maximum(j, 0), axis=axis)
        better = valid & ((cand_v < best_v) if is_min else (cand_v > best_v))
        best_v = np.where(better, cand_v, best_v)
        best_i = np.where(better, cand_i, best_i)
    return best_v, best_i


def _box_extremum(values, tables, is_min):
    vals = values
    idx = np.arange(values.size, dtype=np.int64).reshape(values.shape)
    for axis in (2, 1, 0):
        vals, idx = _reduce_axis(vals, idx, tables[axis], axis, is_min)
    return vals, idx


def path(cfg: Config) -> Path:
    return cfg.output_dir / "grid_set.npz"


def build(cfg: Config, grid_input: GridInput) -> GridSet:
    _, e_hi = error_limits(cfg.system)
    tables = [neighbor_table(ax, e_hi[d], PERIODIC[d]) for d, ax in enumerate(grid_input.axes)]
    lower, lo_idx = _box_extremum(grid_input.values, tables, is_min=True)
    upper, hi_idx = _box_extremum(grid_input.values, tables, is_min=False)
    gs = GridSet(lower, upper, np.stack([lo_idx, hi_idx], axis=-1).astype(np.int32), grid_input.axes)
    meta = {"config": cfg.to_dict(), "neighbors_per_axis": [t.shape[1] for t in tables]}
    arrays = {"lower": lower, "upper": upper, "corner_idx": gs.corner_idx}
    cache.save(path(cfg), {**arrays, **{f"axis{d}": a for d, a in enumerate(gs.axes)}}, meta)
    return gs


def load(cfg: Config) -> GridSet:
    a, _ = cache.load(path(cfg))
    return GridSet(a["lower"], a["upper"], a["corner_idx"], [a[f"axis{d}"] for d in range(3)])
