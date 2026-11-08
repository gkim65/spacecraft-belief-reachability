"""State grid used by GridInput and GridSet (rover-core convention: inclusive linspace)."""

import numpy as np

from toy_example.config import Config


def state_axes(cfg: Config) -> list[np.ndarray]:
    lo, hi, n = cfg.system.state_lo, cfg.system.state_hi, cfg.grid.resolution
    return [np.linspace(lo[d], hi[d], n[d]) for d in range(3)]


def state_points(axes: list[np.ndarray]) -> np.ndarray:
    """All grid states, shape (n0, n1, n2, 3)."""
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Nearest index on a sorted axis, clamped to the ends (port of rover-core utils/grids.py)."""
    idx = np.clip(np.searchsorted(axis, values, side="left"), 0, axis.size - 1)
    prev = np.maximum(idx - 1, 0)
    use_prev = np.abs(values - axis[prev]) < np.abs(values - axis[idx])
    return np.where(use_prev, prev, idx)


def nearest_indices(axes: list[np.ndarray], states: np.ndarray) -> np.ndarray:
    """Per-axis nearest indices, shape (..., 3)."""
    return np.stack([nearest_axis_indices(ax, states[..., d]) for d, ax in enumerate(axes)], axis=-1)


def lookup(values: np.ndarray, axes: list[np.ndarray], states: np.ndarray) -> np.ndarray:
    """Nearest-node lookup of a grid array with trailing channel dims."""
    idx = nearest_indices(axes, states)
    return values[idx[..., 0], idx[..., 1], idx[..., 2]]
