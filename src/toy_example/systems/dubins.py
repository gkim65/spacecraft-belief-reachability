"""Dubins car: dynamics, failure function and perception-error bounds."""

import numpy as np

from toy_example.config import SystemConfig


def dynamics(state: np.ndarray, omega: np.ndarray, sys: SystemConfig) -> np.ndarray:
    theta = state[..., 2]
    return np.stack([sys.speed * np.cos(theta), sys.speed * np.sin(theta), omega], axis=-1)


def failure(state: np.ndarray, sys: SystemConfig) -> np.ndarray:
    """Signed distance to the nearest obstacle (negative inside)."""
    xy = state[..., :2]
    dists = [np.hypot(xy[..., 0] - o.cx, xy[..., 1] - o.cy) - o.radius for o in sys.obstacles]
    return np.min(np.stack(dists), axis=0)


def error_limits(sys: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    """Constant perception-error box E = [-ē, ē]."""
    e = np.asarray(sys.error_bound, dtype=float)
    return -e, e


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """Wrap to [-π, π)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def wrap_state(state: np.ndarray) -> np.ndarray:
    out = np.array(state, dtype=float, copy=True)
    out[..., 2] = wrap_angle(out[..., 2])
    return out
