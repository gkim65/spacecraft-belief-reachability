"""Perception-error players: e = player(state, t), with state of shape (B, 3)."""

from typing import Callable

import numpy as np

from toy_example.config import SystemConfig
from toy_example.systems.dubins import error_limits

Player = Callable[[np.ndarray, float], np.ndarray]


def zero() -> Player:
    return lambda state, t: np.zeros_like(state)


def uniform_random(sys: SystemConfig, seed: int = 0) -> Player:
    lo, hi = error_limits(sys)
    rng = np.random.default_rng(seed)
    return lambda state, t: rng.uniform(lo, hi, size=state.shape)
