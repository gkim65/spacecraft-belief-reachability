"""Perception-error players: e = player(state, t), with state of shape (B, 3)."""

from typing import Callable

import numpy as np

from toy_example.config import SystemConfig
from toy_example.grids.grid_set import GridSet
from toy_example.reachability.grid_value import GridValue
from toy_example.systems.dubins import error_limits, wrap_angle

Player = Callable[[np.ndarray, float], np.ndarray]


def zero() -> Player:
    return lambda state, t: np.zeros_like(state)


def uniform_random(sys: SystemConfig, seed: int = 0) -> Player:
    lo, hi = error_limits(sys)
    rng = np.random.default_rng(seed)
    return lambda state, t: rng.uniform(lo, hi, size=state.shape)


def worst_case(value: GridValue, control_set: GridSet) -> Player:
    """Error whose perceived state x̂ yields the bound minimizing ∂V/∂θ ω (rover-core RuntimeRoverDark).

    x̂ is the stored extremizer node of the GridSet; e_θ uses the wrapped angle difference.
    """
    def player(state: np.ndarray, t: float) -> np.ndarray:
        direction = -value.gradient(state, t)[..., 2]
        e = control_set.extremizer_state(state, use_upper=direction >= 0) - state
        e[..., 2] = wrap_angle(e[..., 2])
        return e
    return player
