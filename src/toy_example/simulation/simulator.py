"""Forward-Euler closed-loop rollout (port of rover-core core/simulators.py)."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from toy_example.config import SystemConfig
from toy_example.simulation.uncertainty import Player
from toy_example.systems.dubins import dynamics, error_limits, failure, wrap_state

Controller = Callable[[np.ndarray], np.ndarray]


@dataclass
class Rollout:
    times: np.ndarray  # (T+1,)
    states: np.ndarray  # (B, T+1, 3)
    estimates: np.ndarray  # (B, T, 3)
    controls: np.ndarray  # (B, T)
    errors: np.ndarray  # (B, T, 3)
    failure: np.ndarray  # (B, T+1) signed distance along the true trajectory

    @property
    def collided(self) -> np.ndarray:
        return (self.failure < 0).any(axis=1)

    @property
    def collision_time(self) -> np.ndarray:
        """First time with failure < 0, NaN if safe."""
        hit = self.failure < 0
        first = np.argmax(hit, axis=1)
        return np.where(hit.any(axis=1), self.times[first], np.nan)


def simulate(
    sys: SystemConfig,
    controller: Controller,
    player: Player,
    initial_states: np.ndarray,
    dt: float = 0.01,
    duration: float | None = None,
) -> Rollout:
    duration = sys.time_horizon if duration is None else duration
    steps = int(round(duration / dt))
    e_lo, e_hi = error_limits(sys)
    s_lo, s_hi = np.array(sys.state_lo), np.array(sys.state_hi)
    omega_max = sys.omega_max

    state = np.clip(wrap_state(np.atleast_2d(initial_states)), s_lo, s_hi)
    b = len(state)
    states = np.empty((b, steps + 1, 3))
    estimates = np.empty((b, steps, 3))
    errors = np.empty((b, steps, 3))
    controls = np.empty((b, steps))
    states[:, 0] = state

    for k in range(steps):
        t = k * dt
        e = np.clip(player(state, t), e_lo, e_hi)
        x_est = wrap_state(state + e)
        u = np.clip(controller(x_est), -omega_max, omega_max)
        state = np.clip(wrap_state(state + dt * dynamics(state, u, sys)), s_lo, s_hi)
        errors[:, k], estimates[:, k], controls[:, k], states[:, k + 1] = e, x_est, u, state

    times = dt * np.arange(steps + 1)
    return Rollout(times, states, estimates, controls, errors, failure(states, sys))
