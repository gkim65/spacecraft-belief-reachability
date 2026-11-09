"""HJ dynamics for the Dubins car under a given controller (port of rover-core impl/hj_reachability/rover_dark.py).

H = ∂V/∂x v cosθ + ∂V/∂y v sinθ + ∂V/∂θ ω, with
  nominal: ω = GridInput(x)
  robust:  ω = argmin over [lower(x), upper(x)] of ∂V/∂θ ω  (worst perception error)
"""

import hj_reachability as hj
import jax.numpy as jnp

from toy_example.config import SystemConfig
from toy_example.grids.grid_input import GridInput
from toy_example.grids.grid_set import GridSet
from toy_example.grids.jax_lookup import JaxGridInput, JaxGridSet


class _Dubins(hj.Dynamics):
    def __init__(self, sys: SystemConfig):
        self.speed = sys.speed

    def omega(self, state, grad_value):
        raise NotImplementedError

    def omega_bounds(self, state):
        raise NotImplementedError

    def hamiltonian(self, state, time, value, grad_value):
        theta = state[2]
        drift = grad_value[0] * self.speed * jnp.cos(theta) + grad_value[1] * self.speed * jnp.sin(theta)
        return drift + grad_value[2] * self.omega(state, grad_value)

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        theta = state[2]
        lo, hi = self.omega_bounds(state)
        return jnp.array([
            jnp.abs(self.speed * jnp.cos(theta)),
            jnp.abs(self.speed * jnp.sin(theta)),
            jnp.maximum(jnp.abs(lo), jnp.abs(hi)),
        ])

    def __call__(self, state, control, disturbance, time):
        raise NotImplementedError("the solver only uses hamiltonian()")

    def optimal_control_and_disturbance(self, state, time, grad_value):
        raise NotImplementedError("the solver only uses hamiltonian()")


class Nominal(_Dubins):
    def __init__(self, sys: SystemConfig, grid_input: GridInput):
        super().__init__(sys)
        self.omega_max = sys.omega_max
        self.policy = JaxGridInput(grid_input)

    def omega(self, state, grad_value):
        return self.policy(state)

    def omega_bounds(self, state):
        return -self.omega_max, self.omega_max


class Robust(_Dubins):
    def __init__(self, sys: SystemConfig, grid_set: GridSet):
        super().__init__(sys)
        self.control_set = JaxGridSet(grid_set)

    def omega(self, state, grad_value):
        return self.control_set.argmax_support(-grad_value[2], state)

    def omega_bounds(self, state):
        return self.control_set.bounds(state)
