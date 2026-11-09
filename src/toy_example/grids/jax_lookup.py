"""JAX nearest-node lookups of GridInput / GridSet for use inside the HJ Hamiltonian
(port of rover-core impl/inputs/jax_grid_input.py and impl/sets/jax_grid_set.py, box case)."""

import jax.numpy as jnp

from toy_example.grids.grid_input import GridInput
from toy_example.grids.grid_set import GridSet


def _nearest_index(axes: list[jnp.ndarray], state: jnp.ndarray) -> tuple:
    return tuple(jnp.argmin(jnp.abs(ax - state[d])) for d, ax in enumerate(axes))


class JaxGridInput:
    def __init__(self, grid_input: GridInput):
        self.values = jnp.asarray(grid_input.values)
        self.axes = [jnp.asarray(ax) for ax in grid_input.axes]

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.values[_nearest_index(self.axes, state)]


class JaxGridSet:
    def __init__(self, grid_set: GridSet):
        self.lower = jnp.asarray(grid_set.lower)
        self.upper = jnp.asarray(grid_set.upper)
        self.axes = [jnp.asarray(ax) for ax in grid_set.axes]

    def bounds(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        i = _nearest_index(self.axes, state)
        return self.lower[i], self.upper[i]

    def argmax_support(self, direction: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        lower, upper = self.bounds(state)
        return jnp.where(direction >= 0, upper, lower)
