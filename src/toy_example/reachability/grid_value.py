"""BRT value function on the hj grid (port of rover-core scripts/grid_value/build_grid_value.py, obstacle BRT).

Stored with ascending times [0..T]; V(t, x) < 0 means x reaches the obstacles within T - t.
Lookups use the nearest node and nearest time, as rover-core's GridValue does in simulation.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import hj_reachability as hj
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from toy_example import cache
from toy_example.config import Config
from toy_example.grids.axes import nearest_axis_indices, nearest_indices
from toy_example.grids.grid_input import GridInput
from toy_example.grids.grid_set import GridSet
from toy_example.reachability.dynamics import Nominal, Robust
from toy_example.systems.dubins import failure

TIME_STEPS = 100


@dataclass
class GridValue:
    values: np.ndarray  # (T, n0, n1, n2)
    gradients: np.ndarray  # (T, n0, n1, n2, 3)
    times: np.ndarray  # (T,) ascending
    axes: list[np.ndarray]

    def _index(self, states: np.ndarray, t: float) -> tuple:
        k = int(nearest_axis_indices(self.times, np.array([t]))[0])
        i = nearest_indices(self.axes, states)
        return k, i[..., 0], i[..., 1], i[..., 2]

    def value(self, states: np.ndarray, t: float) -> np.ndarray:
        return self.values[self._index(states, t)]

    def gradient(self, states: np.ndarray, t: float) -> np.ndarray:
        return self.gradients[self._index(states, t)]


def hj_grid(cfg: Config) -> hj.Grid:
    s = cfg.system
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(lo=jnp.array(s.state_lo), hi=jnp.array(s.state_hi)),
        shape=cfg.grid.resolution,
        periodic_dims=2,
    )


def solve(cfg: Config, dynamics: hj.Dynamics) -> GridValue:
    grid = hj_grid(cfg)
    initial_values = jnp.asarray(failure(np.asarray(grid.states), cfg.system), dtype=jnp.float32)
    settings = hj.SolverSettings.with_accuracy(
        "very_high",
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        value_postprocessor=lambda t, v: jnp.clip(v, jnp.min(initial_values), None),
    ).replace(artificial_dissipation_scheme=hj.artificial_dissipation.local_lax_friedrichs, CFL_number=0.75)

    times = np.linspace(cfg.system.time_horizon, 0.0, TIME_STEPS)
    values = hj.solve(settings, dynamics, grid, times, initial_values, progress_bar=True)
    grad_fn = jax.jit(lambda v: grid.grad_values(v, settings.upwind_scheme))
    gradients = np.stack([np.asarray(grad_fn(v)) for v in tqdm(values, desc="gradients")])

    axes = [np.asarray(c) for c in grid.coordinate_vectors]
    return GridValue(np.asarray(values)[::-1].copy(), gradients[::-1].copy(), times[::-1].copy(), axes)


def path(cfg: Config, kind: str) -> Path:
    return cfg.output_dir / f"grid_value_{kind}.npz"


def build(cfg: Config, kind: str, control: GridInput | GridSet) -> GridValue:
    dynamics = Nominal(cfg.system, control) if kind == "nominal" else Robust(cfg.system, control)
    start = time.perf_counter()
    gv = solve(cfg, dynamics)
    meta = {"config": cfg.to_dict(), "kind": kind, "time_steps": TIME_STEPS, "build_seconds": time.perf_counter() - start}
    arrays = {"values": gv.values, "gradients": gv.gradients, "times": gv.times}
    cache.save(path(cfg, kind), {**arrays, **{f"axis{d}": a for d, a in enumerate(gv.axes)}}, meta, compress=False)
    return gv


def load(cfg: Config, kind: str) -> GridValue:
    a, _ = cache.load(path(cfg, kind))
    return GridValue(a["values"], a["gradients"], a["times"], [a[f"axis{d}"] for d in range(3)])
