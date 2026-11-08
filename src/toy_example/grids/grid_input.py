"""GridInput: the MPC evaluated at every state-grid node (rover-core scripts/grid_input)."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from toy_example import cache
from toy_example.config import Config
from toy_example.controllers.mpc import MPC
from toy_example.grids.axes import lookup, state_axes, state_points


@dataclass
class GridInput:
    values: np.ndarray  # omega at each node, shape (n0, n1, n2)
    axes: list[np.ndarray]

    def __call__(self, states: np.ndarray) -> np.ndarray:
        return lookup(self.values, self.axes, states)


def path(cfg: Config) -> Path:
    return cfg.output_dir / "grid_input.npz"


def build(cfg: Config) -> GridInput:
    axes = state_axes(cfg)
    flat = state_points(axes).reshape(-1, 3)
    out = np.empty(len(flat))
    bs = cfg.mpc.batch_size
    start = time.perf_counter()
    with MPC(cfg.mpc, cfg.system) as mpc:
        for i in tqdm(range(0, len(flat), bs), desc="GridInput"):
            out[i : i + bs] = mpc(flat[i : i + bs])
        margin = mpc.margin
    grid_input = GridInput(out.reshape(cfg.grid.resolution), axes)
    meta = {"config": cfg.to_dict(), "effective_margin": margin, "build_seconds": time.perf_counter() - start}
    cache.save(path(cfg), {"values": grid_input.values, **{f"axis{d}": a for d, a in enumerate(axes)}}, meta)
    return grid_input


def load(cfg: Config) -> GridInput:
    arrays, _ = cache.load(path(cfg))
    return GridInput(arrays["values"], [arrays[f"axis{d}"] for d in range(3)])
