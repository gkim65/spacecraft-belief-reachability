"""Robust BRT as the perception-error bound is scaled: E -> s E for each s, same GridInput controller.

Only the (x, y) slice at one θ is kept, for every solver time. Scale 0 gives the nominal BRT.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from tqdm import tqdm

from toy_example import cache
from toy_example.config import Config
from toy_example.grids import grid_set
from toy_example.grids.axes import nearest_axis_indices
from toy_example.grids.grid_input import GridInput
from toy_example.reachability.dynamics import Robust
from toy_example.reachability.grid_value import solve_values


@dataclass
class BRTSweep:
    scales: np.ndarray  # (S,)
    times: np.ndarray  # (T,) ascending
    values: np.ndarray  # (S, T, n0, n1) at θ
    axes: list[np.ndarray]  # x, y
    theta: float

    def value_slice(self, i: int, t: float) -> np.ndarray:
        k = int(nearest_axis_indices(self.times, np.array([t]))[0])
        return self.values[i, k]


def scaled(cfg: Config, s: float) -> Config:
    e = tuple(s * x for x in cfg.system.error_bound)
    return replace(cfg, system=replace(cfg.system, error_bound=e))


def sweep_config(cfg: Config, scales, theta: float) -> dict:
    return {**cfg.to_dict(), "sweep": {"scales": [float(s) for s in scales], "theta": float(theta)}}


def path(cfg: Config) -> Path:
    return cfg.cache_dir / "brt_sweep.npz"


def build(cfg: Config, gi: GridInput, scales, theta: float) -> BRTSweep:
    slices = []
    for s in tqdm(scales, desc="scales"):
        c = scaled(cfg, s)
        values, times, axes = solve_values(c, Robust(c.system, grid_set.compute(c, gi)), progress_bar=False)
        k = int(nearest_axis_indices(axes[2], np.array([theta]))[0])
        slices.append(values[:, :, :, k])
    sw = BRTSweep(np.asarray(scales, dtype=float), times, np.stack(slices), axes[:2], float(axes[2][k]))
    arrays = {"scales": sw.scales, "times": sw.times, "values": sw.values, "axis0": axes[0], "axis1": axes[1]}
    cache.save(path(cfg), arrays, {"config": sweep_config(cfg, scales, theta), "theta_node": sw.theta})
    return sw


def load(cfg: Config) -> BRTSweep:
    a, meta = cache.load(path(cfg))
    return BRTSweep(a["scales"], a["times"], a["values"], [a["axis0"], a["axis1"]], meta["theta_node"])
