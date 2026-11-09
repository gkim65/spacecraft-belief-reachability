"""Experiment configuration: Dubins car, one circular obstacle, MPC under perception error."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache"
OUTPUT_ROOT = ROOT / "outputs"

# (x, y, θ) nodes; odd θ count puts nodes exactly at θ = 0, ±π/2, ±π
RESOLUTION = (101, 101, 101)


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    radius: float


OBSTACLES = (Circle(10.0, 0.0, 2.0),)


@dataclass(frozen=True)
class SystemConfig:
    speed: float = 3.0
    omega_max: float = 1.0
    state_lo: tuple[float, float, float] = (0.0, -5.0, -math.pi)
    state_hi: tuple[float, float, float] = (20.0, 5.0, math.pi)
    obstacles: tuple[Circle, ...] = OBSTACLES
    goal: tuple[float, float] = (20.0, 0.0)
    time_horizon: float = 5.0
    error_bound: tuple[float, float, float] = (0.5, 0.5, 0.1)


@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.1
    horizon: int = 5
    control_weight: float = 1e-2
    obstacle_weight: float = 20.0
    obstacle_margin: float = 0.5
    robustify_uncertainty: bool = True
    robust_margin_factor: float = 1.0
    num_workers: int = -1
    batch_size: int = 100_000


@dataclass(frozen=True)
class GridConfig:
    resolution: tuple[int, int, int] = RESOLUTION


@dataclass(frozen=True)
class Config:
    tag: str = "one_obstacle"
    system: SystemConfig = field(default_factory=SystemConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    grid: GridConfig = field(default_factory=GridConfig)

    @property
    def cache_dir(self) -> Path:
        """Intermediate stage results (GridInput, GridSet, GridValue)."""
        return CACHE_ROOT / self.tag

    @property
    def output_dir(self) -> Path:
        """Final outcomes (figures)."""
        return OUTPUT_ROOT / self.tag

    def to_dict(self) -> dict:
        return asdict(self)
