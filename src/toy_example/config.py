"""Experiment configuration. Defaults reproduce rover-core's RoverDark setup."""

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    radius: float


ROVER_DARK_OBSTACLES = (
    Circle(3.0, 2.0, 1.0),
    Circle(6.0, -1.0, 0.3),
    Circle(9.0, 0.8, 0.5),
    Circle(12.0, -1.5, 0.7),
    Circle(14.0, 1.0, 0.4),
    Circle(17.0, -0.5, 0.6),
    Circle(18.5, 1.5, 0.4),
)


@dataclass(frozen=True)
class SystemConfig:
    speed: float = 1.0
    omega_max: float = 1.0
    state_lo: tuple[float, float, float] = (0.0, -5.0, -math.pi)
    state_hi: tuple[float, float, float] = (20.0, 5.0, math.pi)
    obstacles: tuple[Circle, ...] = ROVER_DARK_OBSTACLES
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
    resolution: tuple[int, int, int] = (100, 100, 100)


@dataclass(frozen=True)
class Config:
    tag: str = "rover_dark"
    system: SystemConfig = field(default_factory=SystemConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    grid: GridConfig = field(default_factory=GridConfig)

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / self.tag

    def to_dict(self) -> dict:
        return asdict(self)


def rover_dark(resolution: tuple[int, int, int] = (100, 100, 100), tag: str | None = None) -> Config:
    tag = tag or ("rover_dark" if resolution == (100, 100, 100) else "rover_dark_" + "x".join(map(str, resolution)))
    return Config(tag=tag, grid=GridConfig(resolution))


def nominal(cfg: Config) -> Config:
    """Same setup with zero perception error."""
    return replace(cfg, system=replace(cfg.system, error_bound=(0.0, 0.0, 0.0)))
