"""Build the MPC GridSet from a cached GridInput.

    uv run python -m toy_example.scripts.build_grid_set [--scenario rover_dark|one_obstacle] [--resolution 100 100 100]
"""

import argparse

from toy_example.config import SCENARIOS
from toy_example.grids import grid_input, grid_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="rover_dark")
    parser.add_argument("--resolution", type=int, nargs=3, default=[100, 100, 100])
    args = parser.parse_args()
    cfg = SCENARIOS[args.scenario](tuple(args.resolution))
    gs = grid_set.build(cfg, grid_input.load(cfg))
    width = gs.upper - gs.lower
    print(f"saved {grid_set.path(cfg)}  width mean {width.mean():.3f}  max {width.max():.3f}")


if __name__ == "__main__":
    main()
