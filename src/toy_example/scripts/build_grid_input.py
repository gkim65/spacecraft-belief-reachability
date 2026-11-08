"""Build the MPC GridInput.

    uv run python -m toy_example.scripts.build_grid_input [--resolution 100 100 100]
"""

import argparse
import warnings

from toy_example.config import rover_dark
from toy_example.grids import grid_input


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, nargs=3, default=[100, 100, 100])
    args = parser.parse_args()
    warnings.filterwarnings("ignore", module="do_mpc")
    cfg = rover_dark(tuple(args.resolution))
    gi = grid_input.build(cfg)
    print(f"saved {grid_input.path(cfg)}  omega range [{gi.values.min():.3f}, {gi.values.max():.3f}]")


if __name__ == "__main__":
    main()
