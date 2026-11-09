"""Solve the nominal (GridInput) and/or robust (GridSet) BRT.

    uv run python -m toy_example.scripts.build_grid_value [--kind nominal robust] [--resolution 100 100 100]
"""

import argparse

from toy_example.config import rover_dark
from toy_example.grids import grid_input, grid_set
from toy_example.reachability import grid_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", nargs="+", choices=["nominal", "robust"], default=["nominal", "robust"])
    parser.add_argument("--resolution", type=int, nargs=3, default=[100, 100, 100])
    args = parser.parse_args()
    cfg = rover_dark(tuple(args.resolution))
    for kind in args.kind:
        control = grid_input.load(cfg) if kind == "nominal" else grid_set.load(cfg)
        gv = grid_value.build(cfg, kind, control)
        print(f"saved {grid_value.path(cfg, kind)}  unsafe fraction at t=0: {(gv.values[0] < 0).mean():.2%}")


if __name__ == "__main__":
    main()
