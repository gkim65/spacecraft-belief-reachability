"""Initial states per scenario. rover_dark: rover-core config/simulations.yaml (RoverDark default)."""

import numpy as np

ROVER_DARK = np.array(
    [
        [4.00, 0.50, 0.00],
        [4.35, 1.57, 0.00],
        [12.14, 0.77, 0.00],
        [14.30, -0.55, 0.00],
        [13.89, 3.56, 0.00],
        [9.60, 2.30, 0.00],
    ]
)

ONE_OBSTACLE = np.array(
    [
        [3.00, 0.00, 0.00],
        [3.00, 1.00, 0.00],
        [3.00, -1.00, 0.00],
        [6.00, 0.50, 0.00],
        [6.00, -0.50, 0.00],
        [7.50, 0.00, 0.00],
    ]
)

INITIAL_STATES = {"rover_dark": ROVER_DARK, "one_obstacle": ONE_OBSTACLE}
