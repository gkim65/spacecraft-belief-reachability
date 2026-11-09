"""Simulation initial states (x, y, θ), all facing θ = 0 so one BRT slice covers them."""

import numpy as np

INITIAL_STATES = np.array(
    [
        [1.00, 1.00, 0.00],
        [2.00, -4.00, 0.00],
        [2.00, -2.00, 0.00],
        [2.00, 0.00, 0.00],
        [2.00, 2.00, 0.00],
        [4.00, -1.00, 0.00],
        [5.00, 4.50, 0.00],
        [6.00, -1.50, 0.00],
        [6.00, 0.50, 0.00],
        [6.00, 2.50, 0.00],
    ]
)
