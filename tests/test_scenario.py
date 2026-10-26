import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from spacecraft_belief_reachability.simulation import CrossingConjunction


@pytest.fixture(scope="module")
def scenario():
    return CrossingConjunction()


@pytest.fixture(scope="module")
def tca(scenario):
    return scenario.find_tca()


def test_tca_found_near_one_day(scenario, tca):
    assert 0.0 < tca.t_s < scenario.duration_s
    assert abs(tca.t_s - scenario.target_tca_s) < 3600.0


def test_miss_distance_reasonable(tca):
    assert 0.0 < tca.miss_m < 5000.0
    assert 500.0 <= tca.miss_m <= 1000.0


def test_tca_is_a_range_minimum(tca):
    r, v = tca.rel_rtn[:3], tca.rel_rtn[3:]
    assert np.isclose(np.linalg.norm(r), tca.miss_m, rtol=1e-9)
    # Range rate r.v/|r| vanishes at a range minimum; |r| is frame-invariant, so RTN components suffice.
    assert abs(r @ v / np.linalg.norm(r)) < 1.0


def test_plot_conjunction_returns_figure(scenario, tca):
    import matplotlib.pyplot as plt

    fig = scenario.plot_conjunction(tca=tca)
    assert len(fig.axes) == 3
    plt.close(fig)
