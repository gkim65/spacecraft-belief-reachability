import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spacecraft_belief_reachability.plotting import plot_conjunction, plot_orbits_3d
from spacecraft_belief_reachability.simulation import ConjunctionScenario, DebrisOrbit

PRESETS = ["crossing", "high_angle", "eccentric_debris"]


@pytest.fixture(scope="module", params=PRESETS)
def scenario(request):
    return getattr(ConjunctionScenario, request.param)()


@pytest.fixture(scope="module")
def tca(scenario):
    return scenario.find_tca()


def test_tca_found_near_one_day(scenario, tca):
    assert 0.0 < tca.t_s < scenario.duration_s
    assert abs(tca.t_s - scenario.target_tca_s) < 3600.0
    assert abs(tca.t_s - scenario.tca_design_s) < 1.0


def test_miss_distance_reasonable(scenario, tca):
    assert 0.0 < tca.miss_m < 5000.0
    assert abs(tca.miss_m - scenario.target_miss_m) < 5.0


def test_tca_is_a_range_minimum(tca):
    r, v = tca.rel_rtn[:3], tca.rel_rtn[3:]
    assert np.isclose(np.linalg.norm(r), tca.miss_m, rtol=1e-9)
    # Range rate r.v/|r| vanishes at a range minimum; |r| is frame-invariant, so RTN components suffice.
    assert abs(r @ v / np.linalg.norm(r)) < 1.0


def test_eccentric_debris_has_a_single_encounter():
    scenario = ConjunctionScenario.eccentric_debris()
    assert len(scenario.close_approaches(max_range_m=100e3)) == 1


def test_high_angle_is_fast():
    tca = ConjunctionScenario.high_angle().find_tca()
    assert tca.relative_speed_m_s > 5e3


def test_unreachable_eccentric_orbit_raises():
    from brahe.constants import R_EARTH

    orbit = DebrisOrbit(54.6, 5.0, eccentricity=0.01, semi_major_axis_m=R_EARTH + 900e3)
    with pytest.raises(ValueError, match="never reaches"):
        ConjunctionScenario(orbit, duration_s=86400.0 + 3600.0)


@pytest.mark.parametrize("plot", [plot_conjunction, plot_orbits_3d])
def test_plots_return_figures(scenario, tca, plot):
    fig = plot(scenario, tca=tca)
    assert fig.axes
    plt.close(fig)
