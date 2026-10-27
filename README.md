# spacecraft-belief-reachability

A stochastic reachability framework for spacecraft–debris conjunction assessment.

The core quantity is the **reach probability**

$$
p_\text{reach}(x, t) = \Pr\big[\text{the relative trajectory from state } x \text{ at time } t \text{ enters the collision set by TCA}\big],
$$

where $x$ is the debris state relative to the spacecraft and TCA is the time of
closest approach. The same function serves two roles:

- **Risk estimator.** Evaluated on the current state estimate, $p_\text{reach}$ is
  the collision probability for the encounter.
- **Control barrier function.** A threshold $\varepsilon$ defines the safe set
  $\{x : p_\text{reach}(x, t) \le \varepsilon\}$. The barrier
  $h(x, t) = \varepsilon - p_\text{reach}(x, t)$ can then constrain a
  safety-critical maneuver controller.

## Structure

The project has two layers.

| Layer | Status | What it does |
|---|---|---|
| Simulation | In progress | Full-fidelity orbit propagation and estimation with [brahe](https://docs.brahe.space/). Generates ground-truth conjunctions. |
| Reachability | Not started | Computes $p_\text{reach}(x, t)$ and exposes it as a risk estimate and a CBF. |

```
src/spacecraft_belief_reachability/
    simulation/      brahe-propagated conjunction scenarios
    dynamics/        relative-motion models
    estimation/      state estimation and belief propagation
    reachability/    p_reach computation and CBF interface
    plotting.py      figure styling and scenario plots
    utils/           shared helpers
notebooks/           exploratory analysis
scripts/             runnable entry points
tests/               pytest suite
```

## Conjunction scenarios

`simulation.ConjunctionScenario` generates a conjunction from a few parameters:

- **Spacecraft:** starts on a circular orbit, 550 km altitude and 51.6° inclination
  by default.
- **Debris:** specified by a `DebrisOrbit` (inclination, RAAN offset, and optionally
  eccentricity and semi-major axis).
- **Encounter:** a target TCA (default about 24 h after the start) and a target
  miss distance (default 750 m).

The generator finds where the spacecraft crosses the debris plane near the target
TCA. It places the debris there, offset by the miss distance, and propagates it
backward to the start epoch. TCA and miss therefore land where requested for any
orbit shape.

Both objects use brahe's `NumericalOrbitPropagator` with `ForceModelConfig.leo_default()`:
30×30 gravity, NRLMSISE-00 drag, solar radiation pressure, and Sun and Moon
gravity. The spacecraft and debris have different area-to-mass ratios, so drag
acts on them differently. Relative states are in the spacecraft's RTN frame via
`brahe.state_eci_to_rtn`.

Three presets come built in:

| Preset | Debris orbit | Relative speed | Other close passes |
|---|---|---|---|
| `crossing()` | circular, +3° inclination, +5° RAAN | ~0.7 km/s | yes, ~13 km, every half orbit |
| `high_angle()` | circular, 97.6° inclination, +30° RAAN | ~6.9 km/s | yes, ~19 km |
| `eccentric_debris()` | 450 × 850 km, crossing plane as `crossing()` | ~0.7 km/s | no, next pass ~290 km |

With circular debris both orbits have the same period, so the two objects meet
again near the plane crossings every half orbit. Eccentric debris has a different
period, which makes the encounter isolated.

```python
from spacecraft_belief_reachability.simulation import ConjunctionScenario, DebrisOrbit
from spacecraft_belief_reachability.plotting import plot_conjunction, save_figure

scenario = ConjunctionScenario.eccentric_debris()  # ~6 s: propagates both objects
tca = scenario.find_tca()
print(f"TCA {tca.t_s / 3600:.2f} h, miss {tca.miss_m:.0f} m")
print(len(scenario.close_approaches(max_range_m=100e3)), "pass(es) within 100 km")

fig = plot_conjunction(scenario, tca=tca, theme="light")  # or theme="dark" for slides
save_figure(fig, "figures/eccentric_debris")  # writes PDF, SVG and PNG

# A custom conjunction: polar debris, 200 m miss, TCA 12 h after the start
custom = ConjunctionScenario(
    DebrisOrbit(inclination_deg=90.0, raan_offset_deg=45.0),
    target_miss_m=200.0,
    target_tca_s=12 * 3600.0,
)
```

`notebooks/01_conjunction_scenarios.ipynb` plots all three presets. It is committed
without outputs, so run it locally to see the figures.

## Setup

The project needs Python ≥ 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # create .venv from uv.lock
uv run pytest           # run the test suite
uv run ruff check .     # lint
```

Running brahe needs Earth orientation parameter (EOP) and space-weather data. On
import, the simulation module calls `brahe.initialize_eop()` and
`brahe.initialize_sw()`, which download the data on first use and cache it under
`~/.cache/brahe/`.
