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
    utils/           shared helpers (figure styling)
notebooks/           exploratory analysis
scripts/             runnable entry points
tests/               pytest suite
```

## Current scenario: `CrossingConjunction`

`simulation.scenario.CrossingConjunction` is a LEO crossing conjunction.

- **Spacecraft:** circular orbit at 550 km altitude, 51.6° inclination.
- **Debris:** also circular at 550 km. Its plane is offset by 5° in RAAN and 3° in
  inclination.
- **Phasing:** the debris starting phase is solved numerically. It places a
  closest approach of about 750 m roughly 24 h after the start epoch.
- **Propagation:** brahe's `NumericalOrbitPropagator` with 20×20 EGM2008 gravity
  (`ForceModelConfig.earth_gravity()`).
- **Relative states:** expressed in the spacecraft's RTN frame via
  `brahe.state_eci_to_rtn`.

```python
from spacecraft_belief_reachability.simulation import CrossingConjunction
from spacecraft_belief_reachability.utils.plotting import save_figure

scenario = CrossingConjunction()          # ~10-15 s: propagates and phases both objects
tca = scenario.find_tca()
print(f"TCA {tca.t_s / 3600:.2f} h, miss {tca.miss_m:.0f} m")

fig = scenario.plot_conjunction(theme="light")   # or theme="dark" for slides
save_figure(fig, "figures/crossing_conjunction") # writes PDF, SVG and PNG
```

## Setup

The project needs Python ≥ 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # create .venv from uv.lock
uv run pytest           # run the test suite
uv run ruff check .     # lint
```

Running brahe needs Earth orientation parameter (EOP) data. On import, the
simulation module calls `brahe.initialize_eop()`. That call downloads the data on
first use and caches it under `~/.cache/brahe/`.
