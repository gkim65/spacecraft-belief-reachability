# Toy example: Dubins car (RoVer-CoRe)

`src/toy_example/` reproduces the RoVer-CoRe pipeline (Lin, Pinto, Bansal 2025) on a
Dubins car with one circular obstacle: MPC → GridInput → GridSet → nominal and robust
BRT → worst-case perception-error simulation.

```
config.py        every experiment setting (system, MPC, grid)
systems/         Dubins dynamics and failure function
controllers/     MPC (do-mpc + CasADi)
grids/           GridInput, GridSet and their JAX lookups
reachability/    hj_reachability dynamics and BRT solve
simulation/      simulator, perception-error players, initial states
visualization/   figures
notebooks/dubins.ipynb   runs the whole pipeline
```

Run `notebooks/dubins.ipynb` with the repo's `.venv` kernel
(`.venv/bin/python`). Each stage caches its result in `.cache/<tag>/`
and rebuilds it when the cache is missing or was built with a different config.
Figures go to `outputs/<tag>/figures/`. Both folders are gitignored,
so the first run builds everything:

- **GridInput** (one MPC solve per grid node, 101³ ≈ 1M) is the slow stage. It uses
  all CPU cores but one: about 2 min on 47 cores, roughly 10 to 15 min on an 8-core
  laptop. Lower `RESOLUTION` in `config.py` for quick runs.
- **BRTs** run on JAX: about 25 s each on CPU, a few seconds on a GPU.
- **Disk and memory:** the 101³ cache takes about 3.3 GB, mostly the two value
  functions with their gradients; a full run peaks at about 6.5 GB of RAM.

Environment setup is in the [top-level README](../../README.md#setup).
