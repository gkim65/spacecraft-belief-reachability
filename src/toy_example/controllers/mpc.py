"""Port of rover-core's RoverDark_MPC (do-mpc + CasADi + IPOPT).

Behaviour matches rover-core: each solver (main or worker) warm-starts from its
previous solve, and the rterm penalizes the change from the previous control.
"""

import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore", message="The .* feature is not available", module="do_mpc")

import casadi as ca
import do_mpc
import numpy as np

from toy_example.config import Circle, MPCConfig, SystemConfig

ca.GlobalOptions.setNumpyMode(-1)  # keep casadi's legacy numpy behaviour without the FutureWarning


def effective_margin(mpc_cfg: MPCConfig, sys: SystemConfig) -> float:
    """Obstacle margin inflated by the xy perception-error radius at the end of the MPC horizon."""
    if not mpc_cfg.robustify_uncertainty:
        return mpc_cfg.obstacle_margin
    ex, ey, _ = sys.error_bound
    return mpc_cfg.obstacle_margin + mpc_cfg.robust_margin_factor * float(np.hypot(ex, ey))


def _build_model(speed: float) -> do_mpc.model.Model:
    model = do_mpc.model.Model("continuous")
    model.set_variable(var_type="_x", var_name="px")
    model.set_variable(var_type="_x", var_name="py")
    theta = model.set_variable(var_type="_x", var_name="theta")
    omega = model.set_variable(var_type="_u", var_name="omega")
    model.set_rhs("px", speed * ca.cos(theta))
    model.set_rhs("py", speed * ca.sin(theta))
    model.set_rhs("theta", omega)
    model.setup()
    return model


def _softplus(value, beta: float = 10.0):
    return ca.log1p(ca.exp(beta * value)) / beta


def _collision_penalty(px, py, obstacles: tuple[Circle, ...], weight: float, margin: float):
    penalties = []
    for o in obstacles:
        distance = ca.sqrt((px - o.cx) ** 2 + (py - o.cy) ** 2 + 1e-8) - o.radius
        penalties.append(_softplus(margin - distance))
    return weight * sum(penalties)


def _build_mpc(mpc_cfg: MPCConfig, sys: SystemConfig) -> do_mpc.controller.MPC:
    model = _build_model(sys.speed)
    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=mpc_cfg.horizon,
        t_step=mpc_cfg.dt,
        n_robust=0,
        store_full_solution=False,
        nlpsol_opts={"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0},
    )
    px, py = model.x["px"], model.x["py"]
    goal_cost = (px - sys.goal[0]) ** 2 + (py - sys.goal[1]) ** 2
    penalty = _collision_penalty(px, py, sys.obstacles, mpc_cfg.obstacle_weight, effective_margin(mpc_cfg, sys))
    mpc.set_objective(lterm=goal_cost + penalty, mterm=goal_cost + penalty)
    mpc.set_rterm(omega=mpc_cfg.control_weight)
    mpc.bounds["lower", "_u", "omega"] = -sys.omega_max
    mpc.bounds["upper", "_u", "omega"] = sys.omega_max
    mpc.setup()
    return mpc


class _WarmStartSolver:
    def __init__(self, mpc_cfg: MPCConfig, sys: SystemConfig):
        self.mpc = _build_mpc(mpc_cfg, sys)
        self.initialized = False

    def solve(self, state: np.ndarray) -> float:
        x0 = np.asarray(state, dtype=float).reshape(3, 1)
        self.mpc.x0 = x0
        if not self.initialized:
            self.mpc.set_initial_guess()
            self.initialized = True
        return float(np.asarray(self.mpc.make_step(x0)).reshape(-1)[0])


_worker: _WarmStartSolver | None = None


def _worker_init(mpc_cfg: MPCConfig, sys: SystemConfig) -> None:
    global _worker
    _worker = _WarmStartSolver(mpc_cfg, sys)


def _worker_solve(state: np.ndarray) -> float:
    return _worker.solve(state)


class MPC:
    """Callable controller: omega = mpc(states) for states of shape (..., 3)."""

    def __init__(self, mpc_cfg: MPCConfig, sys: SystemConfig, parallel_threshold: int = 10):
        self.cfg = mpc_cfg
        self.sys = sys
        self.margin = effective_margin(mpc_cfg, sys)
        self.num_workers = max(1, mp.cpu_count() - 1) if mpc_cfg.num_workers == -1 else max(1, mpc_cfg.num_workers)
        self.parallel_threshold = parallel_threshold
        self._solver = _WarmStartSolver(mpc_cfg, sys)
        self._pool = None

    def __call__(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        flat = states.reshape(-1, 3)
        if self.num_workers > 1 and len(flat) >= self.parallel_threshold:
            omega = np.array(self._get_pool().map(_worker_solve, flat))
        else:
            omega = np.array([self._solver.solve(s) for s in flat])
        return omega.reshape(states.shape[:-1])

    def _get_pool(self):
        if self._pool is None:
            self._pool = mp.get_context("spawn").Pool(
                self.num_workers, initializer=_worker_init, initargs=(self.cfg, self.sys)
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
