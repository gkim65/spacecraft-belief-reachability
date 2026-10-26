"""LEO crossing conjunction between a spacecraft and a debris object, propagated with brahe."""

from __future__ import annotations

from dataclasses import dataclass

import brahe as bh
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import argrelmin

bh.initialize_eop()

ALTITUDE_M = 550e3
"""Spacecraft and debris altitude above `brahe.constants.R_EARTH` at the start epoch (m)."""

SPACECRAFT_INCLINATION_DEG = 51.6
"""Spacecraft inclination (deg)."""

DEBRIS_RAAN_OFFSET_DEG = 5.0
"""Debris RAAN minus spacecraft RAAN at the start epoch (deg)."""

DEBRIS_INCLINATION_OFFSET_DEG = 3.0
"""Debris inclination minus spacecraft inclination (deg)."""


def default_propagation_config() -> bh.NumericalPropagationConfig:
    """
        default_propagation_config() -> NumericalPropagationConfig

    DP54 with abs/rel tolerances 1e-10/1e-12.

    NOTE: brahe's `high_precision()` config takes ~4 min steps, and interpolating
    its stored trajectory between steps is off by tens of meters after a day. This
    config keeps both the integration and the interpolated state at the cm level.
    """
    return (
        bh.NumericalPropagationConfig.default().with_abs_tol(1e-10).with_rel_tol(1e-12)
    )


@dataclass(frozen=True)
class ClosestApproach:
    """
    Time of closest approach and the relative state there.

      - `epoch` — TCA (brahe `Epoch`)
      - `t_s` — TCA measured from the scenario start epoch (s)
      - `miss_m` — miss distance at TCA (m)
      - `rel_rtn` — debris state relative to the spacecraft in the spacecraft RTN frame,
        `[R, T, N, dR, dT, dN]` (m, m/s)
    """

    epoch: bh.Epoch
    t_s: float
    miss_m: float
    rel_rtn: np.ndarray


class CrossingConjunction:
    """
        CrossingConjunction(epoch=None, target_miss_m=750.0, target_tca_s=86400.0,
                            duration_s=None, force_config=None, propagation_config=None)

    A circular 550 km spacecraft and a debris object on a crossing plane, phased so
    they pass within about `target_miss_m` of each other near `target_tca_s`.

    Both start circular at `ALTITUDE_M`. The debris plane is offset by
    `DEBRIS_RAAN_OFFSET_DEG` in RAAN and `DEBRIS_INCLINATION_OFFSET_DEG` in inclination;
    its starting argument of latitude is solved numerically so that, at the moment
    the spacecraft crosses the debris plane nearest `target_tca_s`, the debris trails
    it along its own track by `target_miss_m`.

      - `epoch` — start epoch (brahe `Epoch`); default 2026-01-01 00:00:00 UTC
      - `target_miss_m` — along-track offset the debris phasing targets (m)
      - `target_tca_s` — time after `epoch` near which the encounter is placed (s)
      - `duration_s` — propagation span after `epoch` (s); default `target_tca_s + 7200`
      - `force_config` — brahe `ForceModelConfig`; default `earth_gravity()` (20x20 EGM2008)
      - `propagation_config` — brahe `NumericalPropagationConfig`; default `default_propagation_config()`

    Attributes `spacecraft` and `debris` are the propagated `NumericalOrbitPropagator`s,
    and `debris_arg_latitude_deg` is the solved debris argument of latitude at `epoch` (deg).

    NOTE: the realized miss is close to, not equal to, `target_miss_m` — the closest
    approach falls slightly off the plane-crossing instant. Use `find_tca()` for the
    actual value.
    """

    def __init__(
        self,
        epoch: bh.Epoch | None = None,
        target_miss_m: float = 750.0,
        target_tca_s: float = 86400.0,
        duration_s: float | None = None,
        force_config: bh.ForceModelConfig | None = None,
        propagation_config: bh.NumericalPropagationConfig | None = None,
    ):
        self.epoch = epoch or bh.Epoch.from_datetime(
            2026, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC
        )
        self.target_miss_m = target_miss_m
        self.target_tca_s = target_tca_s
        self.duration_s = (
            duration_s if duration_s is not None else target_tca_s + 7200.0
        )
        self.force_config = force_config or bh.ForceModelConfig.earth_gravity()
        self.propagation_config = propagation_config or default_propagation_config()

        self.semi_major_axis_m = bh.constants.R_EARTH + ALTITUDE_M
        self.spacecraft_oe = np.array(
            [self.semi_major_axis_m, 0.0, SPACECRAFT_INCLINATION_DEG, 0.0, 0.0, 0.0]
        )
        self.spacecraft = self._propagate(self.spacecraft_oe)
        self.debris_arg_latitude_deg = self._solve_debris_phase()
        self.debris = self._propagate(self._debris_oe(self.debris_arg_latitude_deg))

    def _debris_oe(self, arg_latitude_deg: float) -> np.ndarray:
        """Debris Keplerian elements `[a, e, i, RAAN, argp, M]` (m, deg) for a start argument of latitude (deg)."""
        return np.array(
            [
                self.semi_major_axis_m,
                0.0,
                SPACECRAFT_INCLINATION_DEG + DEBRIS_INCLINATION_OFFSET_DEG,
                DEBRIS_RAAN_OFFSET_DEG,
                0.0,
                arg_latitude_deg,
            ]
        )

    def _propagate(self, oe: np.ndarray) -> bh.NumericalOrbitPropagator:
        """Build a propagator from Keplerian elements `[a, e, i, RAAN, argp, M]` (m, deg) and run it over the scenario span."""
        state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
        prop = (
            bh.NumericalOrbitPropagator.builder(self.epoch, state, self.force_config)
            .propagation_config(self.propagation_config)
            .build()
        )
        prop.propagate_to(self.epoch + self.duration_s)
        return prop

    def _solve_debris_phase(self, tol_rad: float = 1e-10, max_iter: int = 10) -> float:
        """
            _solve_debris_phase(tol_rad=1e-10, max_iter=10) -> float

        Fixed-point iteration on the debris start argument of latitude.

          - `tol_rad` — convergence tolerance on the along-track phase error (rad)
          - `max_iter` — iteration cap

        Returns the debris argument of latitude at `epoch` (deg).
        """
        u_deg = 0.0
        for _ in range(max_iter):
            debris = self._propagate(self._debris_oe(u_deg))

            def plane_offset(t_s: float, debris=debris) -> float:
                r_sc = self.spacecraft.state_eci(self.epoch + t_s)[:3]
                x_deb = debris.state_eci(self.epoch + t_s)
                h = np.cross(x_deb[:3], x_deb[3:])
                return float(r_sc @ h / np.linalg.norm(h))

            t_c = self._nearest_root(
                plane_offset, self.target_tca_s, half_width_s=3600.0
            )

            r_sc = self.spacecraft.state_eci(self.epoch + t_c)[:3]
            x_deb = debris.state_eci(self.epoch + t_c)
            h_hat = np.cross(x_deb[:3], x_deb[3:])
            h_hat /= np.linalg.norm(h_hat)
            lead_rad = np.arctan2(np.cross(x_deb[:3], r_sc) @ h_hat, x_deb[:3] @ r_sc)
            err_rad = lead_rad - self.target_miss_m / np.linalg.norm(r_sc)
            u_deg += np.degrees(err_rad)
            if abs(err_rad) < tol_rad:
                return u_deg
        raise RuntimeError(f"debris phasing did not converge in {max_iter} iterations")

    @staticmethod
    def _nearest_root(
        f, t_center_s: float, half_width_s: float, dt_s: float = 20.0
    ) -> float:
        """Root of `f(t)` (t in s) in `t_center_s ± half_width_s` nearest `t_center_s`, bracketed on a `dt_s` grid (s)."""
        ts = np.arange(t_center_s - half_width_s, t_center_s + half_width_s, dt_s)
        fs = np.array([f(t) for t in ts])
        brackets = np.flatnonzero(np.sign(fs[:-1]) != np.sign(fs[1:]))
        if brackets.size == 0:
            raise RuntimeError(
                "spacecraft does not cross the debris plane near the target TCA"
            )
        roots = [brentq(f, ts[i], ts[i + 1], xtol=1e-6) for i in brackets]
        return min(roots, key=lambda t: abs(t - t_center_s))

    def _epochs(self, times_s: np.ndarray) -> list:
        return [self.epoch + float(t) for t in times_s]

    def relative_state_history(
        self, dt_s: float = 10.0, times_s: np.ndarray | None = None
    ):
        """
            relative_state_history(dt_s=10.0, times_s=None) -> (np.ndarray, np.ndarray)

        Debris state relative to the spacecraft in the spacecraft RTN frame, via
        `brahe.state_eci_to_rtn`.

          - `dt_s` — sample spacing over the whole span when `times_s` is not given (s)
          - `times_s` — explicit sample times after `epoch` (s)

        Returns `(times_s, rel_rtn)`: sample times (s) and an `(n, 6)` array
        `[R, T, N, dR, dT, dN]` (m, m/s).
        """
        if times_s is None:
            times_s = np.arange(0.0, self.duration_s + 0.5 * dt_s, dt_s)
        times_s = np.asarray(times_s, dtype=float)
        epochs = self._epochs(times_s)
        x_sc = np.asarray(self.spacecraft.states_eci(epochs))
        x_deb = np.asarray(self.debris.states_eci(epochs))
        return times_s, bh.state_eci_to_rtn(x_sc, x_deb)

    def range_m(self, t_s: float) -> float:
        """range_m(t_s) -> float — spacecraft-debris distance at `t_s` after `epoch` (s), in m."""
        e = self.epoch + float(t_s)
        return float(
            np.linalg.norm(
                self.spacecraft.state_eci(e)[:3] - self.debris.state_eci(e)[:3]
            )
        )

    def find_tca(self, dt_s: float = 5.0) -> ClosestApproach:
        """
            find_tca(dt_s=5.0) -> ClosestApproach

        Global closest approach over the propagated span.

          - `dt_s` — coarse sampling step (s)

        Every local minimum of the sampled range is refined with a bounded scalar
        minimization over one step either side, and the smallest refined minimum wins.

        Returns a `ClosestApproach` (s, m, RTN state in m and m/s).

        NOTE: at ~700 m/s relative speed a 5 s step spans ~3.5 km, so coarse samples
        misrank nearby passes; only the refined values are compared.
        """
        times_s = np.arange(0.0, self.duration_s, dt_s)
        _, rel = self.relative_state_history(times_s=times_s)
        rng = np.linalg.norm(rel[:, :3], axis=1)
        candidates = argrelmin(rng)[0]
        if candidates.size == 0:
            candidates = np.array([int(np.argmin(rng))])

        best_t, best_d = None, np.inf
        for k in candidates:
            lo = max(times_s[k] - dt_s, 0.0)
            hi = min(times_s[k] + dt_s, self.duration_s)
            res = minimize_scalar(
                self.range_m, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6}
            )
            if res.fun < best_d:
                best_t, best_d = float(res.x), float(res.fun)

        epoch = self.epoch + best_t
        rel_rtn = bh.state_eci_to_rtn(
            self.spacecraft.state_eci(epoch), self.debris.state_eci(epoch)
        )
        return ClosestApproach(
            epoch=epoch, t_s=best_t, miss_m=best_d, rel_rtn=np.asarray(rel_rtn)
        )

    def plot_conjunction(
        self,
        window_s: float = 30.0,
        theme: str = "light",
        tca: ClosestApproach | None = None,
    ):
        """
            plot_conjunction(window_s=30.0, theme="light", tca=None) -> matplotlib.figure.Figure

        Relative RTN trajectory of the debris about the spacecraft.

        Top: range over the full span. Bottom left: R, T, N components within
        `window_s` of TCA. Bottom right: the encounter path in the along-track /
        cross-track plane over the same window.

          - `window_s` — half-width of the zoomed window about TCA (s)
          - `theme` — `"light"` or `"dark"`; see `utils.plotting.apply_style`
          - `tca` — a precomputed `find_tca()` result; computed if omitted

        Returns the figure. Save with `utils.plotting.save_figure` for PDF/SVG/PNG.
        """
        import matplotlib.pyplot as plt

        from spacecraft_belief_reachability.utils.plotting import apply_style

        colors = apply_style(theme)
        c_r, c_t, c_n = colors["series"]
        tca = tca or self.find_tca()

        t_all, rel_all = self.relative_state_history(dt_s=10.0)
        rng_km = np.linalg.norm(rel_all[:, :3], axis=1) / 1e3

        t_win = np.linspace(tca.t_s - window_s, tca.t_s + window_s, 601)
        _, rel_win = self.relative_state_history(times_s=t_win)
        tau = t_win - tca.t_s
        rel_km = rel_win[:, :3] / 1e3

        fig = plt.figure(figsize=(7.0, 5.4), layout="constrained")
        gs = fig.add_gridspec(2, 2)
        ax_rng = fig.add_subplot(gs[0, :])
        ax_cmp = fig.add_subplot(gs[1, 0])
        ax_tn = fig.add_subplot(gs[1, 1])

        ax_rng.semilogy(t_all / 3600.0, rng_km, color=c_r, lw=1.0)
        tca_marker = {
            "marker": "o",
            "ms": 7,
            "mfc": "none",
            "mew": 1.5,
            "color": colors["ink"],
            "ls": "none",
        }
        ax_rng.plot(tca.t_s / 3600.0, tca.miss_m / 1e3, **tca_marker)
        ax_rng.annotate(
            f"TCA: {tca.miss_m:.0f} m at {tca.t_s / 3600.0:.2f} h",
            xy=(tca.t_s / 3600.0, tca.miss_m / 1e3),
            xytext=(-12, 0),
            textcoords="offset points",
            ha="right",
            va="center",
        )
        ax_rng.set_xlim(0.0, self.duration_s / 3600.0)
        ax_rng.set_xlabel("Time from start (h)")
        ax_rng.set_ylabel("Range (km)")

        for i, (name, c) in enumerate(
            zip(("Radial", "Along-track", "Cross-track"), (c_r, c_t, c_n))
        ):
            ax_cmp.plot(tau, rel_km[:, i], color=c, label=name)
        ax_cmp.set_xlim(tau[0], tau[-1])
        ax_cmp.set_xlabel("Time from TCA (s)")
        ax_cmp.set_ylabel("Relative position (km)")
        ax_cmp.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.22),
            ncol=3,
            handlelength=1.2,
            columnspacing=0.9,
        )

        ax_tn.plot(rel_km[:, 1], rel_km[:, 2], color=c_n)
        ax_tn.plot(0.0, 0.0, "s", ms=6, color=colors["ink"])
        ax_tn.annotate(
            "Spacecraft",
            xy=(0.0, 0.0),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
        )
        ax_tn.plot(tca.rel_rtn[1] / 1e3, tca.rel_rtn[2] / 1e3, **tca_marker)
        ax_tn.annotate(
            "Debris at TCA",
            xy=(tca.rel_rtn[1] / 1e3, tca.rel_rtn[2] / 1e3),
            xytext=(-6, -10),
            textcoords="offset points",
            ha="right",
            va="top",
        )
        ax_tn.set_xlabel("Along-track (km)")
        ax_tn.set_ylabel("Cross-track (km)")
        t_lim = 2.0 * max(abs(rel_km[:, 1]).max(), tca.miss_m / 1e3)
        ax_tn.set_xlim(-t_lim, t_lim)
        return fig
