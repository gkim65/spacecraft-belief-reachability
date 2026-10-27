"""Spacecraft-debris conjunction scenarios propagated with brahe's numerical propagator."""

from __future__ import annotations

from dataclasses import dataclass

import brahe as bh
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import argrelmin

bh.initialize_eop()
bh.initialize_sw()


@dataclass(frozen=True)
class ObjectParams:
    """
    Physical parameters brahe's drag and SRP models read from the parameter vector.

      - `mass_kg` — mass (kg)
      - `drag_area_m2` — drag cross-section (m^2)
      - `cd` — drag coefficient (-)
      - `srp_area_m2` — SRP cross-section (m^2)
      - `cr` — reflectivity coefficient (-)
    """

    mass_kg: float
    drag_area_m2: float
    cd: float
    srp_area_m2: float
    cr: float

    def vector(self) -> np.ndarray:
        """vector() -> np.ndarray — `[mass, drag_area, Cd, srp_area, Cr]` in brahe's parameter order."""
        return np.array(
            [self.mass_kg, self.drag_area_m2, self.cd, self.srp_area_m2, self.cr]
        )


SPACECRAFT_PARAMS = ObjectParams(
    mass_kg=500.0, drag_area_m2=2.0, cd=2.2, srp_area_m2=2.0, cr=1.3
)
"""Illustrative small-satellite parameters (area/mass 0.004 m^2/kg)."""

DEBRIS_PARAMS = ObjectParams(
    mass_kg=5.0, drag_area_m2=0.05, cd=2.2, srp_area_m2=0.05, cr=1.3
)
"""Illustrative fragment parameters (area/mass 0.01 m^2/kg)."""


@dataclass(frozen=True)
class DebrisOrbit:
    """
    Debris orbit shape at TCA, relative to the spacecraft.

      - `inclination_deg` — debris inclination (deg)
      - `raan_offset_deg` — debris RAAN minus the spacecraft's osculating RAAN at the
        target TCA (deg)
      - `eccentricity` — debris eccentricity (-)
      - `semi_major_axis_m` — debris semi-major axis (m); must be `None` for a circular
        orbit, whose radius is set to pass through the encounter point
      - `rising` — for an eccentric orbit, place the encounter where the debris radius
        is increasing (true anomaly in (0, 180) deg) rather than decreasing

    NOTE: these are osculating elements at TCA, not at the start epoch. The debris
    state at the start epoch comes from propagating backward.
    """

    inclination_deg: float
    raan_offset_deg: float
    eccentricity: float = 0.0
    semi_major_axis_m: float | None = None
    rising: bool = True


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
    A local minimum of spacecraft-debris range and the relative state there.

      - `epoch` — time of the minimum (brahe `Epoch`)
      - `t_s` — time of the minimum after the scenario start epoch (s)
      - `miss_m` — range at the minimum (m)
      - `rel_rtn` — debris state relative to the spacecraft in the spacecraft RTN frame,
        `[R, T, N, dR, dT, dN]` (m, m/s)
    """

    epoch: bh.Epoch
    t_s: float
    miss_m: float
    rel_rtn: np.ndarray

    @property
    def relative_speed_m_s(self) -> float:
        """relative_speed_m_s -> float — relative speed at the minimum (m/s)."""
        return float(np.linalg.norm(self.rel_rtn[3:]))


class ConjunctionScenario:
    """
        ConjunctionScenario(debris_orbit, name="conjunction", epoch=None,
                            spacecraft_altitude_m=550e3, spacecraft_inclination_deg=51.6,
                            target_miss_m=750.0, miss_angle_deg=0.0, target_tca_s=86400.0,
                            duration_s=None, force_config=None, propagation_config=None,
                            spacecraft_params=SPACECRAFT_PARAMS, debris_params=DEBRIS_PARAMS)

    A spacecraft and a debris object constructed to meet at a chosen TCA and miss.

    The spacecraft starts circular at `epoch` and is propagated forward. TCA is the
    spacecraft's crossing of the debris plane nearest `target_tca_s`. There the
    debris is placed on `debris_orbit` at the spacecraft's position, then displaced by
    `target_miss_m` perpendicular to the relative velocity. The debris is propagated
    backward to `epoch` and then forward over the scenario span.

      - `debris_orbit` — debris orbit shape at TCA (`DebrisOrbit`)
      - `name` — label for figures and tables
      - `epoch` — start epoch (brahe `Epoch`); default 2026-01-01 00:00:00 UTC
      - `spacecraft_altitude_m` — spacecraft circular altitude above `R_EARTH` at `epoch` (m)
      - `spacecraft_inclination_deg` — spacecraft inclination (deg)
      - `target_miss_m` — miss distance at TCA (m)
      - `miss_angle_deg` — direction of the miss in the encounter plane, measured from
        the spacecraft radial direction projected into that plane (deg)
      - `target_tca_s` — time after `epoch` near which TCA is placed (s)
      - `duration_s` — propagation span after `epoch` (s); default `target_tca_s + 7200`
      - `force_config` — brahe `ForceModelConfig`; default `leo_default()`
        (30x30 gravity, NRLMSISE-00 drag, SRP, Sun and Moon)
      - `propagation_config` — brahe `NumericalPropagationConfig`; default
        `default_propagation_config()`
      - `spacecraft_params`, `debris_params` — drag/SRP parameters (`ObjectParams`)

    Attributes `spacecraft` and `debris` are the propagated `NumericalOrbitPropagator`s,
    and `tca_design_s` is the designed TCA after `epoch` (s).

    NOTE: the miss is exact only to first order in the relative motion; for the
    presets `find_tca()` agrees with `target_miss_m` to within a meter.

    NOTE: circular debris shares the spacecraft's period, so the two meet again near
    the plane crossings every half orbit (`close_approaches()` lists them). Eccentric
    debris has a different period, which makes the encounter isolated.
    """

    def __init__(
        self,
        debris_orbit: DebrisOrbit,
        name: str = "conjunction",
        epoch: bh.Epoch | None = None,
        spacecraft_altitude_m: float = 550e3,
        spacecraft_inclination_deg: float = 51.6,
        target_miss_m: float = 750.0,
        miss_angle_deg: float = 0.0,
        target_tca_s: float = 86400.0,
        duration_s: float | None = None,
        force_config: bh.ForceModelConfig | None = None,
        propagation_config: bh.NumericalPropagationConfig | None = None,
        spacecraft_params: ObjectParams = SPACECRAFT_PARAMS,
        debris_params: ObjectParams = DEBRIS_PARAMS,
    ):
        self.debris_orbit = debris_orbit
        self.name = name
        self.epoch = epoch or bh.Epoch.from_datetime(
            2026, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC
        )
        self.target_miss_m = target_miss_m
        self.miss_angle_deg = miss_angle_deg
        self.target_tca_s = target_tca_s
        self.duration_s = (
            duration_s if duration_s is not None else target_tca_s + 7200.0
        )
        self.force_config = force_config or bh.ForceModelConfig.leo_default()
        self.propagation_config = propagation_config or default_propagation_config()
        self.spacecraft_params = spacecraft_params
        self.debris_params = debris_params

        self.spacecraft_oe = np.array(
            [
                bh.constants.R_EARTH + spacecraft_altitude_m,
                0.0,
                spacecraft_inclination_deg,
                0.0,
                0.0,
                0.0,
            ]
        )
        x_sc0 = bh.state_koe_to_eci(self.spacecraft_oe, bh.AngleFormat.DEGREES)
        self.spacecraft = self._propagator(self.epoch, x_sc0, spacecraft_params)
        self.spacecraft.propagate_to(self.epoch + self.duration_s)

        self.tca_design_s, x_deb_tca = self._debris_state_at_tca()
        back = self._propagator(
            self.epoch + self.tca_design_s, x_deb_tca, debris_params
        )
        back.propagate_to(self.epoch)
        self.debris = self._propagator(self.epoch, back.current_state(), debris_params)
        self.debris.propagate_to(self.epoch + self.duration_s)

    @classmethod
    def crossing(cls, **kwargs) -> ConjunctionScenario:
        """crossing(**kwargs) -> ConjunctionScenario — circular debris, +3 deg inclination, +5 deg RAAN; ~0.7 km/s, repeat passes ~13 km."""
        return cls(DebrisOrbit(54.6, 5.0), name="Crossing", **kwargs)

    @classmethod
    def high_angle(cls, **kwargs) -> ConjunctionScenario:
        """high_angle(**kwargs) -> ConjunctionScenario — circular debris at 97.6 deg, +30 deg RAAN; ~7 km/s, repeat passes ~19 km."""
        return cls(DebrisOrbit(97.6, 30.0), name="High angle", **kwargs)

    @classmethod
    def eccentric_debris(cls, **kwargs) -> ConjunctionScenario:
        """eccentric_debris(**kwargs) -> ConjunctionScenario — crossing-plane debris on a 450 x 850 km orbit; ~0.7 km/s, next pass ~290 km."""
        r_p = bh.constants.R_EARTH + 450e3
        r_a = bh.constants.R_EARTH + 850e3
        orbit = DebrisOrbit(
            54.6,
            5.0,
            eccentricity=(r_a - r_p) / (r_a + r_p),
            semi_major_axis_m=0.5 * (r_a + r_p),
        )
        return cls(orbit, name="Eccentric debris", **kwargs)

    def _propagator(
        self, epoch: bh.Epoch, state: np.ndarray, params: ObjectParams
    ) -> bh.NumericalOrbitPropagator:
        """Build an unpropagated `NumericalOrbitPropagator` from an ECI state (m, m/s) at `epoch`."""
        builder = bh.NumericalOrbitPropagator.builder(
            epoch, state, self.force_config
        ).propagation_config(self.propagation_config)
        if self.force_config.requires_params():
            builder = builder.params(params.vector())
        return builder.build()

    def _debris_state_at_tca(self) -> tuple[float, np.ndarray]:
        """
            _debris_state_at_tca() -> (float, np.ndarray)

        Place the debris on `debris_orbit` at the spacecraft's debris-plane crossing.

        Returns `(t_tca_s, x_debris)`: the design TCA after `epoch` (s) and the debris
        ECI state there (m, m/s).
        """
        orbit = self.debris_orbit
        oe_sc = self.spacecraft.state_koe_osc(
            self.epoch + self.target_tca_s, bh.AngleFormat.DEGREES
        )
        raan = np.radians(oe_sc[3] + orbit.raan_offset_deg)
        inc = np.radians(orbit.inclination_deg)
        normal = np.array(
            [np.sin(raan) * np.sin(inc), -np.cos(raan) * np.sin(inc), np.cos(inc)]
        )
        node = np.array([np.cos(raan), np.sin(raan), 0.0])

        def plane_offset(t_s: float) -> float:
            return float(self.spacecraft.state_eci(self.epoch + t_s)[:3] @ normal)

        t_tca = _nearest_root(plane_offset, self.target_tca_s, half_width_s=3600.0)
        x_sc = self.spacecraft.state_eci(self.epoch + t_tca)
        r_sc = x_sc[:3]
        radius = float(np.linalg.norm(r_sc))
        arg_lat = np.arctan2(np.cross(node, r_sc) @ normal, node @ r_sc)

        if orbit.eccentricity == 0.0:
            if orbit.semi_major_axis_m is not None:
                raise ValueError(
                    "a circular DebrisOrbit takes its radius from the encounter; "
                    "leave semi_major_axis_m as None"
                )
            a, e, argp, true_anom = radius, 0.0, 0.0, arg_lat
        else:
            if orbit.semi_major_axis_m is None:
                raise ValueError("an eccentric DebrisOrbit needs semi_major_axis_m")
            a, e = orbit.semi_major_axis_m, orbit.eccentricity
            cos_nu = (a * (1.0 - e**2) / radius - 1.0) / e
            if abs(cos_nu) > 1.0:
                raise ValueError(
                    f"debris orbit (r_p={a * (1 - e) / 1e3:.1f} km, "
                    f"r_a={a * (1 + e) / 1e3:.1f} km) never reaches the spacecraft "
                    f"radius {radius / 1e3:.1f} km"
                )
            true_anom = np.arccos(cos_nu) * (1.0 if orbit.rising else -1.0)
            argp = arg_lat - true_anom

        mean_anom = bh.anomaly_true_to_mean(
            true_anom, e, angle_format=bh.AngleFormat.RADIANS
        )
        x_deb = bh.state_koe_to_eci(
            np.array([a, e, inc, raan, argp, mean_anom]), bh.AngleFormat.RADIANS
        )

        v_rel = x_deb[3:] - x_sc[3:]
        v_hat = v_rel / np.linalg.norm(v_rel)
        r_hat = r_sc / radius
        e1 = r_hat - (r_hat @ v_hat) * v_hat
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(v_hat, e1)
        phi = np.radians(self.miss_angle_deg)
        x_deb[:3] = r_sc + self.target_miss_m * (np.cos(phi) * e1 + np.sin(phi) * e2)
        return t_tca, x_deb

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
        times_s = self._times(dt_s, times_s)
        x_sc, x_deb = self.eci_history(times_s=times_s)
        return times_s, bh.state_eci_to_rtn(x_sc, x_deb)

    def eci_history(self, dt_s: float = 10.0, times_s: np.ndarray | None = None):
        """
            eci_history(dt_s=10.0, times_s=None) -> (np.ndarray, np.ndarray)

        Spacecraft and debris ECI states.

          - `dt_s` — sample spacing over the whole span when `times_s` is not given (s)
          - `times_s` — explicit sample times after `epoch` (s)

        Returns `(x_spacecraft, x_debris)`, each `(n, 6)` (m, m/s).
        """
        epochs = [self.epoch + float(t) for t in self._times(dt_s, times_s)]
        return (
            np.asarray(self.spacecraft.states_eci(epochs)),
            np.asarray(self.debris.states_eci(epochs)),
        )

    def _times(self, dt_s: float, times_s: np.ndarray | None) -> np.ndarray:
        if times_s is None:
            return np.arange(0.0, self.duration_s + 0.5 * dt_s, dt_s)
        return np.asarray(times_s, dtype=float)

    def range_m(self, t_s: float) -> float:
        """range_m(t_s) -> float — spacecraft-debris distance at `t_s` after `epoch` (s), in m."""
        e = self.epoch + float(t_s)
        return float(
            np.linalg.norm(
                self.spacecraft.state_eci(e)[:3] - self.debris.state_eci(e)[:3]
            )
        )

    def range_rate_m_s(self, t_s: float) -> float:
        """range_rate_m_s(t_s) -> float — rate of change of spacecraft-debris distance at `t_s` after `epoch` (s), in m/s."""
        e = self.epoch + float(t_s)
        dx = self.debris.state_eci(e) - self.spacecraft.state_eci(e)
        return float(dx[:3] @ dx[3:] / np.linalg.norm(dx[:3]))

    def close_approaches(
        self, max_range_m: float = np.inf, dt_s: float = 5.0
    ) -> list[ClosestApproach]:
        """
            close_approaches(max_range_m=inf, dt_s=5.0) -> list[ClosestApproach]

        Every local minimum of range over the propagated span, in time order.

          - `max_range_m` — keep only minima closer than this (m)
          - `dt_s` — coarse sampling step (s)

        Each sampled local minimum is refined to the zero of range rate within one step
        either side (bounded minimization of range if the rate does not change sign there).

        NOTE: at ~7 km/s relative speed a 5 s step spans ~35 km, so compare only refined
        values, never the coarse samples.
        """
        times_s = np.arange(0.0, self.duration_s, dt_s)
        _, rel = self.relative_state_history(times_s=times_s)
        rng = np.linalg.norm(rel[:, :3], axis=1)
        candidates = argrelmin(rng)[0]
        if candidates.size == 0:
            candidates = np.array([int(np.argmin(rng))])

        out = []
        for k in candidates:
            lo = max(times_s[k] - dt_s, 0.0)
            hi = min(times_s[k] + dt_s, self.duration_s)
            if self.range_rate_m_s(lo) < 0.0 < self.range_rate_m_s(hi):
                t = brentq(self.range_rate_m_s, lo, hi, xtol=1e-9)
            else:
                t = minimize_scalar(
                    self.range_m,
                    bounds=(lo, hi),
                    method="bounded",
                    options={"xatol": 1e-6},
                ).x
            miss = self.range_m(t)
            if miss < max_range_m:
                out.append(self._approach(float(t), miss))
        return out

    def find_tca(self, dt_s: float = 5.0) -> ClosestApproach:
        """
            find_tca(dt_s=5.0) -> ClosestApproach

        Global closest approach over the propagated span.

          - `dt_s` — coarse sampling step (s); see `close_approaches`

        Returns the `ClosestApproach` with the smallest miss.
        """
        return min(self.close_approaches(dt_s=dt_s), key=lambda c: c.miss_m)

    def _approach(self, t_s: float, miss_m: float) -> ClosestApproach:
        epoch = self.epoch + t_s
        rel_rtn = bh.state_eci_to_rtn(
            self.spacecraft.state_eci(epoch), self.debris.state_eci(epoch)
        )
        return ClosestApproach(
            epoch=epoch, t_s=t_s, miss_m=miss_m, rel_rtn=np.asarray(rel_rtn)
        )


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
