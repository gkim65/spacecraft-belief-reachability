"""Figure styling (light/dark themes, Computer Modern, vector export) and scenario plots."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import brahe as bh
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from spacecraft_belief_reachability.simulation import (
        ClosestApproach,
        ConjunctionScenario,
    )

THEMES = {
    "light": {
        "ink": "#0b0b0b",
        "ink_muted": "#52514e",
        "grid": "#d9d8d4",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    },
    "dark": {
        "ink": "#f0efec",
        "ink_muted": "#c3c2b7",
        "grid": "#3a3a38",
        "series": ("#3987e5", "#d95926", "#199e70"),
    },
}
"""Per-theme ink, grid and categorical series colors (hex). Series order is fixed: blue, orange, aqua."""


def apply_style(theme: str = "light", font_size_pt: float = 11.0) -> dict:
    """
        apply_style(theme="light", font_size_pt=11.0) -> dict

    Set matplotlib rcParams for the given theme with Computer Modern serif text.

      - `theme` — `"light"` (dark ink, for white paper) or `"dark"` (light ink, for black slides)
      - `font_size_pt` — base font size at the figure's native size (pt)

    Returns the theme's color dict from `THEMES`.

    NOTE: uses LaTeX (`text.usetex`) when `latex` is on PATH, otherwise mathtext's
    Computer Modern; label strings must be valid in both (escape `%`, `&`, `_`).
    """
    colors = THEMES[theme]
    usetex = shutil.which("latex") is not None
    mpl.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": font_size_pt,
            "axes.labelsize": font_size_pt,
            "axes.titlesize": font_size_pt,
            "xtick.labelsize": font_size_pt - 1,
            "ytick.labelsize": font_size_pt - 1,
            "legend.fontsize": font_size_pt - 1,
            "text.color": colors["ink"],
            "axes.labelcolor": colors["ink"],
            "axes.edgecolor": colors["ink_muted"],
            "xtick.color": colors["ink_muted"],
            "ytick.color": colors["ink_muted"],
            "grid.color": colors["grid"],
            "axes.grid": True,
            "grid.linewidth": 0.6,
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "legend.frameon": False,
            "lines.linewidth": 1.5,
        }
    )
    return colors


def save_figure(
    fig, stem: str | Path, formats=("pdf", "svg", "png"), transparent: bool = True
) -> list[Path]:
    """
        save_figure(fig, stem, formats=("pdf", "svg", "png"), transparent=True) -> list[Path]

    Save a figure under one stem in several formats.

      - `fig` — the matplotlib figure
      - `stem` — output path without extension; parent directories are created
      - `formats` — file extensions to write
      - `transparent` — save with a transparent background so the figure sits on any slide or page

    Returns the written paths.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, transparent=transparent, bbox_inches="tight", dpi=300)
        paths.append(path)
    return paths


def _tca_marker(colors: dict) -> dict:
    return {
        "marker": "o",
        "ms": 7,
        "mfc": "none",
        "mew": 1.5,
        "color": colors["ink"],
        "ls": "none",
    }


def _range_history(scenario, dt_s: float = 10.0, dense_dt_s: float = 0.02):
    """Relative RTN history on a `dt_s` grid (s), resampled every `dense_dt_s` (s) within one step of each sampled range minimum."""
    t, rel = scenario.relative_state_history(dt_s=dt_s)
    rng = np.linalg.norm(rel[:, :3], axis=1)
    mins = np.flatnonzero((rng[1:-1] < rng[:-2]) & (rng[1:-1] < rng[2:])) + 1
    dense = [np.arange(t[k] - dt_s, t[k] + dt_s, dense_dt_s) for k in mins]
    t = np.unique(np.concatenate([t, *dense]))
    t = t[(t >= 0.0) & (t <= scenario.duration_s)]
    return scenario.relative_state_history(times_s=t)


def plot_conjunction(
    scenario: ConjunctionScenario,
    window_km: float = 20.0,
    theme: str = "light",
    tca: ClosestApproach | None = None,
):
    """
        plot_conjunction(scenario, window_km=20.0, theme="light", tca=None) -> Figure

    Relative RTN trajectory of the debris about the spacecraft.

    Top: range over the full span. Bottom left: R, T, N components near TCA. Bottom
    right: the encounter geometry — distance along the TCA relative velocity against
    offset along the TCA miss direction.

      - `scenario` — any object with `relative_state_history`, `find_tca` and `duration_s`
        (a `ConjunctionScenario`)
      - `window_km` — the zoomed panels span the time the debris takes to cover this
        distance either side of TCA at the TCA relative speed (km)
      - `theme` — `"light"` or `"dark"`; see `apply_style`
      - `tca` — a precomputed `find_tca()` result; computed if omitted

    Returns the figure.
    """
    colors = apply_style(theme)
    c_r, c_t, c_n = colors["series"]
    tca = tca or scenario.find_tca()
    marker = _tca_marker(colors)

    t_all, rel_all = _range_history(scenario)
    rng_km = np.linalg.norm(rel_all[:, :3], axis=1) / 1e3

    window_s = window_km * 1e3 / tca.relative_speed_m_s
    t_win = np.linspace(tca.t_s - window_s, tca.t_s + window_s, 601)
    _, rel_win = scenario.relative_state_history(times_s=t_win)
    tau = t_win - tca.t_s
    rel_km = rel_win[:, :3] / 1e3

    fig = plt.figure(figsize=(7.0, 5.4), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    ax_rng = fig.add_subplot(gs[0, :])
    ax_cmp = fig.add_subplot(gs[1, 0])
    ax_enc = fig.add_subplot(gs[1, 1])

    ax_rng.semilogy(t_all / 3600.0, rng_km, color=c_r, lw=1.0)
    ax_rng.plot(tca.t_s / 3600.0, tca.miss_m / 1e3, **marker)
    ax_rng.annotate(
        f"TCA: {tca.miss_m:.0f} m at {tca.t_s / 3600.0:.2f} h",
        xy=(tca.t_s / 3600.0, tca.miss_m / 1e3),
        xytext=(-12, 0),
        textcoords="offset points",
        ha="right",
        va="center",
    )
    ax_rng.set_xlim(0.0, scenario.duration_s / 3600.0)
    ax_rng.set_xlabel("Time from start (h)")
    ax_rng.set_ylabel("Range (km)")

    for i, (label, c) in enumerate(
        zip(("Radial", "Along-track", "Cross-track"), (c_r, c_t, c_n))
    ):
        ax_cmp.plot(tau, rel_km[:, i], color=c, label=label)
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

    miss_hat = tca.rel_rtn[:3] / np.linalg.norm(tca.rel_rtn[:3])
    v_hat = tca.rel_rtn[3:] / np.linalg.norm(tca.rel_rtn[3:])
    ax_enc.plot(
        rel_win[:, :3] @ v_hat / 1e3,
        rel_win[:, :3] @ miss_hat,
        color=colors["ink_muted"],
    )
    ax_enc.plot(0.0, 0.0, "s", ms=6, color=colors["ink"])
    ax_enc.annotate(
        "Spacecraft",
        xy=(0.0, 0.0),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        va="top",
    )
    ax_enc.plot(0.0, tca.miss_m, **marker)
    ax_enc.annotate(
        "Debris at TCA",
        xy=(0.0, tca.miss_m),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
    )
    ax_enc.set_xlim(-window_km, window_km)
    ax_enc.set_ylim(-1.0 * tca.miss_m, 2.0 * tca.miss_m)
    ax_enc.set_xlabel("Along relative velocity (km)")
    ax_enc.set_ylabel("Along miss direction (m)")
    return fig


def plot_orbits_3d(
    scenario: ConjunctionScenario,
    theme: str = "light",
    tca: ClosestApproach | None = None,
    span_s: float | None = None,
):
    """
        plot_orbits_3d(scenario, theme="light", tca=None, span_s=None) -> Figure

    Both orbits in ECI around a wireframe Earth, with the TCA position marked. The view
    looks down between the spacecraft orbit normal and the TCA direction.

      - `scenario` — any object with `eci_history`, `find_tca` and `spacecraft`
        (a `ConjunctionScenario`)
      - `theme` — `"light"` or `"dark"`; see `apply_style`
      - `tca` — a precomputed `find_tca()` result; computed if omitted
      - `span_s` — time plotted, centered on TCA (s); default one spacecraft orbit

    Returns the figure.
    """
    colors = apply_style(theme)
    c_sc, c_deb, _ = colors["series"]
    tca = tca or scenario.find_tca()
    if span_s is None:
        a = scenario.spacecraft.state_koe_osc(tca.epoch, bh.AngleFormat.DEGREES)[0]
        span_s = bh.orbital_period(a)
    t = np.linspace(tca.t_s - 0.5 * span_s, tca.t_s + 0.5 * span_s, 721)
    t = t[(t >= 0.0) & (t <= scenario.duration_s)]
    x_sc, x_deb = scenario.eci_history(times_s=t)
    r_tca = scenario.spacecraft.state_eci(tca.epoch)[:3]

    fig = plt.figure(figsize=(6.0, 5.4), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("ortho")

    u, v = np.mgrid[0 : 2 * np.pi : 37j, 0 : np.pi : 19j]
    re = bh.constants.R_EARTH / 1e3
    ax.plot_wireframe(
        re * np.cos(u) * np.sin(v),
        re * np.sin(u) * np.sin(v),
        re * np.cos(v),
        color=colors["grid"],
        lw=0.4,
    )
    ax.plot(*(x_sc[:, :3].T / 1e3), color=c_sc, label="Spacecraft")
    ax.plot(*(x_deb[:, :3].T / 1e3), color=c_deb, label="Debris")
    ax.plot(*(r_tca / 1e3), **_tca_marker(colors), label="TCA")

    lim = 1.05 * np.abs(np.vstack([x_sc[:, :3], x_deb[:, :3]])).max() / 1e3
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("$x$ (km)")
    ax.set_ylabel("$y$ (km)")
    ax.set_zlabel("$z$ (km)")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
    x_tca = scenario.spacecraft.state_eci(tca.epoch)
    h_hat = np.cross(x_tca[:3], x_tca[3:])
    h_hat /= np.linalg.norm(h_hat)
    look = 0.8 * h_hat + 0.6 * r_tca / np.linalg.norm(r_tca)
    ax.view_init(
        elev=np.degrees(np.arcsin(look[2] / np.linalg.norm(look))),
        azim=np.degrees(np.arctan2(look[1], look[0])),
    )
    fig.legend(loc="outside lower center", ncol=3)
    return fig
