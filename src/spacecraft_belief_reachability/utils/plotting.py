"""Shared figure styling: light/dark themes, Computer Modern typography, vector export."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl

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
