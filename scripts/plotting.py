"""Colours, labels and loaders shared by the figure scripts."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Categorical slots in a fixed order (validated for colour-vision deficiency on both surfaces).
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
    },
}

SOLVERS = {
    "beginner": ("Beginner's method", 0),
    "thistlethwaite": ("Thistlethwaite", 1),
    "kociemba": ("Kociemba two-phase", 2),
    "korf": ("Korf IDA* (optimal)", 3),
    "korf-2x2": ("Korf IDA* (optimal)", 3),
    "learned-2x2": ("Learned cost-to-go + A*", 4),
}


def color(theme: dict, solver: str) -> str:
    return theme["series"][SOLVERS[solver][1]]


def apply_theme(theme: dict) -> None:
    plt.style.use(FIGURES / "style.mplstyle")
    plt.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "text.color": theme["ink"],
            "axes.labelcolor": theme["ink2"],
            "axes.titlecolor": theme["ink"],
            "axes.edgecolor": theme["axis"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "xtick.labelcolor": theme["ink2"],
            "ytick.labelcolor": theme["ink2"],
            "grid.color": theme["grid"],
        }
    )


def save_both(name: str, draw: Callable[[dict], plt.Figure]) -> None:
    """Draw a figure once per theme: ``figures/<name>.png`` and ``figures/<name>-dark.png``."""
    for mode, suffix in (("light", ""), ("dark", "-dark")):
        theme = THEMES[mode]
        apply_theme(theme)
        fig = draw(theme)
        fig.savefig(FIGURES / f"{name}{suffix}.png")
        plt.close(fig)
        print(f"wrote figures/{name}{suffix}.png")


def read_csv(name: str) -> list[dict]:
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))
