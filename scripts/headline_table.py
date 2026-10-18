"""Print the README's headline table from results/summary.csv."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = [
    ("random-3x3", "beginner", "-", "Beginner's method (layer by layer)"),
    ("random-3x3", "thistlethwaite", "-", "Thistlethwaite, 4 phases"),
    ("random-3x3", "kociemba", "target=20", "Kociemba two-phase, stop at 20 moves"),
    ("short-3x3", "korf", "-", "Korf IDA* + pattern databases (optimal)"),
    ("random-2x2", "korf-2x2", "-", "Korf IDA* + pattern databases (optimal)"),
    ("random-2x2", "learned-2x2", "weight=0.6,batch=20", "Learned cost-to-go + weighted A*"),
]
SETS = {
    "random-3x3": "3x3x3, 1,000 random states",
    "short-3x3": "3x3x3, 300 scrambles of 14 moves",
    "random-2x2": "2x2x2, 1,000 random states",
}


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms" if seconds < 0.01 else f"{seconds * 1e3:.0f} ms"
    return f"{seconds:.1f} s"


def main() -> None:
    with open(ROOT / "results" / "summary.csv", newline="") as f:
        summary = {(r["set"], r["solver"], r["config"]): r for r in csv.DictReader(f)}
    print("| Solver | Scrambles | Mean moves (HTM) | Median CPU time | Solved |")
    print("|---|---|---:|---:|---:|")
    for set_name, solver, config, label in ROWS:
        r = summary[(set_name, solver, config)]
        moves = f"{float(r['mean_htm']):.1f}"
        if r["optimal_pct"] and solver == "learned-2x2":
            moves += f" ({float(r['optimal_pct']):.0f}% optimal)"
        print(
            f"| {label} | {SETS[set_name]} | {moves} | {fmt_time(float(r['median_seconds']))} "
            f"| {float(r['solved_pct']):.0f}% |"
        )


if __name__ == "__main__":
    main()
