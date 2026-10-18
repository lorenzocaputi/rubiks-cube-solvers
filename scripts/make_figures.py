"""Draw every static figure from the CSVs in results/ (light and dark variants)."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
from plotting import SOLVERS, color, read_csv, save_both

SET_LABEL = {
    "random-3x3": "3x3x3, 1,000 random states",
    "short-3x3": "3x3x3, 300 scrambles of 14 moves",
    "random-2x2": "2x2x2, 1,000 random states",
}


def load_solves() -> list[dict]:
    rows = read_csv("solves.csv")
    for r in rows:
        r["htm"] = int(r["htm"])
        r["seconds"] = float(r["cpu_seconds"])  # CPU time per solve (the machine was shared)
        r["nodes"] = int(r["nodes"])
    return rows


def select(rows, set_name, solver, config=None):
    return [
        r
        for r in rows
        if r["set"] == set_name and r["solver"] == solver and (config is None or r["config"] == config)
    ]


def seconds_label(v: float, _=None) -> str:
    if v >= 1:
        return f"{v:g} s"
    if v >= 1e-3:
        return f"{v * 1e3:g} ms"
    return f"{v * 1e6:g} µs"


def short_seconds(v: float) -> str:
    """Three significant figures, e.g. 371 µs, 58.3 ms, 8.61 s."""
    for scale, unit in ((1, "s"), (1e-3, "ms"), (1e-6, "µs")):
        if v >= scale:
            return f"{v / scale:.3g} {unit}"
    return f"{v * 1e6:.3g} µs"


# ---- move-count histograms -----------------------------------------------------------
def histograms(rows):
    panels = [
        ("random-3x3", "beginner", None),
        ("random-3x3", "thistlethwaite", None),
        ("random-3x3", "kociemba", "target=20"),
        ("short-3x3", "korf", None),
        ("random-2x2", "korf-2x2", None),
        ("random-2x2", "learned-2x2", None),
    ]

    def draw(theme):
        fig, axes = plt.subplots(2, 3, figsize=(8, 5.0))
        for ax, (set_name, solver, config) in zip(axes.flat, panels):
            data = [r["htm"] for r in select(rows, set_name, solver, config)]
            values, counts = np.unique(data, return_counts=True)
            share = 100 * counts / counts.sum()
            lo, hi = values.min(), values.max()
            span = max(hi - lo, 10)
            center = (lo + hi) / 2
            ax.set_xlim(center - span * 0.6, center + span * 0.6)
            ax.bar(
                values, share, width=0.55 if hi - lo < 40 else 0.9, color=color(theme, solver), linewidth=0
            )
            mean = float(np.mean(data))
            top = share.max() * 1.3
            ax.set_ylim(0, top)
            ax.plot([mean, mean], [0, share.max() * 1.12], color=theme["ink2"], linewidth=0.9)
            ax.text(mean, share.max() * 1.13, f" mean {mean:.1f}", va="bottom", ha="left", fontsize=8.5)
            ax.set_title(SOLVERS[solver][0], fontsize=10, pad=16)
            ax.text(
                0,
                1.02,
                SET_LABEL[set_name],
                transform=ax.transAxes,
                fontsize=8,
                color=theme["muted"],
                va="bottom",
            )
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
            ax.locator_params(axis="x", integer=True, nbins=5)
            ax.locator_params(axis="y", nbins=4)
        for ax in axes[1]:
            ax.set_xlabel("solution length (face turns)")
        fig.suptitle(
            "Solution lengths in the half-turn metric (share of solves)",
            x=0.01,
            ha="left",
            fontsize=11.5,
            color=theme["ink"],
        )
        return fig

    save_both("move-histograms", draw)


# ---- time versus moves -------------------------------------------------------------------
def pareto(rows):
    def point(rs):
        t = np.array([r["seconds"] for r in rs])
        return np.median(t), np.percentile(t, 25), np.percentile(t, 75), np.mean([r["htm"] for r in rs])

    markers = {"beginner": "o", "thistlethwaite": "s", "kociemba": "D", "korf": "^"}

    def draw(theme):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
        # Left: every 3x3x3 solver on the scrambles all of them can finish.
        ax = axes[0]
        points = {}
        for solver in ("beginner", "thistlethwaite", "kociemba", "korf"):
            med, q1, q3, mean = point(select(rows, "short-3x3", solver))
            points[solver] = (med, mean)
            ax.plot([q1, q3], [mean, mean], color=color(theme, solver), linewidth=1.2, alpha=0.7)
            ax.scatter(
                [med],
                [mean],
                s=52,
                marker=markers[solver],
                color=color(theme, solver),
                edgecolor=theme["surface"],
                linewidth=1.5,
                zorder=3,
            )
        offsets = {"beginner": (8, -3), "thistlethwaite": (8, 6), "kociemba": (8, 6), "korf": (-20, -11)}
        align = {"korf": "right"}
        for solver, (med, mean) in points.items():
            ax.annotate(
                f"{SOLVERS[solver][0]}\n{mean:.1f} moves, {short_seconds(med)}",
                (med, mean),
                xytext=offsets[solver],
                textcoords="offset points",
                fontsize=8,
                ha=align.get(solver, "left"),
                color=theme["ink"],
                linespacing=1.25,
            )
        # Pareto front: Thistlethwaite -> Kociemba -> Korf (the beginner's method is dominated).
        front = sorted((points[s] for s in ("thistlethwaite", "kociemba", "korf")), key=lambda p: p[0])
        xs, ys = [front[0][0]], [front[0][1]]
        for x, y in front[1:]:
            xs += [x, x]
            ys += [ys[-1], y]
        ax.plot(xs, ys, color=theme["muted"], linewidth=0.9, zorder=1)
        ax.text(
            np.sqrt(xs[0] * xs[1]) * 2,
            ys[0] / 1.05,
            "Pareto front",
            ha="center",
            va="top",
            fontsize=7.5,
            color=theme["muted"],
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(8, 220)
        ax.set_xlim(1e-4, 200)
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_major_formatter(FuncFormatter(seconds_label))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylabel("mean solution length (face turns)")
        ax.set_xlabel("CPU time per solve (median; bar: middle 50%)")
        ax.set_title("3x3x3, 300 scrambles of 14 moves", fontsize=10)
        ax.grid(True, axis="both")

        # Right: Kociemba's stopping length on the same 200 random states.
        ax2 = axes[1]
        n_sweep = len(select(rows, "kociemba-targets", "kociemba", "target=19"))
        sweep = []
        for target in (30, 22, 21, 20, 19):
            set_name = "random-3x3" if target == 20 else "kociemba-targets"
            rs = [r for r in select(rows, set_name, "kociemba", f"target={target}") if int(r["id"]) < n_sweep]
            sweep.append((target, *point(rs)))
        c = color(theme, "kociemba")
        ax2.plot([s[1] for s in sweep], [s[4] for s in sweep], color=c, linewidth=1.5, zorder=2)
        for target, med, q1, q3, mean in sweep:
            ax2.plot([q1, q3], [mean, mean], color=c, linewidth=1.2, alpha=0.5)
            ax2.scatter(
                [med], [mean], s=46, marker="D", color=c, edgecolor=theme["surface"], linewidth=1.5, zorder=3
            )
            label = "first solution found" if target == 30 else f"stop at {target} or fewer"
            ax2.annotate(
                f"{label}: {mean:.2f}",
                (med, mean),
                xytext=(7, 4),
                textcoords="offset points",
                fontsize=8,
                color=theme["ink"],
            )
        ax2.set_xscale("log")
        ax2.set_xlim(3e-3, 30)
        ax2.set_ylim(19.4, 23.6)
        ax2.xaxis.set_major_formatter(FuncFormatter(seconds_label))
        ax2.xaxis.set_minor_locator(NullLocator())
        ax2.set_xlabel("CPU time per solve (median; bar: middle 50%)")
        ax2.set_ylabel("mean solution length (face turns)")
        ax2.set_title(f"Kociemba's stopping length, {n_sweep} random states", fontsize=10)
        ax2.grid(True, axis="both")
        fig.suptitle(
            "Solution length against search time", x=0.01, ha="left", fontsize=11.5, color=theme["ink"]
        )
        return fig

    save_both("time-vs-moves", draw)


# ---- 2x2x2 distance distribution ---------------------------------------------------------
def cube2_distribution():
    rows = read_csv("cube2_distribution.csv")
    d = np.array([int(r["distance"]) for r in rows if int(r["positions_htm"]) > 0])
    n = np.array([int(r["positions_htm"]) for r in rows if int(r["positions_htm"]) > 0])

    def draw(theme):
        fig, ax = plt.subplots(figsize=(8, 3.7))
        ax.bar(d, n, width=0.5, color=theme["series"][0], linewidth=0)
        ax.set_yscale("log")
        ax.set_ylim(0.5, n.max() * 8)
        ax.yaxis.set_minor_locator(NullLocator())
        for x, y in zip(d, n):
            ax.text(x, y * 1.3, f"{y:,}", ha="center", va="bottom", fontsize=8, color=theme["ink2"])
        ax.set_xticks(d)
        ax.set_xlabel("distance from solved (face turns, half-turn metric)")
        ax.set_ylabel("positions (log scale)")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        ax.set_title(
            f"All {n.sum():,} positions of the 2x2x2, by breadth-first search; none is more than {d.max()} "
            "moves from solved",
            fontsize=10,
            pad=18,
        )
        ax.text(
            0,
            1.02,
            "The counts match OEIS A079761 term for term.",
            transform=ax.transAxes,
            fontsize=8.5,
            color=theme["muted"],
            va="bottom",
        )
        return fig

    save_both("cube2-distances", draw)


# ---- Thistlethwaite subgroup chain --------------------------------------------------------
def _sci(x: int) -> str:
    exponent = len(str(x)) - 1
    return rf"${x / 10**exponent:.1f} \times 10^{{{exponent}}}$"


def thistlethwaite_chain(rows):
    phases = read_csv("thistlethwaite_phases.csv")
    lengths = np.array(
        [
            list(map(int, r["detail"].split("=")[1].split("/")))
            for r in select(rows, "random-3x3", "thistlethwaite")
        ]
    )
    cosets = [int(p["cosets"]) for p in phases]
    orders = [int(np.prod(cosets[i:], dtype=object)) for i in range(4)] + [1]
    labels = [_sci(o) if o > 1e6 else f"{o:,}" for o in orders]
    generators = [
        "U, D, R, L, F, B",
        "U, D, R, L, F2, B2",
        "U, D, R2, L2, F2, B2",
        "U2, D2, R2, L2, F2, B2",
        None,
    ]
    goals = [
        "orient the edges",
        "orient the corners;\nE-slice edges into the E slice",
        "every piece into its\nhalf-turn orbit",
        "solve with half turns",
    ]

    def draw(theme):
        fig, ax = plt.subplots(figsize=(8, 2.9))
        ax.set_xlim(0, 10)
        ax.set_ylim(0.15, 3.85)
        ax.axis("off")
        xs = np.linspace(0.85, 9.15, 5)
        for i, x in enumerate(xs):
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.8, 2.35),
                    1.6,
                    1.3,
                    boxstyle="round,pad=0,rounding_size=0.12",
                    facecolor=theme["surface"],
                    edgecolor=theme["axis"],
                    linewidth=1,
                )
            )
            ax.text(
                x, 3.33, f"G{i}", ha="center", va="center", fontsize=11, color=theme["ink"], weight="bold"
            )
            gens = f"<{generators[i]}>" if generators[i] else "solved cube only"
            ax.text(x, 2.97, gens, ha="center", va="center", fontsize=6.9, color=theme["ink2"])
            unit = "state" if orders[i] == 1 else "states"
            ax.text(
                x, 2.6, f"{labels[i]} {unit}", ha="center", va="center", fontsize=7.6, color=theme["muted"]
            )
        for i in range(4):
            ax.add_patch(
                FancyArrowPatch(
                    (xs[i] + 0.82, 3.0),
                    (xs[i + 1] - 0.82, 3.0),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    color=theme["series"][1],
                    linewidth=1.6,
                )
            )
            mid = (xs[i] + xs[i + 1]) / 2
            p = phases[i]
            ax.text(
                mid,
                2.05,
                f"phase {i + 1}",
                ha="center",
                va="top",
                fontsize=9,
                color=theme["ink"],
                weight="bold",
            )
            ax.text(
                mid, 1.72, goals[i], ha="center", va="top", fontsize=7.4, color=theme["ink2"], linespacing=1.3
            )
            ax.text(
                mid,
                1.05,
                f"{int(p['cosets']):,} cosets\nat most {p['max_moves']} moves\n"
                f"mean {lengths[:, i].mean():.1f} moves",
                ha="center",
                va="top",
                fontsize=7.4,
                color=theme["muted"],
                linespacing=1.35,
            )
        ax.set_title(
            "Thistlethwaite's subgroup chain; each phase is solved optimally from a full BFS table",
            fontsize=10,
        )
        return fig

    save_both("thistlethwaite-chain", draw)


# ---- learned heuristic against the truth --------------------------------------------------
def learned(rows):
    from rubiks_cube_solvers.cube import cube2
    from rubiks_cube_solvers.solvers.learned import CostToGo

    model = CostToGo()
    dist = cube2.distance_table()
    states = np.random.default_rng(5).integers(cube2.N_STATES, size=50_000)
    est = model(states)
    truth = dist[states]
    learned_rows = select(rows, "random-2x2", "learned-2x2")
    excess = np.array([r["htm"] - int(r["optimal_htm"]) for r in learned_rows])
    korf_nodes = np.median([r["nodes"] for r in select(rows, "random-2x2", "korf-2x2")])
    learned_nodes = np.median([r["nodes"] for r in learned_rows])

    def draw(theme):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
        ax = axes[0]
        ds = np.arange(truth.min(), truth.max() + 1)
        med = [np.median(est[truth == k]) for k in ds]
        lo = [np.percentile(est[truth == k], 5) for k in ds]
        hi = [np.percentile(est[truth == k], 95) for k in ds]
        c = color(theme, "learned-2x2")
        ax.fill_between(ds, lo, hi, color=c, alpha=0.18, linewidth=0)
        ax.plot(ds, ds, color=theme["muted"], linewidth=1)
        ax.plot(ds, med, color=c, marker="o", markersize=4.5, markeredgecolor=theme["surface"])
        ax.set_xlabel("true distance from solved (BFS)")
        ax.set_ylabel("network's estimate")
        ax.set_title("Learned cost-to-go: median, 5-95% range", fontsize=10)
        ax.text(
            ds[-1] - 0.4,
            ds[-1] + 0.15,
            "estimate = truth",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=theme["muted"],
        )
        ax.set_xticks(ds)
        ax.grid(True, axis="both")
        ax2 = axes[1]
        values, counts = np.unique(excess, return_counts=True)
        share = 100 * counts / counts.sum()
        ax2.bar(values, share, width=0.45, color=c, linewidth=0)
        for v, s in zip(values, share):
            ax2.text(v, s + 2, f"{s:.1f}%", ha="center", va="bottom", fontsize=8.5, color=theme["ink2"])
        ax2.set_xticks(values)
        ax2.set_xlim(values.min() - 0.7, values.max() + 0.7)
        ax2.set_ylim(0, 112)
        ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax2.set_xlabel("moves above the optimal solution")
        ax2.set_title("Weighted A* with it, 1,000 random states", fontsize=10)
        ax2.text(
            1.0,
            0.97,
            f"median nodes expanded:\n{learned_nodes:,.0f} (optimal IDA*: {korf_nodes:,.0f})",
            transform=ax2.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=theme["ink2"],
            linespacing=1.3,
        )
        return fig

    save_both("learned-vs-optimal", draw)


if __name__ == "__main__":
    solves = load_solves()
    histograms(solves)
    pareto(solves)
    cube2_distribution()
    thistlethwaite_chain(solves)
    learned(solves)
