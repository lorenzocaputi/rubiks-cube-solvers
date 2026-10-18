"""Run every solver on seeded scramble sets and write CSVs to ``results/``.

    python -m rubiks_cube_solvers.benchmark            # full run (about 40 min on 3 workers)
    python -m rubiks_cube_solvers.benchmark --quick    # small sets, small tables, for testing

Scramble sets:

* ``random-3x3``: 1,000 uniformly random 3x3x3 states (seed 2026).
* ``short-3x3``: 300 scrambles of 14 random face turns (seed 2027), short enough
  for the optimal solver.
* ``random-2x2``: 1,000 uniformly random 2x2x2 states (seed 2028).
* ``kociemba-targets``: the first 200 states of ``random-3x3`` again, solved by
  Kociemba with other stopping lengths, for the time-versus-moves trade-off.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cube import (
    cube2,
    format_moves,
    from_facelets,
    htm,
    invert,
    qtm,
    random_moves,
    random_state,
    to_facelets,
)
from .cube.cubie import SOLVED

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Task:
    set: str
    solver: str
    config: str
    id: int
    state: str  # facelet string (3x3x3) or state index (2x2x2)


_SOLVERS: dict[tuple[str, str], object] = {}
_QUICK = False


def _solver(name: str, config: str):
    key = (name, config)
    if key not in _SOLVERS:
        if name == "beginner":
            from .solvers.beginner import BeginnerSolver

            _SOLVERS[key] = BeginnerSolver()
        elif name == "thistlethwaite":
            from .solvers.thistlethwaite import ThistlethwaiteSolver

            _SOLVERS[key] = ThistlethwaiteSolver()
        elif name == "kociemba":
            from .solvers.kociemba import KociembaSolver

            _SOLVERS[key] = KociembaSolver(target=int(config.removeprefix("target=")))
        elif name == "korf":
            from .solvers.korf import KorfSolver

            _SOLVERS[key] = KorfSolver(quick=_QUICK)
        elif name == "korf-2x2":
            from .solvers.korf import Korf2Solver

            _SOLVERS[key] = Korf2Solver()
        elif name == "learned-2x2":
            from .solvers.learned import LearnedSolver

            weight, batch = (float(x.split("=")[1]) for x in config.split(","))
            _SOLVERS[key] = LearnedSolver(weight=weight, batch=int(batch))
        else:
            raise ValueError(name)
        # One untimed solve, so lazy tables and compiled code are loaded before timing starts.
        if name.endswith("2x2"):
            _SOLVERS[key].solve(cube2.apply_moves(cube2.SOLVED_INDEX, [0, 3, 6]))
        else:
            _SOLVERS[key].solve(SOLVED.apply([0, 3, 6]))
    return _SOLVERS[key]


def _init_worker(quick: bool) -> None:
    global _QUICK
    _QUICK = quick


def run_task(task: Task) -> dict:
    solver = _solver(task.solver, task.config)
    if task.set.endswith("2x2"):
        index = int(task.state)
        cpu = time.process_time()
        solution = solver.solve(index)
        cpu = time.process_time() - cpu
        solved = cube2.apply_moves(index, solution.moves) == cube2.SOLVED_INDEX
    else:
        cube = from_facelets(task.state)
        cpu = time.process_time()
        solution = solver.solve(cube)
        cpu = time.process_time() - cpu
        solved = cube.apply(solution.moves).is_solved()
    return {
        "set": task.set,
        "id": task.id,
        "solver": task.solver,
        "config": task.config,
        "solution": format_moves(solution.moves),
        "htm": htm(solution.moves),
        "qtm": qtm(solution.moves),
        "seconds": round(solution.seconds, 6),  # wall clock
        "cpu_seconds": round(cpu, 6),  # CPU time of this process, robust to a busy machine
        "nodes": solution.nodes,
        "solved": bool(solved),
        "detail": _detail(solution.info),
    }


def _detail(info: dict) -> str:
    """Solver-specific extras as ``key=value`` pairs, e.g. ``phases=4/8/9/10``."""
    parts = []
    for key, value in info.items():
        if isinstance(value, dict):
            value = "/".join(f"{k}:{v}" for k, v in value.items())
        elif isinstance(value, list):
            value = "/".join(map(str, value))
        parts.append(f"{key}={value}")
    return " ".join(parts)


def scramble_sets(quick: bool) -> dict[str, list[dict]]:
    n3, n_short, n2 = (20, 20, 50) if quick else (1000, 300, 1000)
    short_length = 8 if quick else 14
    rng = np.random.default_rng(2026)
    random3 = [{"id": i, "state": to_facelets(random_state(rng))} for i in range(n3)]
    rng = np.random.default_rng(2027)
    short = []
    for i in range(n_short):
        moves = random_moves(rng, short_length)
        short.append({"id": i, "state": to_facelets(SOLVED.apply(moves)), "scramble": format_moves(moves)})
    rng = np.random.default_rng(2028)
    random2 = []
    for i in range(n2):
        index = cube2.random_state(rng)
        random2.append({"id": i, "state": str(index)})
    return {"random-3x3": random3, "short-3x3": short, "random-2x2": random2}


def plan(sets: dict[str, list[dict]], quick: bool) -> list[Task]:
    tasks = []
    learned = "weight=0.6,batch=20"
    jobs = {
        "random-3x3": [("beginner", "-"), ("thistlethwaite", "-"), ("kociemba", "target=20")],
        "short-3x3": [("korf", "-"), ("beginner", "-"), ("thistlethwaite", "-"), ("kociemba", "target=20")],
        "random-2x2": [("korf-2x2", "-"), ("learned-2x2", learned)],
    }
    for name, solvers in jobs.items():
        for solver, config in solvers:
            tasks += [Task(name, solver, config, s["id"], s["state"]) for s in sets[name]]
    n_sweep = 10 if quick else 200
    for target in (30, 22, 21, 19):
        tasks += [
            Task("kociemba-targets", "kociemba", f"target={target}", s["id"], s["state"])
            for s in sets["random-3x3"][:n_sweep]
        ]
    # Slow tasks first, so the pool does not end on a long tail.
    slow = {"korf": 0, "kociemba": 1}
    tasks.sort(key=lambda t: (slow.get(t.solver, 2), t.config != "target=19"))
    return tasks


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_random_state_scrambles(sets: dict) -> None:
    """A random state comes without a move sequence.  For the viewer, each gets one: the
    inverse of the first solution Kociemba's search finds (3x3x3), or of an optimal
    solution read from the BFS table trying moves in reverse order (2x2x2), so the
    scramble is not simply the inverse of a benchmarked solution."""
    from .solvers.kociemba import KociembaSolver

    kociemba = KociembaSolver(target=30)
    for s in sets["random-3x3"]:
        s["scramble"] = format_moves(invert(kociemba.solve(from_facelets(s["state"])).moves))
    dist = cube2.distance_table()
    for s in sets["random-2x2"]:
        index, moves = int(s["state"]), []
        while dist[index] > 0:
            for m in range(8, -1, -1):
                nxt = cube2.apply_moves(index, [m])
                if dist[nxt] == dist[index] - 1:
                    moves.append(m)
                    index = nxt
                    break
        s["scramble"] = format_moves(invert(moves))


def cube2_distribution(out: Path) -> None:
    htm_dist = np.bincount(cube2.distance_table("htm"))
    qtm_dist = np.bincount(cube2.distance_table("qtm"))
    rows = [
        {
            "distance": d,
            "positions_htm": int(htm_dist[d]) if d < len(htm_dist) else 0,
            "positions_qtm": int(qtm_dist[d]) if d < len(qtm_dist) else 0,
        }
        for d in range(len(qtm_dist))
    ]
    write_csv(out / "cube2_distribution.csv", rows)


def thistlethwaite_phases(out: Path) -> None:
    from .solvers.thistlethwaite import ThistlethwaiteSolver

    solver = ThistlethwaiteSolver()
    rows = []
    for i, dist in enumerate(solver.dist):
        reachable = dist[dist >= 0]
        rows.append(
            {
                "phase": i + 1,
                "cosets": int(reachable.size),
                "max_moves": int(reachable.max()),
                "mean_moves_over_cosets": round(float(reachable.mean()), 3),
            }
        )
    write_csv(out / "thistlethwaite_phases.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (ROOT / ("results-quick" if args.quick else "results"))
    out.mkdir(parents=True, exist_ok=True)

    # Build every table once in this process, so workers only load them.
    _init_worker(args.quick)
    for name, config in [
        ("beginner", "-"),
        ("thistlethwaite", "-"),
        ("kociemba", "target=20"),
        ("korf", "-"),
        ("korf-2x2", "-"),
    ]:
        _solver(name, config)
    cube2.distance_table("htm")
    cube2.distance_table("qtm")

    sets = scramble_sets(args.quick)
    add_random_state_scrambles(sets)
    tasks = plan(sets, args.quick)
    load_before = os.getloadavg()
    print(f"{len(tasks)} solves on {args.workers} workers", flush=True)
    start = time.perf_counter()
    results = []
    ctx = mp.get_context("spawn")
    partial = open(out / "solves.partial.jsonl", "w")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.quick,)) as pool:
        for k, row in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1)):
            results.append(row)
            partial.write(json.dumps(row) + "\n")
            partial.flush()
            if (k + 1) % 500 == 0:
                print(f"{k + 1}/{len(tasks)} done, {time.perf_counter() - start:.0f} s", flush=True)
    wall = time.perf_counter() - start
    partial.close()
    (out / "solves.partial.jsonl").unlink()
    results.sort(key=lambda r: (r["set"], r["solver"], r["config"], r["id"]))

    dist2 = cube2.distance_table("htm")
    for r in results:
        r["optimal_htm"] = ""
        if r["set"] == "random-2x2":
            r["optimal_htm"] = int(dist2[int(sets["random-2x2"][r["id"]]["state"])])
    korf = {r["id"]: r["htm"] for r in results if r["set"] == "short-3x3" and r["solver"] == "korf"}
    for r in results:
        if r["set"] == "short-3x3":
            r["optimal_htm"] = korf[r["id"]]

    write_csv(out / "solves.csv", results)
    write_csv(
        out / "scrambles.csv",
        [{"set": name, **s} for name, items in sets.items() for s in items],
    )
    cube2_distribution(out)
    thistlethwaite_phases(out)
    write_summary(out, results)
    import numba

    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "workers": args.workers,
        "quick": args.quick,
        "solves": len(results),
        "wall_seconds": round(wall, 1),
        "load_average_1min_before_and_after": [round(load_before[0], 1), round(os.getloadavg()[0], 1)],
    }
    (out / "benchmark_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def cpu_name() -> str:
    import subprocess

    try:
        if platform.system() == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


def write_summary(out: Path, results: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        groups.setdefault((r["set"], r["solver"], r["config"]), []).append(r)
    rows = []
    for (name, solver, config), rs in sorted(groups.items()):
        moves = np.array([r["htm"] for r in rs])
        quarter = np.array([r["qtm"] for r in rs])
        seconds = np.array([r["cpu_seconds"] for r in rs])
        wall = np.array([r["seconds"] for r in rs])
        nodes = np.array([r["nodes"] for r in rs])
        row = {
            "set": name,
            "solver": solver,
            "config": config,
            "n": len(rs),
            "solved_pct": round(100 * np.mean([r["solved"] for r in rs]), 1),
            "mean_htm": round(float(moves.mean()), 2),
            "median_htm": float(np.median(moves)),
            "max_htm": int(moves.max()),
            "mean_qtm": round(float(quarter.mean()), 2),
            "median_seconds": round(float(np.median(seconds)), 5),
            "mean_seconds": round(float(seconds.mean()), 5),
            "max_seconds": round(float(seconds.max()), 4),
            "median_wall_seconds": round(float(np.median(wall)), 5),
            "median_nodes": float(np.median(nodes)),
            "mean_excess_htm": "",
            "optimal_pct": "",
        }
        if rs[0]["optimal_htm"] != "":
            excess = moves - np.array([r["optimal_htm"] for r in rs])
            row["mean_excess_htm"] = round(float(excess.mean()), 3)
            row["optimal_pct"] = round(100 * float(np.mean(excess == 0)), 1)
        rows.append(row)
    write_csv(out / "summary.csv", rows)


if __name__ == "__main__":
    main()
