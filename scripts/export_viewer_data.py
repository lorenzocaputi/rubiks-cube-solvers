"""Write docs/solves.json: the first 40 scrambles of each set with every solver's solution."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PER_SET = 40
SETS = [
    ("random-3x3", "3x3x3, random states", "3x3x3", ["beginner", "thistlethwaite", "kociemba"]),
    ("short-3x3", "3x3x3, 14-move scrambles", "3x3x3", ["beginner", "thistlethwaite", "kociemba", "korf"]),
    ("random-2x2", "2x2x2, random states", "2x2x2", ["korf-2x2", "learned-2x2"]),
]
LABELS = {
    "beginner": "Beginner's method",
    "thistlethwaite": "Thistlethwaite",
    "kociemba": "Kociemba two-phase",
    "korf": "Korf IDA* (optimal)",
    "korf-2x2": "Korf IDA* (optimal)",
    "learned-2x2": "Learned cost-to-go + weighted A*",
}


def main() -> None:
    with open(ROOT / "results" / "solves.csv", newline="") as f:
        solves = [r for r in csv.DictReader(f) if r["config"] in ("-", "target=20", "weight=0.6,batch=20")]
    with open(ROOT / "results" / "scrambles.csv", newline="") as f:
        scrambles = list(csv.DictReader(f))
    out = {"sets": []}
    for set_id, label, puzzle, solvers in SETS:
        items = [s for s in scrambles if s["set"] == set_id][:PER_SET]
        entries = []
        for s in items:
            by_solver = {}
            for r in solves:
                if r["set"] == set_id and r["id"] == s["id"] and r["solver"] in solvers:
                    by_solver[r["solver"]] = {
                        "moves": r["solution"],
                        "htm": int(r["htm"]),
                        "qtm": int(r["qtm"]),
                        "seconds": float(r["cpu_seconds"]),
                        "nodes": int(r["nodes"]),
                    }
            entries.append({"id": int(s["id"]), "scramble": s["scramble"], "solves": by_solver})
        out["sets"].append(
            {
                "id": set_id,
                "label": label,
                "puzzle": puzzle,
                "solvers": [{"id": k, "label": LABELS[k]} for k in solvers],
                "scrambles": entries,
            }
        )
    path = ROOT / "docs" / "solves.json"
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote docs/solves.json ({path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
