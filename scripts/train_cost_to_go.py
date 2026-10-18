"""Train the 2x2x2 cost-to-go network with approximate value iteration (needs torch)."""

import argparse
from pathlib import Path

from rubiks_cube_solvers.solvers.learned import MODEL_PATH, train

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=25.0)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    parser.add_argument("--log", type=Path, default=ROOT / "results" / "training_log.csv")
    args = parser.parse_args()
    meta = train(args.out, args.log, minutes=args.minutes, threads=args.threads)
    print(meta)
