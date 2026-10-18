"""A learned cost-to-go and batch weighted A* on the 2x2x2, after DeepCubeA.

The network maps a state (24 stickers, one-hot over 6 colours) to an estimate of
its distance to solved.  It is trained by approximate value iteration without any
solved examples beyond the goal itself: states come from random walks of 1 to K
moves from solved, and each target is ``min over moves (1 + J'(child))`` with
``J'(solved) = 0``, where ``J'`` is a frozen copy of the network that is refreshed
when the training loss falls below a threshold, or after ``max_wait`` steps.

Search is batch weighted A* (BWAS): each step expands the ``batch`` open nodes
with the lowest ``f = weight * g + h`` and scores all their children with one
network call; it stops when the solved state is selected for expansion.  The
weight 0.6 is the value DeepCubeA used for the 3x3x3; the batch is 20 instead of
10,000 because the 2x2x2 is small.  The network is also much smaller (three
hidden layers of 512 instead of 5,000 and 1,000 units plus four residual blocks)
and trains for 25 minutes on 3 CPU threads instead of 36 hours on GPUs.

The BFS distance table is used only to evaluate the network and the solutions,
never in training or search.

Inference runs in numpy; only training needs torch.
"""

from __future__ import annotations

import heapq
import json
import time
from pathlib import Path

import numpy as np
from numba import njit

from ..cube import cube2
from ..cube.facelet import CORNER_COLORS
from .base import Solution

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "cost_to_go_2x2.npz"
_PIECE_COLORS = np.array([["URFDLB".index(c) for c in colors] for colors in CORNER_COLORS], dtype=np.int64)
N_INPUT = 24 * 6


@njit(cache=True)
def encode(indices):
    """One-hot sticker colours, shape (len(indices), 144)."""
    out = np.zeros((indices.shape[0], N_INPUT), dtype=np.float32)
    cp = np.empty(8, dtype=np.int8)
    co = np.empty(8, dtype=np.int8)
    for b in range(indices.shape[0]):
        cube2.decode(indices[b], cp, co)
        for i in range(8):
            for n in range(3):
                sticker = 3 * i + (n + co[i]) % 3
                out[b, 6 * sticker + _PIECE_COLORS[cp[i], n]] = 1.0
    return out


@njit(cache=True)
def children(indices, perm_table, twist_table):
    """Successor indices, shape (len(indices), 9)."""
    out = np.empty((indices.shape[0], 9), dtype=np.int64)
    for b in range(indices.shape[0]):
        p, t = indices[b] // cube2.N_TWIST, indices[b] % cube2.N_TWIST
        for m in range(9):
            out[b, m] = perm_table[p, m] * cube2.N_TWIST + twist_table[t, m]
    return out


@njit(cache=True)
def random_walks(seed, count, max_length, perm_table, twist_table):
    """End states of ``count`` random walks from solved, each of 1..max_length moves."""
    np.random.seed(seed)
    out = np.empty(count, dtype=np.int64)
    for b in range(count):
        length = np.random.randint(1, max_length + 1)
        p, t = 0, 0
        for _ in range(length):
            m = np.random.randint(9)
            p, t = perm_table[p, m], twist_table[t, m]
        out[b] = p * cube2.N_TWIST + t
    return out


class CostToGo:
    """Numpy forward pass of the trained MLP."""

    def __init__(self, path: Path | str = MODEL_PATH) -> None:
        data = np.load(path)
        n_layers = int(data["n_layers"])
        self.weights = [data[f"w{i}"].astype(np.float32) for i in range(n_layers)]
        self.biases = [data[f"b{i}"].astype(np.float32) for i in range(n_layers)]
        self.meta = json.loads(str(data["meta"]))

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        x = encode(np.asarray(indices, dtype=np.int64))
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if k < len(self.weights) - 1:
                np.maximum(x, 0, out=x)
        h = x[:, 0]
        h[np.asarray(indices) == cube2.SOLVED_INDEX] = 0.0
        return h


class LearnedSolver:
    name = "learned-2x2"
    label = "Learned cost-to-go + weighted A*"
    puzzle = "2x2x2"

    def __init__(self, weight: float = 0.6, batch: int = 20, model: CostToGo | None = None) -> None:
        self.weight = weight
        self.batch = batch
        self.model = model or CostToGo()
        self.perm_table, self.twist_table = cube2.move_tables()

    def solve(self, index: int) -> Solution:
        start = time.perf_counter()
        g = {index: 0}
        parent: dict[int, tuple[int, int]] = {}
        h0 = float(self.model(np.array([index]))[0])
        heap = [(h0, 0, index)]
        tie = 1
        closed: set[int] = set()
        expanded = 0
        while heap:
            batch = []
            while heap and len(batch) < self.batch:
                _, _, node = heapq.heappop(heap)
                if node in closed:
                    continue
                if node == cube2.SOLVED_INDEX:
                    return self._result(node, parent, expanded, start)
                closed.add(node)
                batch.append(node)
            if not batch:
                break
            expanded += len(batch)
            succ = children(np.array(batch, dtype=np.int64), self.perm_table, self.twist_table)
            fresh, fresh_g = [], []
            for node, row in zip(batch, succ):
                cost = g[node] + 1
                for m, child in enumerate(row.tolist()):
                    if g.get(child, 1 << 30) <= cost:
                        continue
                    g[child] = cost  # a cheaper path also reopens a closed node
                    parent[child] = (node, m)
                    closed.discard(child)
                    fresh.append(child)
                    fresh_g.append(cost)
            if fresh:
                h = self.model(np.array(fresh, dtype=np.int64))
                for child, cost, value in zip(fresh, fresh_g, h.tolist()):
                    heapq.heappush(heap, (self.weight * cost + value, tie, child))
                    tie += 1
        raise RuntimeError("search exhausted without reaching the solved state")

    @staticmethod
    def _result(node: int, parent: dict, expanded: int, start: float) -> Solution:
        moves = []
        while node in parent:
            node, m = parent[node]
            moves.append(m)
        return Solution(moves[::-1], expanded, time.perf_counter() - start)


# ---- training ---------------------------------------------------------------------------
def train(
    out_path: Path | str = MODEL_PATH,
    log_path: Path | str | None = None,
    minutes: float = 25.0,
    batch_size: int = 1000,
    hidden: tuple[int, ...] = (512, 512, 512),
    max_scramble: int = 14,
    lr: float = 1e-3,
    check_every: int = 100,
    loss_threshold: float = 0.05,
    max_wait: int = 500,
    seed: int = 0,
    threads: int = 3,
) -> dict:
    """Approximate value iteration (DAVI) for ``minutes`` of wall-clock time."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    perm_table, twist_table = cube2.move_tables()
    dist = cube2.distance_table()
    eval_rng = np.random.default_rng(seed + 1)
    eval_states = eval_rng.integers(cube2.N_STATES, size=20_000)
    eval_x = torch.from_numpy(encode(eval_states))
    eval_d = dist[eval_states].astype(np.float32)

    sizes = (N_INPUT, *hidden, 1)
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    target = nn.Sequential(*[type(layer)(*_layer_args(layer)) for layer in layers])
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log = []
    start = time.perf_counter()
    iteration = 0
    updates = 0
    last_update = 0
    window = []
    while time.perf_counter() - start < 60 * minutes:
        states = random_walks(seed * 1_000_003 + iteration, batch_size, max_scramble, perm_table, twist_table)
        succ = children(states, perm_table, twist_table)
        with torch.no_grad():
            h = target(torch.from_numpy(encode(succ.reshape(-1)))).view(batch_size, 9)
            h[torch.from_numpy(succ == cube2.SOLVED_INDEX)] = 0.0
            y = (1.0 + h).min(dim=1).values
            y[torch.from_numpy(states == cube2.SOLVED_INDEX)] = 0.0
        pred = model(torch.from_numpy(encode(states)))[:, 0]
        loss = torch.mean((pred - y) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        window.append(loss.item())
        iteration += 1
        if iteration % check_every == 0:
            mean_loss = float(np.mean(window))
            window = []
            if mean_loss < loss_threshold or iteration - last_update >= max_wait:
                target.load_state_dict(model.state_dict())
                updates += 1
                last_update = iteration
            with torch.no_grad():
                est = model(eval_x)[:, 0].numpy()
            row = {
                "iteration": iteration,
                "seconds": round(time.perf_counter() - start, 1),
                "loss": round(mean_loss, 5),
                "target_updates": updates,
                "eval_mae": round(float(np.mean(np.abs(est - eval_d))), 4),
                "eval_exact": round(float(np.mean(np.rint(est) == eval_d)), 4),
            }
            log.append(row)
            print(row, flush=True)

    arrays = {}
    linears = [layer for layer in model if isinstance(layer, nn.Linear)]
    for i, layer in enumerate(linears):
        arrays[f"w{i}"] = layer.weight.detach().numpy().T.astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().numpy().astype(np.float32)
    meta = {
        "hidden": list(hidden),
        "batch_size": batch_size,
        "max_scramble": max_scramble,
        "iterations": iteration,
        "target_updates": updates,
        "minutes": minutes,
        "lr": lr,
        "loss_threshold": loss_threshold,
        "max_wait": max_wait,
        "seed": seed,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, n_layers=len(linears), meta=json.dumps(meta), **arrays)
    if log_path is not None:
        import csv

        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log[0]))
            writer.writeheader()
            writer.writerows(log)
    return meta


def _layer_args(layer) -> tuple:
    from torch import nn

    if isinstance(layer, nn.Linear):
        return (layer.in_features, layer.out_features)
    return ()
