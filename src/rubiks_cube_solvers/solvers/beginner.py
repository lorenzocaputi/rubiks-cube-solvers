"""Layer-by-layer beginner's method, with the D face as the first layer.

1. Cross: the four D edges one at a time, each by the shortest sequence that keeps
   the edges already placed (read from small BFS tables; a person does this step by
   trial and error).
2. First-layer corners: bring the corner above its slot, repeat ``R U R' U'``.
3. Middle-layer edges: ``U R U' R' U' F' U F`` or its mirror.
4. Last-layer edge orientation: ``F R U R' U' F'`` and ``F U R U' R' F'``.
5. Last-layer corner orientation: Sune, ``R U R' U R U2 R'``.
6. Last-layer corner permutation: the A-perm ``R' F R' B2 R F' R' B2 R2``.
7. Last-layer edge permutation: the two U-perms.

Steps 4 to 7 pick the shortest combination of these algorithms and U turns by
breadth-first search over algorithm applications.  Each algorithm is used from
all four sides of the cube (relabelling F R B L).
"""

from __future__ import annotations

import time

import numpy as np

from ..cube.cubie import MOVE_EO, MOVE_EP, SOLVED, CubieCube
from ..cube.notation import parse, simplify
from .base import Solution

SIDES = "FRBL"
DFR_SLOT = (4, 7, 6, 5)  # DFR DRB DBL DLF: the front-right D corner seen from each side
URF_SLOT = (0, 3, 2, 1)
FR_SLOT = (8, 11, 10, 9)  # FR BR BL FL
CROSS_EDGES = (5, 4, 7, 6)  # DF DR DB DL

RIGHT_INSERT = "U R U' R' U' F' U F"
LEFT_INSERT = "U' L' U L U F U' F'"
SEXY = "R U R' U'"
POP_CORNER = "R U R'"
EDGE_OLL = ("F R U R' U' F'", "F U R U' R' F'")
SUNE = "R U R' U R U2 R'"
A_PERM = "R' F R' B2 R F' R' B2 R2"
U_PERMS = ("R U' R U R U R U' R' U' R2", "R2 U R U R' U' R' U' R' U R'")


def from_side(alg: str, side: int) -> list[int]:
    """The moves of ``alg`` performed with face ``SIDES[side]`` in front."""
    out = []
    for token in alg.split():
        face = token[0]
        if face in SIDES:
            token = SIDES[(SIDES.index(face) + side) % 4] + token[1:]
        out.extend(parse(token))
    return out


class _Alg:
    __slots__ = ("moves", "cube", "is_u_turn")

    def __init__(self, moves: list[int]) -> None:
        self.moves = moves
        self.cube = SOLVED.apply(moves)
        self.is_u_turn = all(m < 3 for m in moves)


_U_TURNS = [_Alg([m]) for m in (0, 1, 2)]


def _algs(alg: str) -> list[_Alg]:
    return [_Alg(from_side(alg, side)) for side in range(4)]


_MACROS = {
    "edge_oll": _U_TURNS + [a for alg in EDGE_OLL for a in _algs(alg)[:1]],
    "sune": _U_TURNS + _algs(SUNE)[:1],
    "a_perm": _U_TURNS + _algs(A_PERM),
    "u_perm": [a for alg in U_PERMS for a in _algs(alg)],
}


# ---- cross tables ---------------------------------------------------------------
def _edge_code_table() -> np.ndarray:
    """``table[2 * position + flip, move]`` for a single edge."""
    table = np.empty((24, 18), dtype=np.int64)
    for m in range(18):
        for i in range(12):
            j = MOVE_EP[m, i]
            for flip in range(2):
                table[2 * j + flip, m] = 2 * i + (flip + MOVE_EO[m, i]) % 2
    return table


_EDGE_CODES = _edge_code_table()
_CROSS_TABLES: list[np.ndarray] = []


def _cross_table(k: int) -> np.ndarray:
    """BFS distances over the codes of the first ``k`` cross edges (index in base 24)."""
    while len(_CROSS_TABLES) < k:
        n = len(_CROSS_TABLES) + 1
        size = 24**n
        dist = np.full(size, -1, dtype=np.int8)
        start = sum(2 * CROSS_EDGES[i] * 24**i for i in range(n))
        dist[start] = 0
        frontier = np.array([start])
        depth = 0
        powers = 24 ** np.arange(n)
        while frontier.size:
            digits = (frontier[:, None] // powers) % 24
            nxt = []
            for m in range(18):
                index = (_EDGE_CODES[digits, m] * powers).sum(axis=1)
                nxt.append(index[dist[index] < 0])
            frontier = np.unique(np.concatenate(nxt))
            depth += 1
            dist[frontier] = depth
        _CROSS_TABLES.append(dist)
    return _CROSS_TABLES[k - 1]


def _cross_index(cube: CubieCube, k: int) -> int:
    index = 0
    for i in range(k):
        position = cube.ep.index(CROSS_EDGES[i])
        index += (2 * position + cube.eo[position]) * 24**i
    return index


# ---- solver -----------------------------------------------------------------------
class BeginnerSolver:
    name = "beginner"
    label = "Beginner's method"
    puzzle = "3x3x3"

    def solve(self, cube: CubieCube) -> Solution:
        start = time.perf_counter()
        self._cube = cube
        self._moves: list[int] = []
        self._nodes = 0
        steps = {}
        for step in (
            self._cross,
            self._first_layer_corners,
            self._middle_edges,
            self._last_layer,
        ):
            before = len(self._moves)
            step()
            steps[step.__name__.strip("_")] = len(self._moves) - before
        if not self._cube.is_solved():
            raise RuntimeError("beginner's method failed to solve the cube")
        return Solution(simplify(self._moves), self._nodes, time.perf_counter() - start, {"steps": steps})

    def _do(self, moves: list[int]) -> None:
        self._cube = self._cube.apply(moves)
        self._moves.extend(moves)

    def _cross(self) -> None:
        for k in range(1, 5):
            dist = _cross_table(k)
            index = _cross_index(self._cube, k)
            powers = 24 ** np.arange(k)
            while dist[index] > 0:
                digits = (index // powers) % 24
                self._nodes += 1
                for m in range(18):
                    nxt = int((_EDGE_CODES[digits, m] * powers).sum())
                    if dist[nxt] == dist[index] - 1:
                        self._do([m])
                        index = nxt
                        break

    def _corner_done(self, piece: int) -> bool:
        return self._cube.cp[piece] == piece and self._cube.co[piece] == 0

    def _first_layer_corners(self) -> None:
        for side, target in enumerate(DFR_SLOT):
            if self._corner_done(target):
                continue
            position = self._cube.cp.index(target)
            if position >= 4:  # in the wrong D slot: lift it into the U layer
                self._do(from_side(POP_CORNER, DFR_SLOT.index(position)))
            turns = 0
            while self._cube.cp[URF_SLOT[side]] != target:
                self._do([0])
                turns += 1
                assert turns < 4
            for _ in range(6):
                if self._corner_done(target):
                    break
                self._do(from_side(SEXY, side))
            assert self._corner_done(target)

    def _edge_done(self, piece: int, cube: CubieCube | None = None) -> bool:
        cube = cube or self._cube
        return cube.ep[piece] == piece and cube.eo[piece] == 0

    def _middle_edges(self) -> None:
        done: list[int] = []
        for side, target in enumerate(FR_SLOT):
            if not self._edge_done(target):
                position = self._cube.ep.index(target)
                if position >= 8:  # in the wrong middle slot: replace it with a U-layer edge
                    self._do(from_side(RIGHT_INSERT, FR_SLOT.index(position)))
                candidates = [
                    turn + from_side(alg, s)
                    for turn in ([], [0], [1], [2])
                    for alg, s in ((RIGHT_INSERT, side), (LEFT_INSERT, (side + 1) % 4))
                ]
                for moves in candidates:
                    self._nodes += 1
                    after = self._cube.apply(moves)
                    if all(self._edge_done(e, after) for e in done + [target]):
                        self._do(moves)
                        break
                else:
                    raise RuntimeError("no middle-layer insertion found")
            done.append(target)

    def _last_layer(self) -> None:
        cube = self._cube
        goals = {
            "edge_oll": lambda c: c.eo[0] == c.eo[1] == c.eo[2] == c.eo[3] == 0,
            "sune": lambda c: c.co[0] == c.co[1] == c.co[2] == c.co[3] == 0,
            "a_perm": lambda c: c.cp[:4] == (0, 1, 2, 3),
            "u_perm": lambda c: c.ep[:4] == (0, 1, 2, 3),
        }
        for step, goal in goals.items():
            path = self._macro_search(cube, _MACROS[step], goal)
            for alg in path:
                self._do(alg.moves)
                cube = self._cube

    def _macro_search(self, cube: CubieCube, macros: list[_Alg], goal) -> list[_Alg]:
        frontier: list[tuple[CubieCube, list[_Alg]]] = [(cube, [])]
        for _ in range(8):
            nxt = []
            for state, path in frontier:
                if goal(state):
                    return path
            for state, path in frontier:
                self._nodes += 1
                last_u = bool(path) and path[-1].is_u_turn
                for alg in macros:
                    if last_u and alg.is_u_turn:
                        continue
                    nxt.append((state * alg.cube, path + [alg]))
            frontier = nxt
        raise RuntimeError("last-layer search failed")
