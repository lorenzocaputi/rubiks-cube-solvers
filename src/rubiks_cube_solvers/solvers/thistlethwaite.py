"""Thistlethwaite's four-phase algorithm.

The cube is moved down the subgroup chain

    G0 = <U, D, R, L, F, B>
    G1 = <U, D, R, L, F2, B2>      edges oriented
    G2 = <U, D, R2, L2, F2, B2>    corners oriented, E-slice edges in the E slice
    G3 = <U2, D2, R2, L2, F2, B2>  every piece in its own orbit of the half-turn group
    G4 = {solved}

Phase i turns a state of G(i-1) into a state of G(i) using only generators of
G(i-1).  Its state space is the set of cosets G(i) x, which has 2,048, 1,082,565,
29,400 and 663,552 elements.  Each phase has a full BFS distance table, so every
phase is solved optimally by walking down the table.
"""

from __future__ import annotations

import time
from itertools import permutations

import numpy as np
from numba import njit

from ..cube.cubie import MOVE_CP, MOVE_EP, CubieCube
from ..cube.kernels import (
    bfs_product,
    comb_rank,
    comb_unrank,
    descend,
    edge_move,
    flip_move_table,
    get_flip,
    get_slice,
    get_twist,
    perm_rank,
    perm_unrank,
    slice_move_table,
    twist_move_table,
)
from ..cube.notation import simplify
from ..tables import cached
from .base import Solution

ALL_MOVES = np.arange(18, dtype=np.int64)
PHASE_MOVES = (
    ALL_MOVES,
    np.array([m for m in range(18) if m not in (6, 8, 15, 17)], dtype=np.int64),  # no F, F', B, B'
    np.array([0, 1, 2, 9, 10, 11, 4, 13, 7, 16], dtype=np.int64),  # U, D any; R2 L2 F2 B2
    np.array([1, 4, 7, 10, 13, 16], dtype=np.int64),  # half turns
)
M_SLICE = (1, 3, 5, 7)  # UF UB DF DB positions and pieces
S_SLICE = (0, 2, 4, 6)  # UR UL DR DL
E_SLICE = (8, 9, 10, 11)
SOLVED_SLICE = 494


# ---- phase 3: corner cosets of the half-turn group --------------------------------
def half_turn_corner_group() -> np.ndarray:
    """The 96 corner permutations reachable with half turns, as rows of an (96, 8) array."""
    seen = {tuple(range(8))}
    frontier = [tuple(range(8))]
    while frontier:
        nxt = []
        for p in frontier:
            for m in PHASE_MOVES[3]:
                q = tuple(p[j] for j in MOVE_CP[m])
                if q not in seen:
                    seen.add(q)
                    nxt.append(q)
        frontier = nxt
    return np.array(sorted(seen), dtype=np.int8)


@njit(cache=True)
def _coset_minimum(perms, group):
    """For each corner permutation p, the smallest rank of h o p over h in the group."""
    out = np.empty(perms.shape[0], dtype=np.int64)
    q = np.empty(8, dtype=np.int8)
    for r in range(perms.shape[0]):
        best = 1 << 40
        for h in range(group.shape[0]):
            for i in range(8):
                q[i] = group[h, perms[r, i]]
            rank = perm_rank(q, 8)
            if rank < best:
                best = rank
        out[r] = best
    return out


def _phase3_corner_tables() -> tuple[np.ndarray, np.ndarray]:
    perms = np.array(list(permutations(range(8))), dtype=np.int8)  # row index = Lehmer rank
    minima = _coset_minimum(perms, half_turn_corner_group())
    representatives, label = np.unique(minima, return_inverse=True)
    move_table = np.full((len(representatives), 18), -1, dtype=np.int32)
    for coset, rank in enumerate(representatives):
        p = perms[rank]
        for m in PHASE_MOVES[2]:
            move_table[coset, m] = label[perm_rank(p[MOVE_CP[m]], 8)]
    return label.astype(np.int32), move_table


def _m_split(ep) -> int:
    mask = np.array([ep[i] in M_SLICE for i in range(8)])
    return int(comb_rank(mask, 8))


def _m_split_move_table() -> np.ndarray:
    table = np.full((70, 18), -1, dtype=np.int32)
    mask = np.zeros(8, dtype=np.bool_)
    for r in range(70):
        comb_unrank(r, 8, 4, mask)
        m_pieces, s_pieces = iter(M_SLICE), iter(S_SLICE)
        ep = [next(m_pieces) if mask[i] else next(s_pieces) for i in range(8)] + list(E_SLICE)
        for m in PHASE_MOVES[2]:
            table[r, m] = _m_split([ep[j] for j in MOVE_EP[m]])
    return table


# ---- phase 4: inside the half-turn group --------------------------------------------
# Edge positions of the M, S and E slices, and the weight of each slice's rank in the index.
_SLICES = (
    (np.array(M_SLICE, dtype=np.int64), 576),
    (np.array(S_SLICE, dtype=np.int64), 24),
    (np.array(E_SLICE, dtype=np.int64), 1),
)


@njit(cache=True)
def _slice_ranks(ep):
    p = np.empty(4, dtype=np.int8)
    total = 0
    for slots, offset in _SLICES:
        for k in range(4):
            piece = ep[slots[k]]
            for s in range(4):
                if slots[s] == piece:
                    p[k] = s
        total += perm_rank(p, 4) * offset
    return total


@njit(cache=True)
def _phase4_edge_table(moves):
    table = np.full((13824, 18), -1, dtype=np.int32)
    ep = np.empty(12, dtype=np.int8)
    eo = np.zeros(12, dtype=np.int8)
    ep2 = np.empty(12, dtype=np.int8)
    eo2 = np.empty(12, dtype=np.int8)
    p = np.empty(4, dtype=np.int8)
    for r in range(13824):
        rest = r
        for slots, offset in _SLICES:
            perm_unrank(rest // offset, 4, p)
            rest %= offset
            for k in range(4):
                ep[slots[k]] = slots[p[k]]
        for m in moves:
            edge_move(ep, eo, m, ep2, eo2)
            table[r, m] = _slice_ranks(ep2)
    return table


class ThistlethwaiteSolver:
    name = "thistlethwaite"
    label = "Thistlethwaite"
    puzzle = "3x3x3"

    def __init__(self) -> None:
        self.flip = cached("flip-move", flip_move_table)
        self.twist = cached("twist-move", twist_move_table)
        self.slice = cached("slice-move", slice_move_table)
        self.trivial = np.zeros((1, 18), dtype=np.int32)
        label, self.corner_coset = _phase3_corner_tables()
        self.corner_label = label
        self.m_split = _m_split_move_table()
        group = half_turn_corner_group()
        self.h96_index = np.full(40320, -1, dtype=np.int32)
        for i, p in enumerate(group):
            self.h96_index[perm_rank(p, 8)] = i
        self.h96 = np.full((96, 18), -1, dtype=np.int32)
        for i, p in enumerate(group):
            for m in PHASE_MOVES[3]:
                self.h96[i, m] = self.h96_index[perm_rank(p[MOVE_CP[m]], 8)]
        self.edges4 = cached("thistlethwaite-p4-edges", lambda: _phase4_edge_table(PHASE_MOVES[3]))
        self.tables = (
            (self.flip, self.trivial),
            (self.twist, self.slice),
            (self.corner_coset, self.m_split),
            (self.h96, self.edges4),
        )
        self.dist = tuple(
            cached(
                f"thistlethwaite-p{i + 1}",
                lambda i=i: bfs_product(*self.tables[i], PHASE_MOVES[i], self._goal(i)),
            )
            for i in range(4)
        )

    def _goal(self, phase: int) -> int:
        nb = self.tables[phase][1].shape[0]
        a, b = self._coords(phase, CubieCube())
        return a * nb + b

    def _coords(self, phase: int, cube: CubieCube) -> tuple[int, int]:
        cp, co, ep, eo = cube.to_arrays()
        if phase == 0:
            return int(get_flip(eo)), 0
        if phase == 1:
            return int(get_twist(co)), int(get_slice(ep))
        if phase == 2:
            return int(self.corner_label[perm_rank(cp, 8)]), _m_split(ep)
        return int(self.h96_index[perm_rank(cp, 8)]), int(_slice_ranks(ep))

    def solve(self, cube: CubieCube) -> Solution:
        start = time.perf_counter()
        moves: list[int] = []
        lengths = []
        for phase in range(4):
            a, b = self._coords(phase, cube)
            if phase == 3 and a < 0:
                raise RuntimeError("phase 3 did not reach the half-turn group")
            part = [int(m) for m in descend(self.dist[phase], *self.tables[phase], PHASE_MOVES[phase], a, b)]
            cube = cube.apply(part)
            moves.extend(part)
            lengths.append(len(part))
        if not cube.is_solved():
            raise RuntimeError("Thistlethwaite failed to solve the cube")
        return Solution(simplify(moves), sum(lengths), time.perf_counter() - start, {"phases": lengths})
