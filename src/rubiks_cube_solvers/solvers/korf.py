"""Optimal solving with IDA* and pattern databases (Korf 1997).

A pattern database (PDB) stores, for every state of a sub-problem, its exact
distance to solved; that distance is a lower bound for the whole cube, so IDA*
with the maximum over several PDBs returns shortest solutions.

3x3x3, as in Korf's paper: one PDB for the corners (8! * 3^7 = 88,179,840 states)
and one for each of two groups of six edges (12!/6! * 2^6 = 42,577,920 states).
Building them takes a few minutes once; they are cached as ``.npy`` files.

2x2x2: one PDB for the corner permutation (5,040) and one for the twists (729).

``quick=True`` swaps the 3x3x3 PDBs for small ones (corner permutation, corner
twist and three groups of four edges), which build in seconds and are still
admissible, so solutions stay optimal; only the search gets slower.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit

from ..cube import cube2
from ..cube.cubie import MOVE_EO, MOVE_EP, CubieCube
from ..cube.kernels import bfs_product, corner_perm_move_table, perm_rank, twist_move_table
from ..tables import cached
from .base import Solution

ALL_MOVES = np.arange(18, dtype=np.int64)
POPCOUNT = np.array([bin(i).count("1") for i in range(4096)], dtype=np.int64)


def _edge_code_table() -> np.ndarray:
    """``table[2 * position + flip, move]``: where a single edge goes."""
    table = np.empty((24, 18), dtype=np.int64)
    for m in range(18):
        for i in range(12):
            j = MOVE_EP[m, i]
            for flip in range(2):
                table[2 * j + flip, m] = 2 * i + (flip + MOVE_EO[m, i]) % 2
    return table


EDGE_CODES = _edge_code_table()


# ---- edge-group pattern databases ----------------------------------------------------
@njit(cache=True)
def _group_decode(index, k, out):
    """Codes of the k tracked edges for a group index."""
    flips = index & ((1 << k) - 1)
    rank = index >> k
    digits = np.empty(k, dtype=np.int64)
    for i in range(k - 1, -1, -1):
        digits[i] = rank % (12 - i)
        rank //= 12 - i
    used = 0
    for i in range(k):
        c = digits[i]
        for p in range(12):
            if not (used >> p) & 1:
                if c == 0:
                    used |= 1 << p
                    out[i] = 2 * p + ((flips >> (k - 1 - i)) & 1)
                    break
                c -= 1


@njit(cache=True)
def _group_neighbor(codes, m, k):
    """Index of the group after move m."""
    used = 0
    rank = 0
    flips = 0
    for i in range(k):
        c = EDGE_CODES[codes[i], m]
        p = c >> 1
        rank = rank * (12 - i) + POPCOUNT[~used & ((1 << p) - 1)]
        used |= 1 << p
        flips = 2 * flips + (c & 1)
    return (rank << k) + flips


@njit(cache=True)
def _group_rank(codes, k):
    """Index of a group from the codes (2 * position + flip) of its k pieces."""
    used = 0
    rank = 0
    flips = 0
    for i in range(k):
        c = codes[i]
        p = c >> 1
        rank = rank * (12 - i) + POPCOUNT[~used & ((1 << p) - 1)]
        used |= 1 << p
        flips = 2 * flips + (c & 1)
    return (rank << k) + flips


# ---- 3x3x3 IDA* -------------------------------------------------------------------------
@njit(cache=True)
def _skip(m, last):
    if last < 0:
        return False
    face, prev = m // 3, last // 3
    return face == prev or (face == (prev + 3) % 6 and face < prev)


@njit(cache=True)
def _heuristic(cp, tw, codes, corner_pdb, cp_pdb, tw_pdb, groups, edge_pdbs, scratch):
    h = max(cp_pdb[cp], tw_pdb[tw])
    if corner_pdb.shape[0] > 1:
        h = max(h, corner_pdb[cp * 2187 + tw])
    k = groups.shape[1]
    for g in range(groups.shape[0]):
        for i in range(k):
            scratch[i] = codes[groups[g, i]]
        v = edge_pdbs[g, _group_rank(scratch, k)]
        if v > h:
            h = v
    return h


@njit(cache=True)
def _within(limit, cp, tw, codes, corner_pdb, cp_pdb, tw_pdb, groups, edge_pdbs, scratch):
    """Whether the heuristic is at most ``limit``, reading the tables one at a time and
    stopping at the first that exceeds it (most children are cut by the corners alone)."""
    if corner_pdb.shape[0] > 1:
        if corner_pdb[cp * 2187 + tw] > limit:
            return False
    elif cp_pdb[cp] > limit or tw_pdb[tw] > limit:
        return False
    k = groups.shape[1]
    for g in range(groups.shape[0]):
        for i in range(k):
            scratch[i] = codes[groups[g, i]]
        if edge_pdbs[g, _group_rank(scratch, k)] > limit:
            return False
    return True


@njit(cache=True)
def _ida_star_3(
    cp0, tw0, codes0, cp_move, tw_move, corner_pdb, cp_pdb, tw_pdb, groups, edge_pdbs, max_depth, out
):
    """Returns (length or -1, nodes expanded)."""
    scratch = np.empty(groups.shape[1], dtype=np.int64)
    h0 = _heuristic(cp0, tw0, codes0, corner_pdb, cp_pdb, tw_pdb, groups, edge_pdbs, scratch)
    nodes = 0
    cps = np.empty(max_depth + 1, dtype=np.int64)
    tws = np.empty(max_depth + 1, dtype=np.int64)
    codes = np.empty((max_depth + 1, 12), dtype=np.int64)
    choice = np.empty(max_depth + 1, dtype=np.int64)
    for bound in range(h0, max_depth + 1):
        if bound == 0:
            return 0, nodes
        cps[0], tws[0] = cp0, tw0
        codes[0, :] = codes0
        choice[0] = -1
        nodes += 1
        d = 0
        while d >= 0:
            choice[d] += 1
            m = choice[d]
            if m == 18:
                d -= 1
                continue
            if d > 0 and _skip(m, out[d - 1]):
                continue
            ncp = cp_move[cps[d], m]
            ntw = tw_move[tws[d], m]
            for e in range(12):
                codes[d + 1, e] = EDGE_CODES[codes[d, e], m]
            if not _within(
                bound - d - 1, ncp, ntw, codes[d + 1], corner_pdb, cp_pdb, tw_pdb, groups, edge_pdbs, scratch
            ):
                continue
            out[d] = m
            if d + 1 == bound:
                return bound, nodes  # h == 0 with every edge group and the corners home: solved
            d += 1
            cps[d], tws[d] = ncp, ntw
            choice[d] = -1
            nodes += 1
    return -1, nodes


FULL_GROUPS = np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=np.int64)
QUICK_GROUPS = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=np.int64)


@njit(cache=True)
def _edge_pdb_for(pieces):
    """BFS distances for one group of edge pieces with their real home positions."""
    k = pieces.shape[0]
    size = 1
    for i in range(k):
        size *= 12 - i
    size <<= k
    dist = np.full(size, -1, dtype=np.int8)
    codes = np.empty(k, dtype=np.int64)
    home = np.empty(k, dtype=np.int64)
    for i in range(k):
        home[i] = 2 * pieces[i]
    dist[_group_rank(home, k)] = 0
    depth = 0
    filled = 1
    count = 1
    while count > 0:
        count = 0
        backward = filled > size // 4
        for i in range(size):
            if backward:
                if dist[i] >= 0:
                    continue
                _group_decode(i, k, codes)
                for m in range(18):
                    if dist[_group_neighbor(codes, m, k)] == depth:
                        dist[i] = depth + 1
                        count += 1
                        break
            elif dist[i] == depth:
                _group_decode(i, k, codes)
                for m in range(18):
                    j = _group_neighbor(codes, m, k)
                    if dist[j] < 0:
                        dist[j] = depth + 1
                        count += 1
        filled += count
        depth += 1
    return dist


class KorfSolver:
    """Optimal 3x3x3 solver.  ``max_depth`` caps the search; beyond about 15 moves it gets slow."""

    name = "korf"
    label = "Korf IDA* (optimal)"
    puzzle = "3x3x3"

    def __init__(self, max_depth: int = 20, quick: bool = False) -> None:
        self.max_depth = max_depth
        self.cp_move = cached("corner-perm-move", corner_perm_move_table)
        self.tw_move = cached("twist-move", twist_move_table)
        trivial = np.zeros((1, 18), np.int32)
        self.cp_pdb = cached("korf-pdb-cp", lambda: bfs_product(self.cp_move, trivial, ALL_MOVES, 0))
        self.tw_pdb = cached("korf-pdb-twist", lambda: bfs_product(self.tw_move, trivial, ALL_MOVES, 0))
        if quick:
            self.corner_pdb = np.zeros(1, dtype=np.int8)
            self.groups = QUICK_GROUPS
        else:
            self.corner_pdb = cached(
                "korf-pdb-corners", lambda: bfs_product(self.cp_move, self.tw_move, ALL_MOVES, 0)
            )
            self.groups = FULL_GROUPS
        self.edge_pdbs = np.stack(
            [
                cached(f"korf-pdb-edges-{'-'.join(map(str, g))}", lambda g=g: _edge_pdb_for(g))
                for g in self.groups
            ]
        )

    def solve(self, cube: CubieCube) -> Solution:
        start = time.perf_counter()
        cp, co, ep, eo = cube.to_arrays()
        codes = np.empty(12, dtype=np.int64)
        for position in range(12):
            codes[ep[position]] = 2 * position + eo[position]
        tw = 0
        for i in range(7):
            tw = 3 * tw + int(co[i])
        out = np.empty(self.max_depth + 1, dtype=np.int64)
        length, nodes = _ida_star_3(
            int(perm_rank(cp, 8)),
            tw,
            codes,
            self.cp_move,
            self.tw_move,
            self.corner_pdb,
            self.cp_pdb,
            self.tw_pdb,
            self.groups,
            self.edge_pdbs,
            self.max_depth,
            out,
        )
        if length < 0:
            raise RuntimeError(f"no solution within {self.max_depth} moves")
        return Solution([int(m) for m in out[:length]], int(nodes), time.perf_counter() - start)


# ---- 2x2x2 IDA* -------------------------------------------------------------------------
@njit(cache=True)
def _ida_star_2(perm0, twist0, perm_move, twist_move, perm_pdb, twist_pdb, max_depth, out):
    nodes = 0
    h0 = max(perm_pdb[perm0], twist_pdb[twist0])
    perms = np.empty(max_depth + 1, dtype=np.int64)
    twists = np.empty(max_depth + 1, dtype=np.int64)
    choice = np.empty(max_depth + 1, dtype=np.int64)
    for bound in range(h0, max_depth + 1):
        if bound == 0:
            return 0, nodes
        perms[0], twists[0] = perm0, twist0
        choice[0] = -1
        nodes += 1
        d = 0
        while d >= 0:
            choice[d] += 1
            m = choice[d]
            if m == 9:
                d -= 1
                continue
            if d > 0 and m // 3 == out[d - 1] // 3:
                continue
            p = perm_move[perms[d], m]
            t = twist_move[twists[d], m]
            if max(perm_pdb[p], twist_pdb[t]) > bound - d - 1:
                continue
            out[d] = m
            if d + 1 == bound:
                return bound, nodes
            d += 1
            perms[d], twists[d] = p, t
            choice[d] = -1
            nodes += 1
    return -1, nodes


class Korf2Solver:
    """Optimal 2x2x2 solver: IDA* with permutation and twist pattern databases."""

    name = "korf-2x2"
    label = "Korf IDA* (optimal)"
    puzzle = "2x2x2"

    def __init__(self) -> None:
        self.perm_move, self.twist_move = cube2.move_tables()
        trivial = np.zeros((1, 9), dtype=np.int32)
        self.perm_pdb = bfs_product(self.perm_move, trivial, cube2.MOVES, 0)
        self.twist_pdb = bfs_product(self.twist_move, trivial, cube2.MOVES, 0)

    def solve(self, index: int) -> Solution:
        start = time.perf_counter()
        out = np.empty(12, dtype=np.int64)
        length, nodes = _ida_star_2(
            index // cube2.N_TWIST,
            index % cube2.N_TWIST,
            self.perm_move,
            self.twist_move,
            self.perm_pdb,
            self.twist_pdb,
            11,
            out,
        )
        return Solution([int(m) for m in out[:length]], int(nodes), time.perf_counter() - start)
