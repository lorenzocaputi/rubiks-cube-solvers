"""Kociemba's two-phase algorithm.

Phase 1 takes the cube into G1 = <U, D, R2, L2, F2, B2> (corners twisted 0, edges
flipped 0, E-slice edges in the E slice) with IDA* over (twist, flip, slice).
Phase 2 solves inside G1 with IDA* over (corner permutation, U/D edge permutation,
E-slice permutation).  Both heuristics take the maximum of two pruning tables, each
a BFS over a pair of coordinates (about 1 million entries each).

The search does not stop at the first solution: it lengthens phase 1 and keeps
searching for shorter totals until it finds one of at most ``target`` moves, until
phase 1 alone is as long as the best total, or until ``max_nodes`` nodes have been
expanded.  This is the original algorithm without the symmetry reduction and
inverse-search refinements of later implementations.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit

from ..cube.cubie import CubieCube
from ..cube.kernels import (
    bfs_product,
    corner_move,
    corner_perm_move_table,
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
from ..tables import cached
from .base import Solution

ALL_MOVES = np.arange(18, dtype=np.int64)
PHASE2_MOVES = np.array([0, 1, 2, 9, 10, 11, 4, 7, 13, 16], dtype=np.int64)  # U, D; R2 F2 L2 B2
SOLVED_SLICE = 494


@njit(cache=True)
def _ud_edge_move_table(moves):
    table = np.full((40320, 18), -1, dtype=np.int32)
    ep = np.empty(12, dtype=np.int8)
    eo = np.zeros(12, dtype=np.int8)
    ep2 = np.empty(12, dtype=np.int8)
    eo2 = np.empty(12, dtype=np.int8)
    for r in range(40320):
        perm_unrank(r, 8, ep)
        for i in range(8, 12):
            ep[i] = i
        for m in moves:
            edge_move(ep, eo, m, ep2, eo2)
            table[r, m] = perm_rank(ep2, 8)
    return table


@njit(cache=True)
def _slice_perm_move_table(moves):
    table = np.full((24, 18), -1, dtype=np.int32)
    ep = np.arange(12, dtype=np.int8)
    eo = np.zeros(12, dtype=np.int8)
    ep2 = np.empty(12, dtype=np.int8)
    eo2 = np.empty(12, dtype=np.int8)
    p = np.empty(4, dtype=np.int8)
    for r in range(24):
        perm_unrank(r, 4, p)
        for i in range(4):
            ep[8 + i] = 8 + p[i]
        for m in moves:
            edge_move(ep, eo, m, ep2, eo2)
            for i in range(4):
                p[i] = ep2[8 + i] - 8
            table[r, m] = perm_rank(p, 4)
    return table


@njit(cache=True)
def _skip(m, last):
    """Forbid two turns of one face in a row, and ``D U``-style pairs of opposite faces."""
    if last < 0:
        return False
    face, prev = m // 3, last // 3
    return face == prev or (face == (prev + 3) % 6 and face < prev)


@njit(cache=True)
def _phase2(cp, ud, sp, last, max_depth, moves, cp_move, ud_move, sp_move, prune_cp, prune_ud, out, offset):
    """IDA* inside G1.  Returns (length or -1, nodes expanded)."""
    nodes = 0
    h0 = max(prune_cp[cp * 24 + sp], prune_ud[ud * 24 + sp])
    if h0 > max_depth:
        return -1, nodes
    stack_cp = np.empty(max_depth + 1, dtype=np.int64)
    stack_ud = np.empty(max_depth + 1, dtype=np.int64)
    stack_sp = np.empty(max_depth + 1, dtype=np.int64)
    choice = np.empty(max_depth + 1, dtype=np.int64)
    n_moves = moves.shape[0]
    for depth in range(h0, max_depth + 1):
        if depth == 0:
            return 0, nodes
        d = 0
        stack_cp[0], stack_ud[0], stack_sp[0] = cp, ud, sp
        choice[0] = -1
        nodes += 1
        while d >= 0:
            choice[d] += 1
            if choice[d] == n_moves:
                d -= 1
                continue
            m = moves[choice[d]]
            prev = out[offset + d - 1] if d > 0 else last
            if _skip(m, prev):
                continue
            ncp = cp_move[stack_cp[d], m]
            nud = ud_move[stack_ud[d], m]
            nsp = sp_move[stack_sp[d], m]
            h = max(prune_cp[ncp * 24 + nsp], prune_ud[nud * 24 + nsp])
            if h > depth - d - 1:
                continue
            out[offset + d] = m
            if d + 1 == depth:
                return depth, nodes
            d += 1
            stack_cp[d], stack_ud[d], stack_sp[d] = ncp, nud, nsp
            choice[d] = -1
            nodes += 1
    return -1, nodes


@njit(cache=True)
def _search(
    cp0,
    co0,
    ep0,
    eo0,
    twist_move,
    flip_move,
    slice_move,
    prune_twist,
    prune_flip,
    cp_move,
    ud_move,
    sp_move,
    prune_cp,
    prune_ud,
    phase2_moves,
    max_length,
    target,
    max_nodes,
    best_moves,
):
    """Returns (best length or -1, phase-1 length of the best, nodes expanded)."""
    is_phase2 = np.zeros(18, dtype=np.bool_)
    for m in phase2_moves:
        is_phase2[m] = True
    best = max_length + 1
    best_p1 = -1
    nodes = 0
    path = np.empty(64, dtype=np.int64)
    tw = np.empty(32, dtype=np.int64)
    fl = np.empty(32, dtype=np.int64)
    sl = np.empty(32, dtype=np.int64)
    choice = np.empty(32, dtype=np.int64)
    cp = np.empty(8, dtype=np.int8)
    co = np.empty(8, dtype=np.int8)
    ep = np.empty(12, dtype=np.int8)
    eo = np.empty(12, dtype=np.int8)
    cp2 = np.empty(8, dtype=np.int8)
    co2 = np.empty(8, dtype=np.int8)
    ep2 = np.empty(12, dtype=np.int8)
    eo2 = np.empty(12, dtype=np.int8)
    sp = np.empty(4, dtype=np.int8)
    t0, f0, s0 = get_twist(co0), get_flip(eo0), get_slice(ep0)
    h_start = max(prune_twist[t0 * 495 + s0], prune_flip[f0 * 495 + s0])
    for depth1 in range(h_start, 31):
        if depth1 >= best or nodes > max_nodes:
            break
        tw[0], fl[0], sl[0] = t0, f0, s0
        choice[0] = -1
        d = 0
        nodes += 1
        while d >= 0:
            if depth1 == 0:
                reached = True
            else:
                reached = False
                choice[d] += 1
                if choice[d] == 18:
                    d -= 1
                    continue
                m = choice[d]
                if d > 0 and _skip(m, path[d - 1]):
                    continue
                nt = twist_move[tw[d], m]
                nf = flip_move[fl[d], m]
                ns = slice_move[sl[d], m]
                h = max(prune_twist[nt * 495 + ns], prune_flip[nf * 495 + ns])
                togo = depth1 - d - 1
                if h > togo:
                    continue
                path[d] = m
                if togo > 0:
                    if h == 0 and togo < 5:
                        continue  # already in G1: a shorter phase 1 covered this
                    d += 1
                    tw[d], fl[d], sl[d] = nt, nf, ns
                    choice[d] = -1
                    nodes += 1
                    continue
                if is_phase2[m]:
                    continue  # the previous state was already in G1
                reached = True
            if reached:
                # Phase 1 done in exactly depth1 moves: compute the phase-2 coordinates.
                cp[:] = cp0
                co[:] = co0
                ep[:] = ep0
                eo[:] = eo0
                for k in range(depth1):
                    corner_move(cp, co, path[k], cp2, co2)
                    edge_move(ep, eo, path[k], ep2, eo2)
                    cp[:] = cp2
                    co[:] = co2
                    ep[:] = ep2
                    eo[:] = eo2
                for i in range(4):
                    sp[i] = ep[8 + i] - 8
                last = path[depth1 - 1] if depth1 > 0 else -1
                limit = min(best - 1 - depth1, 18)
                if limit >= 0:
                    length, n2 = _phase2(
                        perm_rank(cp, 8),
                        perm_rank(ep, 8),
                        perm_rank(sp, 4),
                        last,
                        limit,
                        phase2_moves,
                        cp_move,
                        ud_move,
                        sp_move,
                        prune_cp,
                        prune_ud,
                        path,
                        depth1,
                    )
                    nodes += n2
                    if length >= 0:
                        best = depth1 + length
                        best_p1 = depth1
                        best_moves[:best] = path[:best]
                        if best <= target:
                            return best, best_p1, nodes
                if depth1 == 0 or nodes > max_nodes:
                    break
    if best > max_length:
        return -1, -1, nodes
    return best, best_p1, nodes


class KociembaSolver:
    name = "kociemba"
    label = "Kociemba two-phase"
    puzzle = "3x3x3"

    def __init__(self, target: int = 20, max_length: int = 30, max_nodes: int = 2_000_000) -> None:
        self.target = target
        self.max_length = max_length
        self.max_nodes = max_nodes
        self.twist = cached("twist-move", twist_move_table)
        self.flip = cached("flip-move", flip_move_table)
        self.slice = cached("slice-move", slice_move_table)
        self.cp = cached("corner-perm-move", corner_perm_move_table)
        self.ud = cached("kociemba-ud-edge-move", lambda: _ud_edge_move_table(PHASE2_MOVES))
        self.sp = cached("kociemba-slice-perm-move", lambda: _slice_perm_move_table(PHASE2_MOVES))
        self.prune_twist = cached(
            "kociemba-prune-twist-slice", lambda: bfs_product(self.twist, self.slice, ALL_MOVES, SOLVED_SLICE)
        )
        self.prune_flip = cached(
            "kociemba-prune-flip-slice", lambda: bfs_product(self.flip, self.slice, ALL_MOVES, SOLVED_SLICE)
        )
        self.prune_cp = cached("kociemba-prune-cp-sp", lambda: bfs_product(self.cp, self.sp, PHASE2_MOVES, 0))
        self.prune_ud = cached("kociemba-prune-ud-sp", lambda: bfs_product(self.ud, self.sp, PHASE2_MOVES, 0))

    def solve(self, cube: CubieCube) -> Solution:
        start = time.perf_counter()
        best_moves = np.empty(64, dtype=np.int64)
        length, phase1, nodes = _search(
            *cube.to_arrays(),
            self.twist,
            self.flip,
            self.slice,
            self.prune_twist,
            self.prune_flip,
            self.cp,
            self.ud,
            self.sp,
            self.prune_cp,
            self.prune_ud,
            PHASE2_MOVES,
            self.max_length,
            self.target,
            self.max_nodes,
            best_moves,
        )
        if length < 0:
            raise RuntimeError("no solution within the length limit")
        moves = [int(m) for m in best_moves[:length]]
        return Solution(moves, int(nodes), time.perf_counter() - start, {"phase1": int(phase1)})
