"""Numba kernels shared by the solvers: cubie products, coordinates, move tables, BFS.

A *coordinate* maps part of the cube state to an integer in ``range(N)``; a *move
table* of shape (N, 18) gives the coordinate after each move.  Entries for moves a
table does not support are -1.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .cubie import MOVE_CO, MOVE_CP, MOVE_EO, MOVE_EP

N_MOVES = 18


def _binomials(n: int) -> np.ndarray:
    table = np.zeros((n + 1, n + 1), dtype=np.int64)
    for a in range(n + 1):
        table[a, 0] = 1
        for b in range(1, a + 1):
            table[a, b] = table[a - 1, b - 1] + table[a - 1, b]
    return table


BINOM = _binomials(12)
FACTORIAL = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800], dtype=np.int64)


# ---- products ---------------------------------------------------------------
@njit(cache=True)
def corner_move(cp, co, m, out_cp, out_co):
    for i in range(8):
        j = MOVE_CP[m, i]
        out_cp[i] = cp[j]
        out_co[i] = (co[j] + MOVE_CO[m, i]) % 3


@njit(cache=True)
def edge_move(ep, eo, m, out_ep, out_eo):
    for i in range(12):
        j = MOVE_EP[m, i]
        out_ep[i] = ep[j]
        out_eo[i] = (eo[j] + MOVE_EO[m, i]) & 1


# ---- permutation and combination ranks ---------------------------------------
@njit(cache=True)
def perm_rank(p, n):
    """Lehmer rank of the permutation ``p[:n]`` of ``0..n-1`` (identity is 0)."""
    r = 0
    for i in range(n):
        smaller = 0
        for j in range(i + 1, n):
            if p[j] < p[i]:
                smaller += 1
        r = r * (n - i) + smaller
    return r


@njit(cache=True)
def perm_unrank(r, n, out):
    digits = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        base = n - i
        digits[i] = r % base
        r //= base
    used = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        k = digits[i]
        for v in range(n):
            if not used[v]:
                if k == 0:
                    out[i] = v
                    used[v] = True
                    break
                k -= 1


@njit(cache=True)
def comb_rank(mask_positions, n):
    """Rank of a k-subset of ``range(n)`` given as a boolean array (combinatorial number system)."""
    r = 0
    k = 0
    for pos in range(n):
        if mask_positions[pos]:
            k += 1
            r += BINOM[pos, k]
    return r


@njit(cache=True)
def comb_unrank(r, n, k, out_mask):
    for pos in range(n):
        out_mask[pos] = False
    for pos in range(n - 1, -1, -1):
        if k > 0 and BINOM[pos, k] <= r:
            r -= BINOM[pos, k]
            out_mask[pos] = True
            k -= 1


# ---- coordinates shared by Thistlethwaite, Kociemba and Korf ----------------
@njit(cache=True)
def get_twist(co):
    t = 0
    for i in range(7):
        t = 3 * t + co[i]
    return t


@njit(cache=True)
def set_twist(t, co):
    total = 0
    for i in range(6, -1, -1):
        co[i] = t % 3
        total += co[i]
        t //= 3
    co[7] = (3 - total % 3) % 3


@njit(cache=True)
def get_flip(eo):
    f = 0
    for i in range(11):
        f = 2 * f + eo[i]
    return f


@njit(cache=True)
def set_flip(f, eo):
    total = 0
    for i in range(10, -1, -1):
        eo[i] = f & 1
        total += eo[i]
        f >>= 1
    eo[11] = total & 1


@njit(cache=True)
def get_slice(ep):
    """Which 4 of the 12 edge positions hold the E-slice edges FR, FL, BL, BR (0..494)."""
    mask = np.zeros(12, dtype=np.bool_)
    for i in range(12):
        mask[i] = ep[i] >= 8
    return comb_rank(mask, 12)


@njit(cache=True)
def set_slice(r, ep):
    mask = np.zeros(12, dtype=np.bool_)
    comb_unrank(r, 12, 4, mask)
    slice_piece = 8
    other_piece = 0
    for i in range(12):
        if mask[i]:
            ep[i] = slice_piece
            slice_piece += 1
        else:
            ep[i] = other_piece
            other_piece += 1


@njit(cache=True)
def twist_move_table():
    table = np.empty((2187, N_MOVES), dtype=np.int16)
    cp = np.arange(8, dtype=np.int8)
    co = np.zeros(8, dtype=np.int8)
    cp2 = np.empty(8, dtype=np.int8)
    co2 = np.empty(8, dtype=np.int8)
    for t in range(2187):
        set_twist(t, co)
        for m in range(N_MOVES):
            corner_move(cp, co, m, cp2, co2)
            table[t, m] = get_twist(co2)
    return table


@njit(cache=True)
def flip_move_table():
    table = np.empty((2048, N_MOVES), dtype=np.int16)
    ep = np.arange(12, dtype=np.int8)
    eo = np.zeros(12, dtype=np.int8)
    ep2 = np.empty(12, dtype=np.int8)
    eo2 = np.empty(12, dtype=np.int8)
    for f in range(2048):
        set_flip(f, eo)
        for m in range(N_MOVES):
            edge_move(ep, eo, m, ep2, eo2)
            table[f, m] = get_flip(eo2)
    return table


@njit(cache=True)
def slice_move_table():
    table = np.empty((495, N_MOVES), dtype=np.int16)
    ep = np.empty(12, dtype=np.int8)
    eo = np.zeros(12, dtype=np.int8)
    ep2 = np.empty(12, dtype=np.int8)
    eo2 = np.empty(12, dtype=np.int8)
    for r in range(495):
        set_slice(r, ep)
        for m in range(N_MOVES):
            edge_move(ep, eo, m, ep2, eo2)
            table[r, m] = get_slice(ep2)
    return table


@njit(cache=True)
def corner_perm_move_table():
    table = np.empty((40320, N_MOVES), dtype=np.int32)
    cp = np.empty(8, dtype=np.int8)
    co = np.zeros(8, dtype=np.int8)
    cp2 = np.empty(8, dtype=np.int8)
    co2 = np.empty(8, dtype=np.int8)
    for r in range(40320):
        perm_unrank(r, 8, cp)
        for m in range(N_MOVES):
            corner_move(cp, co, m, cp2, co2)
            table[r, m] = perm_rank(cp2, 8)
    return table


# ---- BFS over a product of two coordinates ----------------------------------
@njit(cache=True)
def bfs_product(table_a, table_b, moves, start):
    """Distances from ``start`` over states ``a * nb + b`` (-1 where unreachable).

    The move set must be closed under inverses, which lets large levels be
    filled by scanning unvisited states for a neighbour at the current depth.
    """
    na = table_a.shape[0]
    nb = table_b.shape[0]
    n = na * nb
    dist = np.full(n, -1, dtype=np.int8)
    dist[start] = 0
    depth = 0
    filled = 1
    count = 1
    while count > 0:
        count = 0
        if filled < n // 4:
            for i in range(n):
                if dist[i] == depth:
                    a = i // nb
                    b = i - a * nb
                    for m in moves:
                        j = table_a[a, m] * nb + table_b[b, m]
                        if dist[j] < 0:
                            dist[j] = depth + 1
                            count += 1
        else:
            for i in range(n):
                if dist[i] < 0:
                    a = i // nb
                    b = i - a * nb
                    for m in moves:
                        j = table_a[a, m] * nb + table_b[b, m]
                        if dist[j] == depth:
                            dist[i] = depth + 1
                            count += 1
                            break
        filled += count
        depth += 1
    return dist


@njit(cache=True)
def descend(dist, table_a, table_b, moves, a, b):
    """Optimal move sequence to distance 0 in a full product distance table."""
    nb = table_b.shape[0]
    d = dist[a * nb + b]
    out = np.empty(max(d, 0), dtype=np.int64)
    k = 0
    while d > 0:
        for m in moves:
            na = table_a[a, m]
            nb_ = table_b[b, m]
            if dist[na * nb + nb_] == d - 1:
                out[k] = m
                k += 1
                a = na
                b = nb_
                d -= 1
                break
    return out
