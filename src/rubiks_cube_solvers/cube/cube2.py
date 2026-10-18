"""The 2x2x2 cube, modelled as the eight corners of the 3x3x3.

A 2x2x2 has no fixed centres, so every state is taken in the frame where the DBL
corner is home and untwisted.  Turns of U, R and F never move DBL, which leaves
9 moves (3x3x3 move indices 0..8) and 7! * 3^6 = 3,674,160 states.  A state's
index is ``perm * 729 + twist``: ``perm`` ranks the other seven corners and
``twist`` encodes the twists of positions URF..DLF.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ..tables import cached
from .cubie import MOVE_CO, MOVE_CP, CubieCube
from .kernels import bfs_product, perm_rank, perm_unrank

N_PERM = 5040
N_TWIST = 729
N_STATES = N_PERM * N_TWIST
MOVES = np.arange(9, dtype=np.int64)  # U U2 U' R R2 R' F F2 F'
QUARTER_MOVES = np.array([0, 2, 3, 5, 6, 8], dtype=np.int64)  # U U' R R' F F'
_POSITIONS = np.array([0, 1, 2, 3, 4, 5, 7], dtype=np.int64)  # all corners but DBL


@njit(cache=True)
def encode(cp, co):
    p = np.empty(7, dtype=np.int64)
    for k in range(7):
        piece = cp[_POSITIONS[k]]
        p[k] = 6 if piece == 7 else piece
    t = 0
    for k in range(6):
        t = 3 * t + co[k]
    return perm_rank(p, 7) * N_TWIST + t


@njit(cache=True)
def decode(index, cp, co):
    perm, t = index // N_TWIST, index % N_TWIST
    p = np.empty(7, dtype=np.int64)
    perm_unrank(perm, 7, p)
    for k in range(7):
        cp[_POSITIONS[k]] = 7 if p[k] == 6 else p[k]
    cp[6] = 6
    total = 0
    for k in range(5, -1, -1):
        co[k] = t % 3
        total += co[k]
        t //= 3
    co[6] = 0
    co[7] = (3 - total % 3) % 3


@njit(cache=True)
def apply_move(index, m):
    cp = np.empty(8, dtype=np.int8)
    co = np.empty(8, dtype=np.int8)
    decode(index, cp, co)
    cp2 = np.empty(8, dtype=np.int8)
    co2 = np.empty(8, dtype=np.int8)
    for i in range(8):
        j = MOVE_CP[m, i]
        cp2[i] = cp[j]
        co2[i] = (co[j] + MOVE_CO[m, i]) % 3
    return encode(cp2, co2)


@njit(cache=True)
def _move_tables():
    perm_table = np.empty((N_PERM, 9), dtype=np.int32)
    twist_table = np.empty((N_TWIST, 9), dtype=np.int32)
    for perm in range(N_PERM):
        for m in range(9):
            perm_table[perm, m] = apply_move(perm * N_TWIST, m) // N_TWIST
    for t in range(N_TWIST):
        for m in range(9):
            twist_table[t, m] = apply_move(t, m) % N_TWIST
    return perm_table, twist_table


_TABLES: tuple[np.ndarray, np.ndarray] | None = None


def move_tables() -> tuple[np.ndarray, np.ndarray]:
    """Move tables for the permutation (5040 x 9) and twist (729 x 9) coordinates."""
    global _TABLES
    if _TABLES is None:
        _TABLES = _move_tables()
    return _TABLES


SOLVED_INDEX = 0


def distance_table(metric: str = "htm") -> np.ndarray:
    """Distance to solved for all 3,674,160 states by breadth-first search.

    ``metric`` is ``"htm"`` (half turns count 1) or ``"qtm"`` (quarter turns only).
    """
    moves = {"htm": MOVES, "qtm": QUARTER_MOVES}[metric]

    def build() -> np.ndarray:
        perm_table, twist_table = move_tables()
        return bfs_product(perm_table, twist_table, moves, SOLVED_INDEX)

    return cached(f"cube2-distance-{metric}", build)


def state_index(cube: CubieCube) -> int:
    """Index of the corner state of ``cube`` (DBL must be home and untwisted)."""
    if cube.cp[6] != 6 or cube.co[6] != 0:
        raise ValueError("2x2x2 states are taken with the DBL corner fixed")
    return int(encode(np.array(cube.cp, np.int8), np.array(cube.co, np.int8)))


def state_cube(index: int) -> CubieCube:
    cp = np.empty(8, dtype=np.int8)
    co = np.empty(8, dtype=np.int8)
    decode(index, cp, co)
    return CubieCube(cp, co)


def random_state(rng: np.random.Generator) -> int:
    """Index of a uniformly random 2x2x2 state."""
    return int(rng.integers(N_STATES))


def apply_moves(index: int, moves) -> int:
    for m in moves:
        if not 0 <= m < 9:
            raise ValueError("the 2x2x2 model uses only U, R and F turns")
        index = int(apply_move(index, m))
    return index


# Stickers of the corner positions, in the facelet numbering of the 3x3x3 (24 of 54).
def sticker_colors(index: int) -> np.ndarray:
    """Colour (0..5 for U R F D L B) of the 24 corner stickers, for rendering and for the network."""
    from .facelet import CORNER_FACELETS, to_facelets

    text = to_facelets(state_cube(index))
    order = [f for corner in CORNER_FACELETS for f in corner]
    return np.array(["URFDLB".index(text[f]) for f in order], dtype=np.int8)
