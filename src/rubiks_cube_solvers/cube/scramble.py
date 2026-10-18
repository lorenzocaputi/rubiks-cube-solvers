"""Seeded scrambles: uniformly random states and random move sequences."""

from __future__ import annotations

import numpy as np

from .cubie import OPPOSITE, CubieCube, permutation_parity


def random_state(rng: np.random.Generator) -> CubieCube:
    """A state drawn uniformly from all 43,252,003,274,489,856,000 reachable states."""
    cp = rng.permutation(8)
    ep = rng.permutation(12)
    if permutation_parity(cp) != permutation_parity(ep):
        ep[[0, 1]] = ep[[1, 0]]
    co = rng.integers(0, 3, size=8)
    co[7] = (-co[:7].sum()) % 3
    eo = rng.integers(0, 2, size=12)
    eo[11] = eo[:11].sum() % 2
    return CubieCube(cp, co, ep, eo)


def random_moves(rng: np.random.Generator, length: int, faces: int = 6) -> list[int]:
    """Random face turns with no two consecutive turns of one face and no ``D U``-style
    pair (a face followed by its lower-numbered opposite face), so no move cancels."""
    moves: list[int] = []
    while len(moves) < length:
        face = int(rng.integers(faces))
        if moves:
            last = moves[-1] // 3
            if face == last or (face == OPPOSITE[last] and face < last):
                continue
        moves.append(3 * face + int(rng.integers(3)))
    return moves
