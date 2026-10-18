"""Cubie-level model of the 3x3x3 cube.

A state is four arrays, in the "replaced-by" convention used by Kociemba:
``cp[i]`` is the corner piece sitting in corner position ``i`` and ``co[i]`` its
twist (0, 1 or 2); ``ep`` and ``eo`` are the same for edges.  The product
``a * b`` means "apply a, then b", so applying a move to a state is ``state * move``.

Orientation conventions:
* a corner has twist 0 when its U or D sticker is on the U or D face;
* an edge has flip 0 when it is in the orientation it would have if it were
  brought home using only U, D, R, L, F2 and B2.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

FACES = "URFDLB"
OPPOSITE = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}

CORNERS = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")
EDGES = ("UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR")
URF, UFL, ULB, UBR, DFR, DLF, DBL, DRB = range(8)
UR, UF, UL, UB, DR, DF, DL, DB, FR, FL, BL, BR = range(12)

# Clockwise quarter turns of the six faces (looking at the face).
_BASIC = {
    "U": (
        (UBR, URF, UFL, ULB, DFR, DLF, DBL, DRB),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (UB, UR, UF, UL, DR, DF, DL, DB, FR, FL, BL, BR),
        (0,) * 12,
    ),
    "R": (
        (DFR, UFL, ULB, URF, DRB, DLF, DBL, UBR),
        (2, 0, 0, 1, 1, 0, 0, 2),
        (FR, UF, UL, UB, BR, DF, DL, DB, DR, FL, BL, UR),
        (0,) * 12,
    ),
    "F": (
        (UFL, DLF, ULB, UBR, URF, DFR, DBL, DRB),
        (1, 2, 0, 0, 2, 1, 0, 0),
        (UR, FL, UL, UB, DR, FR, DL, DB, UF, DF, BL, BR),
        (0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0),
    ),
    "D": (
        (URF, UFL, ULB, UBR, DLF, DBL, DRB, DFR),
        (0,) * 8,
        (UR, UF, UL, UB, DF, DL, DB, DR, FR, FL, BL, BR),
        (0,) * 12,
    ),
    "L": (
        (URF, ULB, DBL, UBR, DFR, UFL, DLF, DRB),
        (0, 1, 2, 0, 0, 2, 1, 0),
        (UR, UF, BL, UB, DR, DF, FL, DB, FR, UL, DL, BR),
        (0,) * 12,
    ),
    "B": (
        (URF, UFL, UBR, DRB, DFR, DLF, ULB, DBL),
        (0, 0, 1, 2, 0, 0, 2, 1),
        (UR, UF, UL, BR, DR, DF, DL, BL, FR, FL, UB, DB),
        (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1),
    ),
}


class CubieCube:
    """Permutation and orientation of the 8 corners and 12 edges."""

    __slots__ = ("cp", "co", "ep", "eo")

    def __init__(
        self,
        cp: Sequence[int] | None = None,
        co: Sequence[int] | None = None,
        ep: Sequence[int] | None = None,
        eo: Sequence[int] | None = None,
    ) -> None:
        self.cp = tuple(range(8)) if cp is None else tuple(int(x) for x in cp)
        self.co = (0,) * 8 if co is None else tuple(int(x) for x in co)
        self.ep = tuple(range(12)) if ep is None else tuple(int(x) for x in ep)
        self.eo = (0,) * 12 if eo is None else tuple(int(x) for x in eo)

    # ---- group operations -------------------------------------------------
    def __mul__(self, other: CubieCube) -> CubieCube:
        acp, aco, aep, aeo = self.cp, self.co, self.ep, self.eo
        bcp, bco, bep, beo = other.cp, other.co, other.ep, other.eo
        cube = CubieCube.__new__(CubieCube)
        cube.cp = tuple([acp[j] for j in bcp])
        cube.co = tuple([(aco[j] + t) % 3 for j, t in zip(bcp, bco)])
        cube.ep = tuple([aep[j] for j in bep])
        cube.eo = tuple([(aeo[j] + f) & 1 for j, f in zip(bep, beo)])
        return cube

    def inverse(self) -> CubieCube:
        cp = [0] * 8
        co = [0] * 8
        ep = [0] * 12
        eo = [0] * 12
        for i, j in enumerate(self.cp):
            cp[j] = i
        for j in range(8):
            co[j] = (-self.co[cp[j]]) % 3
        for i, j in enumerate(self.ep):
            ep[j] = i
        for j in range(12):
            eo[j] = self.eo[ep[j]]
        return CubieCube(cp, co, ep, eo)

    def apply(self, moves: Iterable[int] | str) -> CubieCube:
        """Return the state after applying ``moves`` (move indices or notation)."""
        from .notation import parse

        if isinstance(moves, str):
            moves = parse(moves)
        cube = self
        for m in moves:
            cube = cube * MOVE_CUBES[m]
        return cube

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CubieCube):
            return NotImplemented
        return (self.cp, self.co, self.ep, self.eo) == (other.cp, other.co, other.ep, other.eo)

    def __hash__(self) -> int:
        return hash((self.cp, self.co, self.ep, self.eo))

    def __repr__(self) -> str:
        return f"CubieCube(cp={list(self.cp)}, co={list(self.co)}, ep={list(self.ep)}, eo={list(self.eo)})"

    def is_solved(self) -> bool:
        return self == SOLVED

    # ---- invariants -------------------------------------------------------
    def corner_parity(self) -> int:
        return permutation_parity(self.cp)

    def edge_parity(self) -> int:
        return permutation_parity(self.ep)

    def validity_errors(self) -> list[str]:
        """Reasons why this is not a reachable cube state (empty if it is)."""
        errors = []
        if sorted(self.cp) != list(range(8)):
            errors.append("corner permutation is not a permutation of 8 corners")
        if sorted(self.ep) != list(range(12)):
            errors.append("edge permutation is not a permutation of 12 edges")
        if any(t not in (0, 1, 2) for t in self.co) or sum(self.co) % 3:
            errors.append("corner twists do not sum to 0 mod 3")
        if any(f not in (0, 1) for f in self.eo) or sum(self.eo) % 2:
            errors.append("edge flips do not sum to 0 mod 2")
        if not errors and self.corner_parity() != self.edge_parity():
            errors.append("corner and edge permutation parities differ")
        return errors

    def is_valid(self) -> bool:
        return not self.validity_errors()

    # ---- conversion -------------------------------------------------------
    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.cp, dtype=np.int8),
            np.array(self.co, dtype=np.int8),
            np.array(self.ep, dtype=np.int8),
            np.array(self.eo, dtype=np.int8),
        )


def permutation_parity(perm: Sequence[int]) -> int:
    """0 for an even permutation, 1 for an odd one."""
    seen = [False] * len(perm)
    parity = 0
    for start in range(len(perm)):
        if seen[start]:
            continue
        length = 0
        j = start
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        parity ^= (length - 1) & 1
    return parity


SOLVED = CubieCube()


def _build_move_cubes() -> list[CubieCube]:
    cubes = []
    for face in FACES:
        quarter = CubieCube(*_BASIC[face])
        power = quarter
        for _ in range(3):
            cubes.append(power)
            power = power * quarter
    return cubes


#: ``MOVE_CUBES[3 * face + power - 1]`` for faces in ``FACES`` and powers 1, 2, 3.
MOVE_CUBES: list[CubieCube] = _build_move_cubes()

#: The same move definitions as int8 arrays of shape (18, 8) and (18, 12), for numba kernels.
MOVE_CP = np.array([m.cp for m in MOVE_CUBES], dtype=np.int8)
MOVE_CO = np.array([m.co for m in MOVE_CUBES], dtype=np.int8)
MOVE_EP = np.array([m.ep for m in MOVE_CUBES], dtype=np.int8)
MOVE_EO = np.array([m.eo for m in MOVE_CUBES], dtype=np.int8)
