"""Facelet (sticker) representation and its geometry.

The 54 stickers are numbered face by face in the order U R F D L B, each face
row by row as seen from outside with the usual net orientation:

                 U0 U1 U2
                 U3 U4 U5
                 U6 U7 U8
     L0 L1 L2    F0 F1 F2    R0 R1 R2    B0 B1 B2
     L3 L4 L5    F3 F4 F5    R3 R4 R5    B3 B4 B5
     L6 L7 L8    F6 F7 F8    R6 R7 R8    B6 B7 B8
                 D0 D1 D2
                 D3 D4 D5
                 D6 D7 D8

A facelet string lists the face letter of each sticker's colour in that order,
so the solved cube is ``"UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"``.
"""

from __future__ import annotations

import numpy as np

from .cubie import CORNERS, EDGES, FACES, CubieCube

SOLVED_FACELETS = "".join(f * 9 for f in FACES)


def _index(name: str) -> int:
    return 9 * FACES.index(name[0]) + int(name[1]) - 1


# Stickers of each corner and edge position, U/D sticker first, then clockwise.
CORNER_FACELETS = tuple(
    tuple(_index(n) for n in names)
    for names in (
        ("U9", "R1", "F3"),
        ("U7", "F1", "L3"),
        ("U1", "L1", "B3"),
        ("U3", "B1", "R3"),
        ("D3", "F9", "R7"),
        ("D1", "L9", "F7"),
        ("D7", "B9", "L7"),
        ("D9", "R9", "B7"),
    )
)
EDGE_FACELETS = tuple(
    tuple(_index(n) for n in names)
    for names in (
        ("U6", "R2"),
        ("U8", "F2"),
        ("U4", "L2"),
        ("U2", "B2"),
        ("D6", "R8"),
        ("D2", "F8"),
        ("D4", "L8"),
        ("D8", "B8"),
        ("F6", "R4"),
        ("F4", "L6"),
        ("B6", "L4"),
        ("B4", "R6"),
    )
)
CORNER_COLORS = tuple(tuple(name) for name in CORNERS)
EDGE_COLORS = tuple(tuple(name) for name in EDGES)


def to_facelets(cube: CubieCube) -> str:
    stickers = list(SOLVED_FACELETS)
    for i in range(8):
        piece, twist = cube.cp[i], cube.co[i]
        for n in range(3):
            stickers[CORNER_FACELETS[i][(n + twist) % 3]] = CORNER_COLORS[piece][n]
    for i in range(12):
        piece, flip = cube.ep[i], cube.eo[i]
        for n in range(2):
            stickers[EDGE_FACELETS[i][(n + flip) % 2]] = EDGE_COLORS[piece][n]
    return "".join(stickers)


def from_facelets(text: str) -> CubieCube:
    """Convert a facelet string to a cubie state, checking that it is a real cube."""
    text = text.strip().upper()
    if len(text) != 54 or any(c not in FACES for c in text):
        raise ValueError("a facelet string has 54 letters from URFDLB")
    if any(text.count(f) != 9 for f in FACES):
        raise ValueError("each colour must appear on exactly 9 stickers")
    if any(text[9 * k + 4] != FACES[k] for k in range(6)):
        raise ValueError("centres must be in the order URFDLB")
    cp, co, ep, eo = [0] * 8, [0] * 8, [0] * 12, [0] * 12
    for i, facelets in enumerate(CORNER_FACELETS):
        colors = [text[f] for f in facelets]
        twist = next((t for t in range(3) if colors[t] in "UD"), None)
        if twist is None:
            raise ValueError(f"corner position {CORNERS[i]} has no U or D sticker")
        key = (colors[twist], colors[(twist + 1) % 3], colors[(twist + 2) % 3])
        if key not in CORNER_COLORS:
            raise ValueError(f"corner position {CORNERS[i]} holds no real corner: {''.join(colors)}")
        cp[i], co[i] = CORNER_COLORS.index(key), twist
    for i, facelets in enumerate(EDGE_FACELETS):
        colors = tuple(text[f] for f in facelets)
        if colors in EDGE_COLORS:
            ep[i], eo[i] = EDGE_COLORS.index(colors), 0
        elif colors[::-1] in EDGE_COLORS:
            ep[i], eo[i] = EDGE_COLORS.index(colors[::-1]), 1
        else:
            raise ValueError(f"edge position {EDGES[i]} holds no real edge: {''.join(colors)}")
    cube = CubieCube(cp, co, ep, eo)
    errors = cube.validity_errors()
    if errors:
        raise ValueError("; ".join(errors))
    return cube


# ---- geometry -------------------------------------------------------------
# x points to R, y to U and z to F.  Cubie centres have coordinates in {-1, 0, 1}.
FACE_NORMALS = {
    "U": (0, 1, 0),
    "R": (1, 0, 0),
    "F": (0, 0, 1),
    "D": (0, -1, 0),
    "L": (-1, 0, 0),
    "B": (0, 0, -1),
}


def _sticker_cubie(face: str, row: int, col: int) -> tuple[int, int, int]:
    if face == "U":
        return (col - 1, 1, row - 1)
    if face == "D":
        return (col - 1, -1, 1 - row)
    if face == "F":
        return (col - 1, 1 - row, 1)
    if face == "B":
        return (1 - col, 1 - row, -1)
    if face == "R":
        return (1, 1 - row, 1 - col)
    return (-1, 1 - row, col - 1)  # L


#: Centre of the cubie each sticker sits on, shape (54, 3).
STICKER_CUBIE = np.array([_sticker_cubie(face, k // 3, k % 3) for face in FACES for k in range(9)], dtype=int)
#: Outward normal of each sticker, shape (54, 3).
STICKER_NORMAL = np.array([FACE_NORMALS[face] for face in FACES for _ in range(9)], dtype=int)


def rotation_matrix(axis: tuple[int, int, int], quarter_turns: int) -> np.ndarray:
    """Rotation by ``-90 * quarter_turns`` degrees about ``axis``: clockwise seen from the axis tip."""
    angle = -np.pi / 2 * quarter_turns
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.rint(np.eye(3) + s * k + (1 - c) * k @ k).astype(int)


def facelet_move_permutation(face: str) -> list[int]:
    """``perm[i]`` is the sticker that a clockwise turn of ``face`` carries to position ``i``."""
    normal = np.array(FACE_NORMALS[face])
    rot = rotation_matrix(FACE_NORMALS[face], 1)
    lookup = {(tuple(STICKER_CUBIE[i]), tuple(STICKER_NORMAL[i])): i for i in range(54)}
    perm = list(range(54))
    for i in range(54):
        if STICKER_CUBIE[i] @ normal == 1:
            target = lookup[(tuple(rot @ STICKER_CUBIE[i]), tuple(rot @ STICKER_NORMAL[i]))]
            perm[target] = i
    return perm


def apply_facelet_move(text: str, face: str) -> str:
    perm = facelet_move_permutation(face)
    return "".join(text[perm[i]] for i in range(54))
