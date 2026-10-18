"""Draw a cube in 3D (U, F and R faces visible) with Pillow, including a turning layer."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .cube.facelet import FACE_NORMALS, STICKER_CUBIE, STICKER_NORMAL

STICKER_COLORS = {
    "U": (244, 244, 240),
    "R": (200, 32, 52),
    "F": (0, 155, 72),
    "D": (255, 210, 0),
    "L": (255, 112, 0),
    "B": (0, 75, 180),
}
BODY = (24, 24, 24)


def _rotation(axis, angle: float) -> np.ndarray:
    """Rotation by ``angle`` radians about ``axis`` (counter-clockwise seen from its tip)."""
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


# Camera space: x right, y up, z towards the viewer.
VIEW = _rotation((1, 0, 0), np.radians(30)) @ _rotation((0, 1, 0), np.radians(-38))


def _square(center: np.ndarray, normal: np.ndarray, half: float) -> list[np.ndarray]:
    u = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.5 else np.array([0, 1.0, 0])
    v = np.cross(normal, u)
    return [center + half * (a * u + b * v) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def sticker_map(facelets: str, n: int = 3) -> dict[tuple[int, ...], str]:
    """Map (cubie x, y, z, normal x, y, z) to the colour letter of each visible sticker.

    For ``n = 2`` only the corner stickers of the 3x3x3 facelet string are kept."""
    out = {}
    for i in range(54):
        if n == 2 and np.abs(STICKER_CUBIE[i]).sum() != 3:
            continue
        out[(*map(int, STICKER_CUBIE[i]), *map(int, STICKER_NORMAL[i]))] = facelets[i]
    return out


def draw_cube(
    stickers: dict[tuple[int, ...], str],
    size: int,
    n: int = 3,
    turn: tuple[str, float] | None = None,
    background=(255, 255, 255),
) -> Image.Image:
    """Render one cube of ``size`` x ``size`` pixels.

    ``turn = (face, angle)`` shows that face's layer rotated by ``angle`` radians
    clockwise as seen from the face, for frames in the middle of a move.
    """
    ss = 3  # supersampling factor
    img = Image.new("RGB", (size * ss, size * ss), background)
    draw = ImageDraw.Draw(img)
    spacing, half = (2 / 3, 1 / 3) if n == 3 else (0.5, 0.5)

    layer_rotation = np.eye(3)
    axis = None
    if turn is not None:
        axis = np.array(FACE_NORMALS[turn[0]], dtype=float)
        layer_rotation = _rotation(axis, -turn[1])

    quads = []
    for c in sorted({key[:3] for key in stickers}):
        moving = axis is not None and np.dot(c, axis) > 0
        rot = VIEW @ (layer_rotation if moving else np.eye(3))
        center = np.array(c, dtype=float) * spacing
        depth = (rot @ center)[2]
        for k in range(3):
            for sign in (-1.0, 1.0):
                normal = np.zeros(3)
                normal[k] = sign
                if (rot @ normal)[2] <= 1e-6:
                    continue  # facing away from the camera
                face_center = center + normal * half
                quads.append((moving, depth, 0, [rot @ p for p in _square(face_center, normal, half)], BODY))
                color = stickers.get((*c, *map(int, normal)))
                if color is not None:
                    pts = [rot @ p for p in _square(face_center + normal * 1e-3, normal, half * 0.84)]
                    quads.append((moving, depth, 1, pts, STICKER_COLORS[color]))

    # The turning layer and the rest are separated by a plane: draw the farther part first.
    moving_first = axis is not None and (VIEW @ axis)[2] < 0
    quads.sort(key=lambda q: (q[0] != moving_first, q[1], q[2]))
    extent = 1.95
    for _, _, _, pts, color in quads:
        xy = [((p[0] / extent + 1) * size * ss / 2, (1 - p[1] / extent) * size * ss / 2) for p in pts]
        draw.polygon(xy, fill=color)
    return img.resize((size, size), Image.LANCZOS)
