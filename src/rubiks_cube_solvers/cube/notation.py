"""Singmaster notation for face turns.

A move is an integer ``3 * face + power - 1`` with faces in the order U R F D L B
and powers 1 (clockwise quarter turn), 2 (half turn) and 3 (counter-clockwise).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .cubie import FACES, OPPOSITE

_SUFFIX = {1: "", 2: "2", 3: "'"}
_TOKEN = re.compile(r"([URFDLB])(2'|2|'|)")


def move_name(move: int) -> str:
    return FACES[move // 3] + _SUFFIX[move % 3 + 1]


MOVE_NAMES = tuple(move_name(m) for m in range(18))


def parse(text: str) -> list[int]:
    """Parse e.g. ``"R U R' U' F2"``.  ``2'`` is read as a half turn."""
    moves = []
    for token in text.split():
        match = _TOKEN.fullmatch(token)
        if match is None:
            raise ValueError(f"not a face turn: {token!r}")
        face = FACES.index(match.group(1))
        power = {"": 1, "2": 2, "2'": 2, "'": 3}[match.group(2)]
        moves.append(3 * face + power - 1)
    return moves


def format_moves(moves: Iterable[int]) -> str:
    return " ".join(MOVE_NAMES[m] for m in moves)


def invert(moves: Sequence[int]) -> list[int]:
    return [3 * (m // 3) + 2 - m % 3 for m in reversed(moves)]


def htm(moves: Sequence[int]) -> int:
    """Half-turn metric: every face turn counts 1."""
    return len(moves)


def qtm(moves: Sequence[int]) -> int:
    """Quarter-turn metric: half turns count 2."""
    return sum(2 if m % 3 == 1 else 1 for m in moves)


def simplify(moves: Iterable[int]) -> list[int]:
    """Merge consecutive turns of one face, also across a turn of the opposite face.

    ``U U`` becomes ``U2``, ``U U'`` disappears and ``U D U`` becomes ``U2 D``.
    """
    stack: list[list[int]] = []  # [face, power]
    for m in moves:
        face, power = m // 3, m % 3 + 1
        if stack and stack[-1][0] == face:
            target = len(stack) - 1
        elif len(stack) >= 2 and stack[-1][0] == OPPOSITE[face] and stack[-2][0] == face:
            target = len(stack) - 2
        else:
            stack.append([face, power])
            continue
        stack[target][1] = (stack[target][1] + power) % 4
        if stack[target][1] == 0:
            del stack[target]
    return [3 * face + power - 1 for face, power in stack]
