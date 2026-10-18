"""What every solver returns."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Solution:
    moves: list[int]
    nodes: int  # states expanded, i.e. whose successors were generated
    seconds: float
    info: dict = field(default_factory=dict)
