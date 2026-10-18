"""Lookup tables cached as .npy files under ``~/.cache/rubiks-cube-solvers``.

Set ``RUBIKS_CUBE_SOLVERS_CACHE`` to use another directory.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

TABLE_VERSION = 1
_MEMORY: dict[str, np.ndarray] = {}


def cache_dir() -> Path:
    path = os.environ.get("RUBIKS_CUBE_SOLVERS_CACHE")
    root = Path(path) if path else Path.home() / ".cache" / "rubiks-cube-solvers"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cached(name: str, build: Callable[[], np.ndarray], verbose: bool = True) -> np.ndarray:
    """Load table ``name`` from memory or disk, building and saving it on first use."""
    if name in _MEMORY:
        return _MEMORY[name]
    path = cache_dir() / f"{name}-v{TABLE_VERSION}.npy"
    if path.exists():
        table = np.asarray(np.load(path, mmap_mode="r"))
    else:
        start = time.perf_counter()
        table = np.ascontiguousarray(build())
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            np.save(f, table)
        os.replace(tmp, path)
        if verbose:
            mb = table.nbytes / 1e6
            print(f"built table {name} ({mb:.1f} MB) in {time.perf_counter() - start:.1f} s")
    _MEMORY[name] = table
    return table
