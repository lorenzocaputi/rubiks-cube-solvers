import os
from itertools import product

import numpy as np
import pytest

from rubiks_cube_solvers.cube import SOLVED, cube2, random_moves
from rubiks_cube_solvers.solvers.korf import Korf2Solver, KorfSolver

FULL = os.environ.get("RUBIKS_FULL_TESTS") == "1"


@pytest.fixture(scope="module")
def solver():
    # The quick tables are small but still admissible, so solutions are still optimal.
    return KorfSolver(quick=not FULL)


def brute_force_distance(cube, limit):
    for depth in range(limit + 1):
        for moves in product(range(18), repeat=depth):
            if cube.apply(moves).is_solved():
                return depth
    return None


def test_optimal_on_very_short_scrambles(solver):
    rng = np.random.default_rng(7)
    for _ in range(40):
        cube = SOLVED.apply(random_moves(rng, int(rng.integers(0, 4))))
        assert len(solver.solve(cube).moves) == brute_force_distance(cube, 3)


def test_solves_short_scrambles(solver):
    rng = np.random.default_rng(8)
    length = 12 if FULL else 7
    for _ in range(200):
        cube = SOLVED.apply(random_moves(rng, length))
        solution = solver.solve(cube)
        assert cube.apply(solution.moves).is_solved()
        assert len(solution.moves) <= length


def test_2x2_matches_bfs_distance():
    solver = Korf2Solver()
    dist = cube2.distance_table()
    rng = np.random.default_rng(9)
    for _ in range(300):
        index = cube2.random_state(rng)
        solution = solver.solve(index)
        assert cube2.apply_moves(index, solution.moves) == cube2.SOLVED_INDEX
        assert len(solution.moves) == dist[index]
