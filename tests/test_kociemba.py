import pytest

from rubiks_cube_solvers.cube import SOLVED, parse
from rubiks_cube_solvers.solvers.kociemba import KociembaSolver


@pytest.fixture(scope="module")
def solver():
    return KociembaSolver(target=21)


def test_trivial_cases(solver):
    assert solver.solve(SOLVED).moves == []
    assert KociembaSolver(target=1).solve(SOLVED.apply("R")).moves == parse("R'")
    cube = SOLVED.apply("R U2 F' D L2")
    assert cube.apply(solver.solve(cube).moves).is_solved()
    assert len(KociembaSolver(target=5).solve(cube).moves) == 5


def test_solves_random_states(solver, random_states):
    for cube in random_states:
        solution = solver.solve(cube)
        assert cube.apply(solution.moves).is_solved()
        assert len(solution.moves) <= 22
