import math

import pytest

from rubiks_cube_solvers.cube import cube_group_order
from rubiks_cube_solvers.solvers.thistlethwaite import ThistlethwaiteSolver, half_turn_corner_group


@pytest.fixture(scope="module")
def solver():
    return ThistlethwaiteSolver()


def test_half_turn_group_has_96_corner_permutations():
    assert len(half_turn_corner_group()) == 96


def test_coset_spaces(solver):
    sizes = [int((d >= 0).sum()) for d in solver.dist]
    assert sizes == [2048, 1_082_565, 29_400, 663_552]
    assert math.prod(sizes) == cube_group_order()
    assert [int(d.max()) for d in solver.dist] == [7, 10, 13, 15]


def test_solves_random_states(solver, random_states):
    for cube in random_states:
        solution = solver.solve(cube)
        assert cube.apply(solution.moves).is_solved()
        assert len(solution.moves) <= 45
