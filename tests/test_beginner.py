from rubiks_cube_solvers.cube import SOLVED
from rubiks_cube_solvers.solvers import beginner
from rubiks_cube_solvers.solvers.beginner import BeginnerSolver, from_side


def first_two_layers_solved(cube) -> bool:
    return all(cube.cp[i] == i and cube.co[i] == 0 for i in range(4, 8)) and all(
        cube.ep[i] == i and cube.eo[i] == 0 for i in range(4, 12)
    )


def test_last_layer_algorithms_keep_the_first_two_layers():
    for alg in (*beginner.EDGE_OLL, beginner.SUNE, beginner.A_PERM, *beginner.U_PERMS):
        for side in range(4):
            assert first_two_layers_solved(SOLVED.apply(from_side(alg, side)))


def test_a_perm_cycles_three_corners_only():
    cube = SOLVED.apply(beginner.A_PERM)
    assert cube.ep == SOLVED.ep and cube.eo == SOLVED.eo and cube.co == SOLVED.co
    assert sum(cube.cp[i] != i for i in range(8)) == 3


def test_u_perms_cycle_three_edges_only():
    for alg in beginner.U_PERMS:
        cube = SOLVED.apply(alg)
        assert cube.cp == SOLVED.cp and cube.co == SOLVED.co and cube.eo == SOLVED.eo
        assert sum(cube.ep[i] != i for i in range(12)) == 3


def test_solves_random_states(random_states):
    solver = BeginnerSolver()
    for cube in random_states:
        solution = solver.solve(cube)
        assert cube.apply(solution.moves).is_solved()
        assert 40 <= len(solution.moves) <= 250
