import numpy as np
import pytest

from rubiks_cube_solvers.cube import cube2
from rubiks_cube_solvers.solvers.learned import MODEL_PATH, LearnedSolver, children, encode, random_walks


def test_encoding_matches_sticker_colours():
    rng = np.random.default_rng(0)
    states = rng.integers(cube2.N_STATES, size=50)
    onehot = encode(states).reshape(50, 24, 6)
    assert (onehot.sum(axis=2) == 1).all()
    for k, index in enumerate(states):
        assert (onehot[k].argmax(axis=1) == cube2.sticker_colors(int(index))).all()


def test_children_and_walks_stay_in_range():
    perm_table, twist_table = cube2.move_tables()
    walks = random_walks(1, 500, 14, perm_table, twist_table)
    assert walks.min() >= 0 and walks.max() < cube2.N_STATES
    succ = children(walks[:20], perm_table, twist_table)
    for index, row in zip(walks[:20], succ):
        assert row.tolist() == [cube2.apply_moves(int(index), [m]) for m in range(9)]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="no trained model")
def test_weighted_a_star_solves_random_states():
    solver = LearnedSolver()
    dist = cube2.distance_table()
    rng = np.random.default_rng(11)
    for _ in range(200):
        index = cube2.random_state(rng)
        solution = solver.solve(index)
        assert cube2.apply_moves(index, solution.moves) == cube2.SOLVED_INDEX
        assert len(solution.moves) >= dist[index]  # never shorter than the BFS distance
