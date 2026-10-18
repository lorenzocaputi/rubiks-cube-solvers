import numpy as np

from rubiks_cube_solvers.cube import SOLVED, cube2

# https://oeis.org/A079761 and https://oeis.org/A079762, fetched 2026-10-17.
A079761_HTM = [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644]
A079762_QTM = [1, 6, 27, 120, 534, 2256, 8969, 33058, 114149, 360508, 930588, 1350852, 782536, 90280, 276]


def test_htm_distance_distribution_matches_oeis():
    dist = cube2.distance_table("htm")
    assert dist.min() == 0
    assert np.bincount(dist).tolist() == A079761_HTM
    assert dist.max() == 11  # God's number for the 2x2x2 in HTM


def test_qtm_distance_distribution_matches_oeis():
    dist = cube2.distance_table("qtm")
    assert np.bincount(dist).tolist() == A079762_QTM


def test_index_round_trip_and_moves():
    rng = np.random.default_rng(0)
    for _ in range(200):
        index = cube2.random_state(rng)
        cube = cube2.state_cube(index)
        assert cube2.state_index(cube) == index
        moves = [int(m) for m in rng.integers(0, 9, size=10)]
        assert cube2.state_index(cube.apply(moves)) == cube2.apply_moves(index, moves)
    assert cube2.state_index(SOLVED) == cube2.SOLVED_INDEX
