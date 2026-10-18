import numpy as np
import pytest

from rubiks_cube_solvers.cube import (
    FACES,
    MOVE_CUBES,
    SOLVED,
    SOLVED_FACELETS,
    CubieCube,
    cube_group_order,
    format_moves,
    from_facelets,
    htm,
    invert,
    parse,
    qtm,
    random_moves,
    random_state,
    schreier_sims_order,
    simplify,
    to_facelets,
)
from rubiks_cube_solvers.cube.facelet import apply_facelet_move, facelet_move_permutation


def order(cube: CubieCube) -> int:
    x, n = cube, 1
    while not x.is_solved():
        x, n = x * cube, n + 1
    return n


def test_move_orders():
    for m, cube in enumerate(MOVE_CUBES):
        assert order(cube) == (2 if m % 3 == 1 else 4)
    assert order(SOLVED.apply("R U")) == 105
    assert order(SOLVED.apply("R U R' U'")) == 6


def test_cubie_moves_match_sticker_geometry():
    # The cubie move tables are typed in by hand; the sticker permutations come from
    # rotating 3D sticker coordinates.  They must agree on arbitrary states.
    rng = np.random.default_rng(1)
    for _ in range(50):
        state = random_state(rng)
        for f, face in enumerate(FACES):
            assert to_facelets(state * MOVE_CUBES[3 * f]) == apply_facelet_move(to_facelets(state), face)


def test_facelet_round_trip():
    rng = np.random.default_rng(2)
    assert to_facelets(SOLVED) == SOLVED_FACELETS
    for _ in range(200):
        state = random_state(rng)
        assert state.is_valid()
        assert from_facelets(to_facelets(state)) == state


def test_inverse():
    rng = np.random.default_rng(3)
    for _ in range(50):
        state = random_state(rng)
        assert (state * state.inverse()).is_solved()
        assert (state.inverse() * state).is_solved()
        moves = random_moves(rng, 25)
        assert SOLVED.apply(moves).apply(invert(moves)).is_solved()


def test_validity_checks():
    twisted = CubieCube(co=[1, 0, 0, 0, 0, 0, 0, 0])
    flipped = CubieCube(eo=[1] + [0] * 11)
    swapped = CubieCube(ep=[1, 0] + list(range(2, 12)))
    assert "corner twists" in twisted.validity_errors()[0]
    assert "edge flips" in flipped.validity_errors()[0]
    assert "parities differ" in swapped.validity_errors()[0]
    assert SOLVED.is_valid()
    with pytest.raises(ValueError, match="twists"):
        from_facelets(to_facelets(twisted))
    with pytest.raises(ValueError, match="parities"):
        from_facelets(to_facelets(swapped))
    with pytest.raises(ValueError, match="54 letters"):
        from_facelets("U" * 53)
    with pytest.raises(ValueError, match="exactly 9"):
        from_facelets("U" * 10 + SOLVED_FACELETS[10:])


def test_notation():
    moves = parse("R U R' U' F2 B2' D")
    assert format_moves(moves) == "R U R' U' F2 B2 D"
    assert format_moves(invert(parse("R U F'"))) == "F U' R'"
    assert htm(parse("R U2 F'")) == 3
    assert qtm(parse("R U2 F'")) == 4
    assert format_moves(simplify(parse("R R U U' F B F'"))) == "R2 B"
    assert format_moves(simplify(parse("U D U D'"))) == "U2"
    assert simplify(parse("R L R' L'")) == []
    with pytest.raises(ValueError):
        parse("R X")


def test_random_moves_do_not_cancel():
    rng = np.random.default_rng(4)
    for _ in range(100):
        moves = random_moves(rng, 20)
        assert simplify(moves) == moves


def test_random_state_parities_are_balanced():
    rng = np.random.default_rng(5)
    parities = [random_state(rng).corner_parity() for _ in range(2000)]
    assert 0.45 < np.mean(parities) < 0.55


def test_group_order():
    assert cube_group_order() == 43_252_003_274_489_856_000
    generators = [facelet_move_permutation(face) for face in FACES]
    assert schreier_sims_order(generators) == 43_252_003_274_489_856_000
