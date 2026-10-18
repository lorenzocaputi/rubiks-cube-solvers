import numpy as np
import pytest

from rubiks_cube_solvers.cube import random_state


@pytest.fixture(scope="session")
def random_states():
    """200 seeded random-state 3x3x3 scrambles."""
    rng = np.random.default_rng(20261017)
    return [random_state(rng) for _ in range(200)]
