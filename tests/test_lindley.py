import numpy as np
import pytest

from qmatic.lindley import lindley


def test_initial_value_preserved():
    L = lindley(7.0, np.array([1.0, 2.0]), np.array([0.0, 1.0]))
    assert L[0] == pytest.approx(7.0)


def test_matches_manual_recursion():
    A = np.array([1.0, 2.0, 1.0])
    D = np.array([0.0, 2.0, 1.0])
    np.testing.assert_allclose(lindley(0.0, A, D), [0.0, 1.0, 1.0, 1.0])


def test_follows_lindley_recurrence():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 5, size=20).astype(float)
    D = rng.integers(0, 5, size=20).astype(float)
    L = lindley(3.0, A, D)
    for t in range(A.size):
        assert L[t + 1] == pytest.approx(max(0.0, L[t] + A[t] - D[t]))


def test_output_is_nonnegative():
    rng = np.random.default_rng(1)
    A = rng.integers(0, 3, size=50).astype(float)
    D = rng.integers(0, 8, size=50).astype(float)
    assert np.all(lindley(0.0, A, D) >= 0.0)


def test_size_mismatch_raises():
    with pytest.raises(ValueError):
        lindley(0.0, np.array([1.0, 2.0]), np.array([1.0]))
