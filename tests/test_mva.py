import numpy as np
import pytest

from qmatic.mva import matrix_minor, mean_value_analysis, visit_ratio_vector

STOCHASTIC_P = np.array(
    [
        [0.2, 0.5, 0.3],
        [0.4, 0.1, 0.5],
        [0.3, 0.2, 0.5],
    ]
)


def test_matrix_minor_removes_row_and_column():
    arr = np.arange(1, 10).reshape(3, 3)
    np.testing.assert_array_equal(matrix_minor(arr, 1, 2), [[1, 2], [7, 8]])


def test_matrix_minor_is_non_destructive():
    arr = np.arange(1, 10).reshape(3, 3)
    original = arr.copy()
    matrix_minor(arr, 0, 0)
    np.testing.assert_array_equal(arr, original)


def test_visit_ratio_two_node_cycle():
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    v = visit_ratio_vector(P)
    np.testing.assert_allclose(v, [0.5, 0.5])
    assert v.sum() == pytest.approx(1.0)


def test_visit_ratio_is_nonnegative_and_satisfies_balance():
    v = visit_ratio_vector(STOCHASTIC_P)
    assert np.all(v >= 0.0)
    np.testing.assert_allclose(v, v @ STOCHASTIC_P, atol=1e-12)


def test_visit_ratio_rejects_non_square():
    with pytest.raises(ValueError):
        visit_ratio_vector(np.ones((2, 3)))


def test_visit_ratio_rejects_non_stochastic():
    with pytest.raises(ValueError):
        visit_ratio_vector(np.array([[0.5, 0.2], [0.1, 0.1]]))


def test_mva_single_queue_matches_closed_form():
    L, W, lambdas, v = mean_value_analysis(np.zeros(1), 4, np.array([2.0]), np.array([[1.0]]))
    np.testing.assert_allclose(L[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(lambdas, [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(v, [1.0])


def test_mva_conserves_customers():
    M = 6
    L, W, lambdas, v = mean_value_analysis(np.zeros(3), M, np.array([0.2, 0.4, 0.3]), STOCHASTIC_P)
    assert L.shape == (M + 1, 3)
    assert W.shape == (M, 3)
    assert lambdas.shape == (M,)
    np.testing.assert_allclose(L.sum(axis=1), np.arange(M + 1))
    np.testing.assert_allclose(L[0], np.zeros(3))


def test_mva_rejects_bad_inputs():
    mu = np.array([0.2, 0.4, 0.3])
    with pytest.raises(ValueError):  # P not square
        mean_value_analysis(np.zeros(3), 5, mu, STOCHASTIC_P[:-1])
    with pytest.raises(ValueError):  # non-positive service rate
        mean_value_analysis(np.zeros(3), 5, np.array([0.0, 0.4, 0.3]), STOCHASTIC_P)
    with pytest.raises(ValueError):  # L_0 length mismatch
        mean_value_analysis(np.zeros(2), 5, mu, STOCHASTIC_P)
    with pytest.raises(ValueError):  # M must be positive
        mean_value_analysis(np.zeros(3), 0, mu, STOCHASTIC_P)
