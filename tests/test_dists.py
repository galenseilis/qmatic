import numpy as np
import pytest

from qmatic.dists import mm1_len, sample_mm1_len_seq


def test_pmf_is_point_mass_at_zero_time():
    d = mm1_len()
    assert float(d.pmf(5, 1.0, 1.0, 5, 0.0)) == pytest.approx(1.0)
    assert float(d.pmf(4, 1.0, 1.0, 5, 0.0)) == pytest.approx(0.0)
    assert float(d.pmf(6, 1.0, 1.0, 5, 0.0)) == pytest.approx(0.0)


def test_pmf_sums_to_one():
    d = mm1_len()
    for lam, mu, t in [(1.0, 2.0, 0.5), (1.0, 2.0, 2.0), (2.0, 3.0, 3.0)]:
        values = np.array([d.pmf(k, lam, mu, 0, t) for k in range(300)])
        assert values.sum() == pytest.approx(1.0, abs=1e-6)


def test_pmf_is_nonnegative():
    d = mm1_len()
    values = np.array([d.pmf(k, 1.0, 2.0, 3, 1.5) for k in range(50)])
    assert np.all(values >= -1e-12)


def test_pmf_approaches_steady_state():
    d = mm1_len()
    lam, mu = 0.001, 0.002
    rho = lam / mu
    for k in range(10):
        expected = (1 - rho) * rho**k
        assert float(d.pmf(k, lam, mu, 0, 100000.0)) == pytest.approx(expected, abs=1e-9)


def test_pmf_accepts_array_input():
    d = mm1_len()
    out = d.pmf(np.array([0, 1, 2]), 1.0, 2.0, 0, 1.0)
    assert out.shape == (3,)


def test_sample_mm1_len_seq_returns_nonnegative_int():
    np.random.seed(0)
    x = sample_mm1_len_seq(10.0, 5, [1.0, 2.0, 3.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0])
    assert isinstance(x, (int, np.integer))
    assert x >= 0


def test_sample_mm1_len_seq_validates_inputs():
    with pytest.raises(ValueError):  # length mismatch
        sample_mm1_len_seq(10.0, 5, [1.0, 2.0, 3.0], [1.0, 1.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):  # taus not strictly increasing
        sample_mm1_len_seq(10.0, 5, [1.0, 1.0, 3.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):  # t not beyond the last rate change
        sample_mm1_len_seq(2.0, 5, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
