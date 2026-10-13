import numpy as np
from scipy.special import iv
from scipy.stats import rv_discrete


class mm1_len(rv_discrete):
    """
    Distribution of the number of customers in an M/M/1 queue.

    This is the transient queue-length distribution ``p_k(t)`` for an M/M/1
    queue with arrival rate ``lam``, service rate ``mu``, and ``i`` customers
    initially in the system. The parameters are passed to the methods in the
    order ``(lam, mu, i, t)``.

    Examples
    --------
    >>> d = mm1_len()
    >>> float(d.pmf(3, 1.0, 1.0, 3, 0.0))
    1.0
    >>> float(d.pmf(0, 1.0, 1.0, 3, 0.0))
    0.0
    """

    def _argcheck(self, lam: float, mu: float, i: int, t: float) -> bool:
        """Validate arrival rate, service rate, initial customers, and time."""
        return (lam > 0) & (mu > 0) & (i >= 0) & (t >= 0)

    def _pmf(self, k: int, lam: float, mu: float, i: int, t: float) -> float:
        """
        Compute the transient probability ``p_k(t)`` for an M/M/1 queue.

        Parameters
        ----------
        k : int
            Number of customers in the queue.
        lam : float
            Arrival rate parameter.
        mu : float
            Service rate parameter.
        i : int
            Initial number of customers in the queue.
        t : float
            Time parameter.

        Returns
        -------
        float
            The computed probability p_k(t).
        """

        def scalar_pmf(kk: float, lam: float, mu: float, i: int, t: float) -> float:
            rho = lam / mu
            at = 2 * np.sqrt(lam * mu) * t
            exp_decay_factor = np.exp(-(lam + mu) * t)
            expansion_factor = (1 - rho) * np.power(rho, kk)

            term1 = np.power(rho, (kk - i) / 2) * iv(kk - i, at)
            term2 = np.power(rho, (kk - i - 1) / 2) * iv(kk + i + 1, at)

            expansion = 0.0
            max_terms = 10_000
            j0 = int(round(kk)) + i + 2
            for j in range(j0, j0 + max_terms):
                new_term = np.power(rho, -j / 2) * iv(j, at)
                expansion += new_term
                if np.abs(new_term) < 1e-15:
                    break

            return exp_decay_factor * (term1 + term2 + expansion_factor * expansion)

        lam_a, mu_a, i_a, t_a, k_arr = np.broadcast_arrays(
            np.asarray(lam, dtype=float),
            np.asarray(mu, dtype=float),
            np.asarray(i, dtype=float),
            np.asarray(t, dtype=float),
            np.asarray(k, dtype=float),
        )

        out = np.empty(k_arr.shape, dtype=float)
        for idx in np.ndindex(k_arr.shape):
            out[idx] = scalar_pmf(
                float(k_arr[idx]),
                float(lam_a[idx]),
                float(mu_a[idx]),
                int(i_a[idx]),
                float(t_a[idx]),
            )

        return out


def sample_mm1_len_seq(t, k0, taus, lambdas, mus):
    """
    Simulate one M/M/1 queue-length path with piecewise-constant rates.

    The rates change at the times in `taus`; each interval uses the
    corresponding entry of `lambdas` and `mus`. A single sample of the
    queue length is returned at time `t`, which must lie after the last
    rate change.

    Parameters
    ----------
    t : float
        Time at which to sample the queue length (greater than ``max(taus)``).
    k0 : int
        Initial number of customers in the queue.
    taus : sequence of float
        Strictly increasing times at which the rates change.
    lambdas : sequence of float
        Arrival rates for each interval.
    mus : sequence of float
        Service rates for each interval.

    Returns
    -------
    int
        A sampled queue length at time ``t``.
    """
    dist = mm1_len()

    taus = np.asarray(taus, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    mus = np.asarray(mus, dtype=float)

    if not (len(taus) == len(lambdas) == len(mus)):
        raise ValueError("taus, lambdas, and mus must have the same length.")
    if not np.all(np.diff(taus) > 0):
        raise ValueError("taus must be strictly increasing.")
    if t <= np.max(taus):
        raise ValueError("t must be greater than max(taus).")

    k = k0
    for tau, lam, mu in zip(taus, lambdas, mus):
        k = dist.rvs(lam, mu, int(k), tau)

    return dist.rvs(lambdas[-1], mus[-1], int(k), t)


if __name__ == "__main__":
    np.random.seed(2018)
    print(sample_mm1_len_seq(10, 5, list(range(1, 9)), [10] * 8, [10] * 8))
