import numpy as np
from numpy.typing import NDArray


def matrix_minor(arr: NDArray, i: int, j: int) -> NDArray:
    """
    Compute the minor matrix obtained by removing the specified row and column
    from the input 2D NumPy array.

    Parameters:
    -----------
    arr : numpy.ndarray
        Input 2D array from which the minor matrix is obtained.
    i : int
        Index of the row to be removed.
    j : int
        Index of the column to be removed.

    Returns:
    --------
    minor_matrix : numpy.ndarray
        Minor matrix obtained by removing the specified row and column.

    Notes:
    ------
    This function does not modify the original array; instead, it creates and returns
    a new array representing the minor matrix.

    Examples:
    ---------
    >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> matrix_minor(arr, 1, 2)
    array([[1, 2],
           [7, 8]])
    """
    return np.delete(np.delete(arr, i, axis=0), j, axis=1)


def visit_ratio_vector(P: NDArray) -> NDArray:
    """
    Compute the visit ratio vector `v` for a closed queueing network.

    The visit ratios are the stationary probability vector of the routing
    matrix `P`, i.e. the unique positive solution (up to scale) of
    ``v = v P``. The returned vector is normalized to sum to one, which is the
    natural choice because the mean value analysis that consumes it is
    invariant to the scale of `v`.

    Parameters:
    -----------
    P : numpy.ndarray
        Square customer routing matrix where ``P[i, j]`` is the probability
        that a customer finishing service at node `i` moves to node `j`. Each
        row must sum to one (closed network).

    Returns:
    --------
    v : numpy.ndarray
        Visit ratio vector with one entry per node, summing to one.

    Raises:
    -------
    ValueError
        If `P` is not square, its rows do not sum to one, or a positive
        stationary vector cannot be found.

    Examples:
    --------
    >>> P = np.array([[0.0, 1.0], [1.0, 0.0]])
    >>> visit_ratio_vector(P)
    array([0.5, 0.5])
    """
    P = np.asarray(P, dtype=float)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2-D array.")

    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Rows of P must sum to one for a closed network.")

    dim = P.shape[0]

    # Solve v = v P, i.e. (I - P.T) v = 0, via the null space from the SVD.
    _, _, vt = np.linalg.svd(np.eye(dim) - P.T)
    v = vt[-1]

    if v.sum() < 0:
        v = -v

    if np.any(v < -1e-9):
        raise ValueError("Could not compute a positive visit-ratio vector.")

    total = v.sum()
    if np.isclose(total, 0.0):
        raise ValueError("Could not compute a positive visit-ratio vector.")

    return v / total


def mean_value_analysis(L_0: NDArray, M: int, mu: NDArray, P: NDArray):
    """
    Perform mean value analysis on a closed queueing network with K M/M/1 queues.

    Parameters:
    -----------
    L_0 : numpy.ndarray
        Initial mean queue length vector for each queue in the network.
    M : int
        Total number of customers in the system.
    mu : numpy.ndarray
        Service rate vector for each queue in the network.
    P : numpy.ndarray
        Square customer routing matrix where ``P[i, j]`` represents the
        probability that a customer finishing service at node `i` moves to
        node `j` for service. Each row must sum to one.

    Returns:
    --------
    L : numpy.ndarray
        ``(M + 1, K)`` array where ``L[m]`` is the mean queue length at each
        queue with `m` customers in the system.
    W : numpy.ndarray
        ``(M, K)`` array where ``W[m - 1]`` is the mean waiting (residence)
        time at each queue with `m` customers in the system.
    lambdas : numpy.ndarray
        Length ``M`` array of system throughputs for ``m = 1 .. M``.
    v : numpy.ndarray
        Visit ratio vector representing the relative visit rates of each node.

    Raises:
    -------
    ValueError
        If `mu` is not positive, `P` is not a square stochastic matrix,
        `L_0` does not have one entry per queue, or `M` is not a positive
        integer.

    Notes:
    ------
    This function assumes a closed queueing network with M/M/1 queues and uses
    the standard iterative mean value analysis algorithm. It computes mean
    queue lengths, residence times, and system throughput. With a closed
    network the number of customers is conserved, so ``L[m].sum() == m``.

    Example:
    --------
    >>> L_0 = np.zeros(1)
    >>> mu = np.array([2.0])
    >>> P = np.array([[1.0]])
    >>> L, W, lambdas, v = mean_value_analysis(L_0, 3, mu, P)
    >>> L[:, 0]
    array([0., 1., 2., 3.])
    >>> lambdas
    array([2., 2., 2.])
    """
    L_0 = np.asarray(L_0, dtype=float)
    mu = np.asarray(mu, dtype=float)
    P = np.asarray(P, dtype=float)

    if mu.ndim != 1:
        raise ValueError("mu must be a 1-D array of service rates.")

    K = mu.size

    if not np.all(mu > 0):
        raise ValueError("All service rates must be positive.")

    if L_0.ndim != 1 or L_0.size != K:
        raise ValueError("L_0 must have one entry per queue.")

    if P.shape != (K, K):
        raise ValueError("P must be a K x K routing matrix.")

    if not isinstance(M, (int, np.integer)) or M < 1:
        raise ValueError("M must be a positive integer.")

    v = visit_ratio_vector(P)

    L = np.empty((M + 1, K))
    L[0] = L_0
    W = np.empty((M, K))
    lambdas = np.empty(M)

    for m in range(1, M + 1):
        W[m - 1] = (1.0 + L[m - 1]) / mu
        lambdas[m - 1] = m / (W[m - 1] @ v)
        L[m] = v * lambdas[m - 1] * W[m - 1]

    return L, W, lambdas, v
