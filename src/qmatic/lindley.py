from numpy.typing import NDArray

def lindley(L_0:float, A:NDArray, D:NDArray) -> NDArray:
    """
    Compute the Lindley process given initial value, arrivals, and departures.

    Parameters
    ----------
    L_0 : float
        The initial value of the Lindley process.
    A : numpy.ndarray
        1-D array of arrival values.
    D : numpy.ndarray
        1-D array of departure values.
    
    Returns
    -------
    numpy.ndarray
        1-D array representing the Lindley process values over time.

    Raises
    ------
    AssertionError
        If the sizes of the 'A' and 'D' arrays do not match.

    Notes
    -----
    The Lindley process is a stochastic process used to model queue lengths in
    queuing systems. The process is defined by the recursive relation:
    
    L[t+1] = max(0, L[t] + A[t] - D[t])

    where:
    - L[t] is the Lindley process value at time t.
    - A[t] is the arrival value at time t.
    - D[t] is the departure value at time t.

    Examples
    --------
    >>> L_0 = 0.0
    >>> A = np.array([1, 2, 1])
    >>> D = np.array([0, 2, 1])
    >>> lindley(L_0, A, D)
    array([1., 2., 1., 0.])
    """
    if A.size != D.size:
        raise ValueError('A.size != D.size')

    L = np.empty(A.size + 1)
    L[0] = L_0
    
    for t, (a_t, d_t) in enumerate(zip(A,D)):
        L[t+1] = np.maximum(0, L[t] + a_t - d_t)

    return L
