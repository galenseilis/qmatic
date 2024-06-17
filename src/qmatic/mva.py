import numpy as np
from numpy.typing import NDArray

def matrix_minor(arr:NDArray, i:int, j:int) -> NDArray:
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
    array([[1, 3],
           [7, 9]])
    """
    return np.delete(np.delete(arr,i,axis=0), j, axis=1)

def visit_ratio_vector(P):
    """
    Compute the visit ratio vector `v` for a closed queueing network based on
    the customer routing matrix `P` using Cramer's rule.

    Parameters:
    -----------
    P : numpy.ndarray
        Customer routing matrix where `P[i, j]` represents the probability
        that a customer finishing service at node `i` moves to node `j` for service.

    Returns:
    --------
    v : numpy.ndarray
        Visit ratio vector representing the probabilities of visiting each node.
    """
    
    dim = P.shape[0]
    v = np.empty(dim)

    # Use Cramer's rule to solve v=vP
    for i in range(dim):
        v[i] = np.linalg.det(
                matrix_minor(P.T, i, i)
            ) / np.linalg.det(P.T)

    return v

def mean_value_analysis(L_0:NDArray, M:int, mu:NDArray, P:NDArray):
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
        Customer routing matrix where P[i, j] represents the probability
        that a customer finishing service at node i moves to node j for service.

    Returns:
    --------
    L : numpy.ndarray
        Matrix representing the mean queue lengths for each queue over the iterations.
    W : numpy.ndarray
        Matrix representing the mean waiting times for each queue over the iterations.
    lambdas : numpy.ndarray
        Array representing the system throughput at each iteration.
    v : numpy.ndarray
        Visit ratio vector representing the probabilities of visiting each node.

    Raises:
    -------
    ValueError
        If rows of P do not sum to less-than-or-equal-to one, or if P has fewer than 2 dimensions.

    Notes:
    ------
    This function assumes a closed queueing network with M/M/1 queues and uses an iterative
    algorithm to compute mean queue lengths, waiting times, and system throughput.

    Example:
    --------
    >>> L_0 = np.zeros(3)
    >>> M = 100
    >>> mu = np.array([0.2, 0.4, 0.3])
    >>> P = np.array([[0.2, 0.5, 0.3], [0.4, 0.1, 0.5], [0.3, 0.2, 0.5]])
    >>> mean_value_analysis(L_0, M, mu, P)
    """

    if not np.all(P.sum(axis=1) <= 1):
        raise ValueError('Rows of P must sum to less-than-or-equal-to one.')

    if len(P.shape) == 2:
        v = visit_ratio_vector(P)
    elif len(P.shape) > 2:
        multi_p_flag = True
        v = np.array([visit_ratio_vector(p) for p in B])
    else:
        raise ValueError('P must have 2 or more dimensions.')

    L = np.empty((M + 1, L_0.size))
    L[0,:] = L_0
    W = np.empty((M, L_0.size))
    lambdas = np.empty(M)

    for m in range(1,M):
        W[m-1, :] = (1 + L[m-1,:]) * mu[m,:]
        lambdas[m] = (m+1) / (W[m-1,:] @ v)
        L[m,:] = v * lambdas[m] * W[m-1, :]

    return L, W, lambdas, v
    
