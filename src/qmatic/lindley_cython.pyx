# lindley_cython.pyx
import numpy as np
cimport numpy as np

def lindley_cython(double L_0, np.ndarray[np.float64_t, ndim=1] A, np.ndarray[np.float64_t, ndim=1] D):
    cdef int size = A.shape[0]
    assert size == D.shape[0]
    
    cdef np.ndarray[np.float64_t, ndim=1] L = np.empty(size + 1, dtype=np.float64)
    L[0] = L_0
    
    cdef int t
    cdef double a_t, d_t
    for t in range(size):
        a_t = A[t]
        d_t = D[t]
        L[t+1] = max(0, L[t] + a_t - d_t)

    return L

