from numpy.typing import NDArray

def lindley(L_0:float, A:NDArray, D:NDArray):
    assert A.size == D.size
    L = np.empty(A.size + 1)
    L[0] = L_0
    
    for t, (a_t, d_t) in enumerate(zip(A,D)):
        L[t+1] = np.maximum(0, L[t] + a_t - d_t)

    return L

import matplotlib.pyplot as plt
import numpy as np

m = 100
A = np.random.poisson(1,size=m)
D = np.random.poisson(1.1,size=m)
L_0 = 100

##plt.plot(lindley(L_0, A, D))
##plt.show()

result_1 = lindley(L_0, A, D)

L = np.zeros(A.size + 1)
L[0] = L_0
result_2 = np.maximum(0, L[:-1] + A - D, out=L[1:])
