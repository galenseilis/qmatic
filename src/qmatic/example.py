import numpy as np
from lindley_cython import lindley_cython

# Example usage
L_0 = 0.0
A = np.array([1.0, 2.0, 3.0])
D = np.array([0.5, 1.0, 1.5])

result = lindley_cython(L_0, A, D)
print(result)

