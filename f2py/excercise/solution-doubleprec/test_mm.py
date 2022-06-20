import numpy as np
import mm

n = 1000
x = 10. * np.random.random((n, n))

xx = mm.matrix_multiplication(x.T, x.T)

diff = np.abs(xx - x @ x).max()
print(diff)
