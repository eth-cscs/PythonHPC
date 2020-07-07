import time
import numpy as np
from metrics.edm import euclidean_distance_matrix


def euclidean_numpy(x, y):
    """Euclidean square distance matrix.

    Inputs:
    x: (N,) numpy array
    y: (N,) numpy array

    Ouput:
    (N, N) Euclidean square distance matrix:
    r_ij = (x_ij - y_ij)^2
    """

    x2 = np.einsum('ij,ij->i', x, x)[:, np.newaxis]
    y2 = np.einsum('ij,ij->i', y, y)[np.newaxis, :]

    xy = np.dot(x, y.T)

    return np.abs(x2 + y2 - 2. * xy)


nsamples, nfeat = (12000, 50)
x = 10. * np.random.random([nsamples, nfeat])
edm_f90 = np.empty([nsamples, nsamples], order='F')

start = time.time()
euclidean_distance_matrix(x.T, x.T, nsamples, nfeat, edm_f90)
print("f90  : %.2f seconds" % (time.time() - start))

start = time.time()
edm_np = euclidean_numpy(x, x)
print("numpy: %.2f seconds" % (time.time() - start))

print('\ndiff : %.2e'% np.abs(edm_f90 - edm_np).max())
