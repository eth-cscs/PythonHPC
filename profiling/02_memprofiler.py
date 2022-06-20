import numpy as np
from memory_profiler import profile


@profile
def euclidean_trick(x, y):
    """Euclidean square distance matrix.

    Inputs:
    x: (N, m) numpy array
    y: (N, m) numpy array

    Ouput:
    (N, N) Euclidean square distance matrix:
    r_ij = (x_ij - y_ij)^2
    """
    x2 = np.einsum('ij,ij->i', x, x)[:, np.newaxis]
    y2 = np.einsum('ij,ij->i', y, y)[np.newaxis, :]

    xy = x @ y.T

    return np.abs(x2 + y2 - 2. * xy)


if __name__ == "__main__":
    nsamples = 2000
    nfeat = 50
    rng = np.random.default_rng()
    x = 10. * rng.random((nsamples, nfeat))

    euclidean_trick(x, x)
