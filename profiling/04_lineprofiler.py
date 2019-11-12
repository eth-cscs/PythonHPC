import numpy as np
from line_profiler import LineProfiler


def euclidean_trick(x, y):
    """Euclidean square distance matrix.
    
    Inputs:
    x: (N, m) numpy array
    y: (N, m) numpy array
    
    Ouput:
    (N, N) Euclidean square distance matrix:
    r_ij = x_ij^2 - y_ij^2
    """
    x2 = np.einsum('ij,ij->i', x, x)[:, np.newaxis]
    y2 = np.einsum('ij,ij->i', y, y)[np.newaxis, :]

    xy = np.dot(x, y.T)

    return np.abs(x2 + y2 - 2. * xy)


if __name__ == "__main__":
    nsamples = 2000
    nfeat = 50
    x = 10. * np.random.random([nsamples, nfeat])

    lp = LineProfiler()
    lp_wrapper = lp(euclidean_trick)(x,x)
    lp.print_stats()


