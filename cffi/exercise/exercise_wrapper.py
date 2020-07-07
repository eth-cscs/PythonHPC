import numpy as np
from cffi import FFI
from scipy.spatial.distance import cdist
from _cityblock.lib import cbdm


def ccityblock(x, y):
    """Python wrapper for the cbdm C function

    Input:
    x: (N, m) numpy array
    y: (N, m) numpy array
    
    Ouput:
    (N, N) Cityblock distance matrix:
    r_ij = abs(x_ij - y_ij)
    """


nsamples = 2000
nfeat = 50
x = np.random.random([nsamples, nfeat])

print((ccityblock(x, x) - cdist(x, x, 'cityblock')).max())
