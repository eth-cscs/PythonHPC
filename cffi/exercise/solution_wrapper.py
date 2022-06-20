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

    r = np.empty((nsamples, nsamples))

    ffi = FFI()

    x_c = ffi.cast('double *', x.ctypes.data)
    y_c = ffi.cast('double *', y.ctypes.data)
    r_c = ffi.cast('double *', r.ctypes.data)

    cbdm(x_c, y_c, r_c, nsamples, nfeat)

    return r


nsamples = 2000
nfeat = 50
rng = np.random.default_rng()
x = rng.random((nsamples, nfeat))

print((ccityblock(x, x) - cdist(x, x, 'cityblock')).max())
