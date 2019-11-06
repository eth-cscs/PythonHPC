import numpy as np
from cffi import FFI
from scipy.spatial.distance import cdist
from _cityblock.lib import cbdm


nsamples = 2000
nfeat = 50

x = np.random.random([nsamples, nfeat])
r = np.empty((nsamples, nsamples))

ffi = FFI()

x_c = ffi.cast('double *', x.ctypes.data)
r_c = ffi.cast('double *', r.ctypes.data)

cbdm(x_c, x_c, r_c, nsamples, nfeat)

print((r - cdist(x, x, 'cityblock')).max())
