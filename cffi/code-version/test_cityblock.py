import time
import numpy as np
from cffi import FFI
from scipy.spatial.distance import cdist
from _cityblock.lib import cbdm


nsamples = 12000
nfeat = 50

x = np.random.random([nsamples, nfeat])
r = np.empty((nsamples, nsamples))

ffi = FFI()

x_c = ffi.cast('double *', x.ctypes.data)
r_c = ffi.cast('double *', r.ctypes.data)

start = time.time()
cbdm(x_c, x_c, r_c, nsamples, nfeat)
print(f'C     : {time.time() - start:.2f} seconds')

start = time.time()
r_cdist = cdist(x, x, 'cityblock')
print(f'cdist : {time.time() - start:.2f} seconds')

# check the result by comparing to cdist
print(f'\ndiff  : {(r - r_cdist).max():.2e}')
