import time
import numpy as np
from scipy.spatial.distance import cdist
from metrics.cbdm import cityblock_distance_matrix


nsamples, nfeats = (12000, 50)
rng = np.random.default_rng()
x = 10. * rng.random([nsamples, nfeats])
cbdm_f90 = np.empty([nsamples, nsamples], order='F')

start = time.time()
cityblock_distance_matrix(x.T, x.T, nsamples, nfeats, cbdm_f90)
print("f90  : %.2f seconds" % (time.time() - start))

start = time.time()
cbdm_sp = cdist(x, x, 'cityblock')
print("cdist: %.2f seconds" % (time.time() - start))

print('\ndiff %.2e' % np.abs(cbdm_f90 - cbdm_sp).max())
