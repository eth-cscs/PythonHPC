import time
import numpy as np
from scipy.spatial.distance import cdist
from metrics.cbdm import cityblock_distance_matrix


nsamples, nfeat = (12000, 50)
x = 10. * np.random.random([nsamples, nfeat])
cbdm_f90 = np.empty([nsamples, nsamples], order='F')

start = time.time()
cityblock_distance_matrix(x.T, x.T, nsamples, nfeat, cbdm_f90)
print("f90  : %.2f seconds" % (time.time() - start))

start = time.time()
cbdm_sp = cdist(x, x, 'cityblock')
print("cdist: %.2f seconds" % (time.time() - start))

print('\ndiff %.2e' % np.abs(cbdm_f90 - cbdm_sp).max())
