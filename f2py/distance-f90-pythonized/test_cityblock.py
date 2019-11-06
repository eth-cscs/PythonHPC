import time
import numpy as np
from scipy.spatial.distance import cdist
from metrics.cbdm import cityblock_distance_matrix


nsamples, nfeat = (5000, 50)
x = 10. * np.random.random([nsamples, nfeat])

start = time.time()
# euclidean_distance_matrix(x.T, x.T, dist_matrix)
cbdm_f90 = cityblock_distance_matrix(x.T, x.T)
print("f90:   %.2f seconds" % (time.time() - start))

start = time.time()
cbdm_np = cdist(x, x, 'cityblock')
print("scipy: %.2f seconds" % (time.time() - start))

print(np.abs(cbdm_f90 - cbdm_np).max())
