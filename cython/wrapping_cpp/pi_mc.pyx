# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# distutils: extra_link_args = -std=c++11

from cython cimport boundscheck, wraparound
from mt_random cimport mt19937, uniform_real_distribution

@boundscheck(False)
@wraparound(False)
cpdef double pi_mc(long long n=1000):
    '''Calculate PI using Monte Carlo method'''
    cdef long long in_circle = 0
    cdef long long i
    cdef double x, y
    cdef mt19937 gen = mt19937(5)
    cdef uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0,1.0)
    for i in range(n):
        x, y = dist(gen), dist(gen)
        if x * x + y * y <= 1.0:
            in_circle += 1

    return 4.0 * in_circle / n
