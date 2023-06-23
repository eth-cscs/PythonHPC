# distutils: language = c++
# distutils: extra_compile_args = -std=c++11 -fopenmp
# distutils: extra_link_args = -std=c++11 -fopenmp

from cython cimport boundscheck, wraparound, nogil
from mt_random cimport mt19937, uniform_real_distribution

from cython.parallel cimport parallel, prange
from openmp cimport omp_get_thread_num
from libc.stdio cimport printf

@boundscheck(False)
@wraparound(False)
cpdef double pi_mc(long long n=1000):
    '''Calculate PI using Monte Carlo method'''
    cdef long long in_circle = 0
    cdef long long i
    cdef double x, y
    cdef uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0,1.0)
    cdef mt19937 gen

    with nogil, parallel():
        gen = mt19937(omp_get_thread_num())
        for i in prange(n):
            x, y = dist(gen), dist(gen)
            if x * x + y * y <= 1.0:
                in_circle += 1

    return 4.0 * in_circle / n
