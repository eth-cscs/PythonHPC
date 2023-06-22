from libc.math cimport sqrt, floor

from cython cimport boundscheck, wraparound, cdivision


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef bint is_prime(int n):
    cdef int i

    if n <= 1:
        return False

    for i in range(2, <int>floor(sqrt(n))+ 1):
        if n % i == 0:
            return False

    return True
