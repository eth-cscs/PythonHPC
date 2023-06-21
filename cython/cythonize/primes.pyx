from libc.math cimport sqrt, ceil

from cython cimport boundscheck, wraparound, cdivision


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef bint is_prime(int n):
    cdef int i

    if n == 1 or n == 3:
        return True
    elif n == 2:
        return False
    else:
        for i in range(1, <int>ceil(sqrt(n))):
            if n % i == 0:
                return False

    return True
