from math import sqrt, ceil

cpdef bint is_prime(int n):
    cdef int i, n_sqrt = ceil(sqrt(n))
    if n == 1 or n == 3:
        return True
    elif n == 2:
        return False
    else :
        for i in range(2, n_sqrt):
            if n % i == 0:
                return False
    return True
