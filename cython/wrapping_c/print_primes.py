#!/usr/bin/python

import numpy as np
from my_primes import imp_is_prime


if __name__ == '__main__':
    numbers = np.arange(100000, 1000001)
    primes = [imp_is_prime(n) for n in numbers]
    print('The 10 largest primes between 100000 and 1000000 are:')
    for i in numbers[primes][-10:]:
        print(i)
