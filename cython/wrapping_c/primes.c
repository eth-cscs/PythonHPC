#include <stdbool.h>
#include <math.h>
#include "primes.h"

bool is_prime(int n) {
    if (n <= 1)
        return false;

    int n_sqrt = (int)(sqrt(n));

    for (int i = 2; i <= n_sqrt; i++) {
        if (n % i == 0)
            return false;
    }

    return true;
};
