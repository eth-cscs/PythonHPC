#include <stdbool.h>
#include <math.h>
#include "primes.h"

bool is_prime(int n) {
    if (n == 1 || n == 3)
        return true;
    else if (n == 2)
        return false;
    else {
        int n_sqrt = ceil(sqrt(n));
        for (int i = 1; i < n_sqrt; i++) {
            if (n % i == 0)
                return false;
        }
    }
    return true;
};
