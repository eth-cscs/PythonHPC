#
# Linear algebra subroutines
#
# Originally written in C++ by Ben Cumming, CSCS
# Ported to Python by Vasileios Karakasis, CSCS

import math
import numpy as np
import sys

from operators import diffusion

# epsilon value use for matrix-vector approximation
EPS = 1.0e-8
EPS_INV = 1.0 / EPS


def cg(x, b, boundary, options, solution, tolerance, maxiters):

    # Initialize temporary storage
    Fx = np.zeros_like(x)
    Fxold = np.zeros_like(x)
    xold = np.copy(x)

    # matrix vector multiplication is approximated with
    # A*v = 1/epsilon * ( F(x+epsilon*v) - F(x) )
    #     = 1/epsilon * ( F(x+epsilon*v) - Fxold )
    # we compute Fxold at startup
    # we have to keep x so that we can compute the F(x+exps*v)
    diffusion(x, Fxold, boundary, options, solution)

    v = (1. + EPS) * x

    # Fx = F(v)
    diffusion(v, Fx, boundary, options, solution)

    # r = b - A*x
    # where A*x = (Fx-Fxold)/eps
    r = b - EPS_INV * (Fx - Fxold)

    # p = r
    p = np.copy(r)

    # rold = <r,r>
    rold = np.sum(r ** 2)
    rnew = rold

    if math.sqrt(rold) < tolerance:
        return (True, 0)

    for it in range(maxiters):
        # Ap = A*p
        v = xold + EPS * p
        diffusion(v, Fx, boundary, options, solution)

        Ap = EPS_INV * (Fx - Fxold)

        # alpha = rold / p'*Ap
        alpha = rold / (p @ Ap)

        x += alpha * p

        r -= alpha * Ap

        # find new norm
        rnew = np.sum(r ** 2)

        if (math.sqrt(rnew) < tolerance):
            return (True, it)

        p = r + (rnew / rold) * p
        rold = rnew

    print(f'ERROR: CG failed to converge after {it} iterations, '
          f'with residual {math.sqrt(rnew)}', file=sys.stderr)
    return (False, it)
