#
# Linear algebra subroutines
#
# Originally written in C++ by Ben Cumming, CSCS
# Ported to Python by Vasileios Karakasis, CSCS

import math
import numba
import numpy as np
import sys

import operators

# epsilon value use for matrix-vector approximation
EPS = 1.0e-8
EPS_INV = 1.0 / EPS


@numba.jit(nopython=True, cache=True)
def cg(x, x_old, b, boundary, options, tolerance, maxiters):

    # Initialize temporary storage
    Fx = np.zeros_like(x)
    Fxold = np.zeros_like(x)
    xold = np.copy(x)

    # matrix vector multiplication is approximated with
    # A*v = 1/epsilon * ( F(x+epsilon*v) - F(x) )
    #     = 1/epsilon * ( F(x+epsilon*v) - Fxold )
    # we compute Fxold at startup
    # we have to keep x so that we can compute the F(x+exps*v)
    operators.diffusion(x, Fxold, x_old, boundary, options)

    v = (1. + EPS) * x

    # Fx = F(v)
    operators.diffusion(v, Fx, x_old, boundary, options)

    # r = b - A*x
    # where A*x = (Fx-Fxold)/eps
    r = b - EPS_INV * (Fx - Fxold)

    # p = r
    p = np.copy(r)

    # rold = <r,r>
    rold = r @ r
    rnew = rold

    if math.sqrt(rold) < tolerance:
        return (True, 0, rnew)

    for it in range(maxiters):
        # Ap = A*p
        v = xold + EPS * p
        operators.diffusion(v, Fx, x_old, boundary, options)

        Ap = EPS_INV * (Fx - Fxold)

        # alpha = rold / p'*Ap
        alpha = rold / (p @ Ap)

        x += alpha * p

        r -= alpha * Ap

        # find new norm
        rnew = r @ r

        if (math.sqrt(rnew) < tolerance):
            return (True, it, rnew)

        p = r + (rnew / rold) * p
        rold = rnew

    return (False, it, rnew)


def _cg(x, x_old, b, boundary, options, tolerance, maxiters):
    converged, iters, rnew = _cg(x, x_old, b, boundary, options,
                                 tolerance, maxiters)
    if not converged:
        print(f'ERROR: CG failed to converge after {iters} iterations, '
              f'with residual {math.sqrt(rnew)}', file=sys.stderr)

    return (converged, iters)
