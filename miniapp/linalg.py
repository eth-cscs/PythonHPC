#
# Linear algebra subroutines
#
# Originally written in C++ by Ben Cumming, CSCS
# Ported to Python by Vasileios Karakasis, CSCS

import collections
import math
import numpy as np
import sys

import operators

# epsilon value use for matrix-vector approximation
EPS = 1.0e-8
EPS_INV = 1.0 / EPS

CGStatus = collections.namedtuple('CGStatus',
                                  ['converged', 'iters', 'residual'])


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
        return CGStatus(True, 0, math.sqrt(rnew))

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

        residual = math.sqrt(rnew)
        if (residual < tolerance):
            return CGStatus(True, it, residual)

        p = r + (rnew / rold) * p
        rold = rnew

    return CGStatus(False, it, residual)
