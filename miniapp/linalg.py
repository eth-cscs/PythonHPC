#
# Linear algebra subroutines
#
# Originally written in C++ by Ben Cumming, CSCS
# Ported to Python by Vasileios Karakasis, CSCS

import math
import numpy as np
import sys

import operators


Ap = None
r  = None
p  = None
v  = None
Fx = None
xold  = None
Fxold = None


def cg(x, b, maxiters, tol, boundary, options, solution):
    global Ap, r, p, v, Fx, Fxold, xold

    nx = options.nx
    ny = options.ny

    Ap = np.zeros(nx*ny) if Ap is None else Ap
    r  = np.zeros(nx*ny) if r  is None else r
    p  = np.zeros(nx*ny) if p  is None else p
    v  = np.zeros(nx*ny) if v  is None else v
    Fx = np.zeros(nx*ny) if Fx is None else Fx
    Fxold = np.zeros(nx*ny) if Fxold is None else Fxold
    xold  = np.zeros(nx*ny) if xold is None else xold

    # epsilon value use for matrix-vector approximation
    eps = 1e-8
    eps_inv = 1 / eps

    # initialize memory for temporary storage
    Fx.fill(0.0)
    Fxold.fill(0.0)
    np.copyto(xold, x)

    # matrix vector multiplication is approximated with
    # A*v = 1/epsilon * ( F(x+epsilon*v) - F(x) )
    #     = 1/epsilon * ( F(x+epsilon*v) - Fxold )
    # we compute Fxold at startup
    # we have to keep x so that we can compute the F(x+exps*v)
    operators.diffusion(x, Fxold, boundary, options, solution)

    v = (1. + eps)*x

    # Fx = F(v)
    operators.diffusion(v, Fx, boundary, options, solution)

    # r = b - A*x
    # where A*x = (Fx-Fxold)/eps
    r = b - eps_inv*(Fx-Fxold)

    # p = r
    np.copyto(p, r)

    # rold = <r,r>
    rold = r @ r
    rnew = rold

    if math.sqrt(rold) < tol:
        return (True, 0)

    for iter in range(maxiters):
        # Ap = A*p
        v = xold + eps*p
        operators.diffusion(v, Fx, boundary, options, solution)

        Ap = eps_inv*(Fx - Fxold)

        # alpha = rold / p'*Ap
        dot = p @ Ap
        alpha = rold / dot

        x += alpha*p

        r -= alpha*Ap

        # find new norm
        rnew = r @ r

        if (math.sqrt(rnew) < tol):
            return (True, iter)

        p = r + (rnew/rold) * p
        rold = rnew

    print(f'ERROR: CG failed to converge after {iter} iterations, '
          f'with residual {math.sqrt(rnew)}', file=sys.stderr)
    return (False, iter)
