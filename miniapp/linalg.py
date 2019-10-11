#
# Linear algebra subroutines
#
# Originally written in C++ by Ben Cumming, CSCS
# Ported to Python by Vasileios Karakasis, CSCS

import math
import numpy as np
import sys

import operators


def ss_fill(x, val):
    x.fill(val)


def ss_copy(dst, src):
    np.copyto(dst, src)


def ss_norm2(x):
    result = 0.
    for xi in x:
        result += xi*xi

    return math.sqrt(result)


def ss_scale(y, alpha, x):
    for i, _ in enumerate(x):
        y[i] = alpha*x[i]


def ss_add_scaled_diff(y, x, alpha, l, r):
    # computes y = x + alpha*(l-r)
    # y, x, l and r are vectors
    # alpha is a scalar
    for i, _ in enumerate(x):
        y[i] = x[i] + alpha*(l[i] - r[i])


def ss_scaled_diff(y, alpha, l, r):
    for i, _ in enumerate(l):
        y[i] = alpha * (l[i] - r[i])


def ss_dot(x, y):
    return x @ y


def ss_lcomb(y, alpha, x, beta, z):
    for i, _ in enumerate(x):
        y[i] = alpha*x[i] + beta*z[i]


def ss_axpy(y, alpha, x):
    for i, _ in enumerate(x):
        y[i] += alpha*x[i]


Ap = None
r  = None
p  = None
v  = None
Fx = None
xold  = None
Fxold = None


def ss_cg(x, b, maxiters, tol, boundary, options, solution):
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
    ss_fill(Fx, 0.0)
    ss_fill(Fxold, 0.0)
    ss_copy(xold, x)

    # matrix vector multiplication is approximated with
    # A*v = 1/epsilon * ( F(x+epsilon*v) - F(x) )
    #     = 1/epsilon * ( F(x+epsilon*v) - Fxold )
    # we compute Fxold at startup
    # we have to keep x so that we can compute the F(x+exps*v)
    operators.diffusion(x, Fxold, boundary, options, solution)

    # v = x + epsilon*x
    ss_scale(v, 1. + eps, x)

    # Fx = F(v)
    operators.diffusion(v, Fx, boundary, options, solution)

    # r = b - A*x
    # where A*x = (Fx-Fxold)/eps
    ss_add_scaled_diff(r, b, -eps_inv, Fx, Fxold)

    # p = r
    ss_copy(p, r)

    # rold = <r,r>
    rold = ss_dot(r, r)
    rnew = rold

    if math.sqrt(rold) < tol:
        return (True, 0)

    for iter in range(maxiters):
        # Ap = A*p
        ss_lcomb(v, 1.0, xold, eps, p)
        operators.diffusion(v, Fx, boundary, options, solution)
        ss_scaled_diff(Ap, eps_inv, Fx, Fxold)

        # alpha = rold / p'*Ap
        dot = ss_dot(p, Ap)
        alpha = rold / dot

        # x += alpha*p
        ss_axpy(x, alpha, p)

        # r -= alpha*Ap
        ss_axpy(r, -alpha, Ap)

        # find new norm
        rnew = ss_dot(r, r)

        if (math.sqrt(rnew) < tol):
            return (True, iter)

        # p = r + (rnew/rold) * p
        ss_lcomb(p, 1.0, r, rnew / rold, p)
        rold = rnew

    print(f'ERROR: CG failed to converge after {iter} iterations, '
          f'with residual {math.sqrt(rnew)}', file=sys.stderr)
    return (False, iter)
