#
# Linear algebra subroutines
#
# Originally written in C++ by Ben Cumming, CSCS
# Ported to Python by Vasileios Karakasis, CSCS

import math
import numpy as np
import sys

import operators


class ConjugateGradient:
    '''A class implementing a matrix free version of the conjugate gradient
       method for solving linear systems of equations.
    '''

    # epsilon value use for matrix-vector approximation
    EPS = 1.0e-8
    EPS_INV = 1.0 / EPS

    def __init__(self, nx, ny, tolerance, maxiters):
        self.nx = nx
        self.ny = ny
        self.tolerance = tolerance
        self.maxiters = maxiters
        self.Ap = np.zeros(nx * ny)
        self.r = np.zeros(nx * ny)
        self.p = np.zeros(nx * ny)
        self.v = np.zeros(nx * ny)
        self.Fx = np.zeros(nx*ny)
        self.Fxold = np.zeros(nx*ny)
        self.xold  = np.zeros(nx*ny)

    def solve(self, x, b, boundary, options, solution, op):

        # initialize memory for temporary storage
        self.Fx[:] = 0.0
        self.Fxold[:] = 0.0
        self.xold[:] = x

        # matrix vector multiplication is approximated with
        # A*v = 1/epsilon * ( F(x+epsilon*v) - F(x) )
        #     = 1/epsilon * ( F(x+epsilon*v) - Fxold )
        # we compute Fxold at startup
        # we have to keep x so that we can compute the F(x+exps*v)
        op(x, self.Fxold, boundary, options, solution)

        self.v[:] = (1. + self.EPS) * x

        # Fx = F(v)
        op(self.v, self.Fx, boundary, options, solution)

        # r = b - A*x
        # where A*x = (Fx-Fxold)/eps
        self.r = b - self.EPS_INV * (self.Fx - self.Fxold)

        # p = r
        self.p[:] = self.r

        # rold = <r,r>
        rold = self.r @ self.r
        rnew = rold

        if math.sqrt(rold) < self.tolerance:
            return (True, 0)

        for it in range(self.maxiters):
            # Ap = A*p
            self.v[:] = self.xold + self.EPS * self.p
            op(self.v, self.Fx, boundary, options, solution)

            self.Ap[:] = self.EPS_INV * (self.Fx - self.Fxold)

            # alpha = rold / p'*Ap
            dot = self.p @ self.Ap
            alpha = rold / dot

            x += alpha * self.p

            self.r -= alpha * self.Ap

            # find new norm
            rnew = np.sum(self.r ** 2)

            if (math.sqrt(rnew) < self.tolerance):
                return (True, it)

            self.p[:] = self.r + (rnew / rold) * self.p
            rold = rnew

        print(f'ERROR: CG failed to converge after {it} iterations, '
              f'with residual {math.sqrt(rnew)}', file=sys.stderr)
        return (False, it)
