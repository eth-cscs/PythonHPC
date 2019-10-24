#
# Operators based on min-app code written by Oliver Fuhrer, MeteoSwiss
#
# Modified by Ben Cumming, CSCS
# Ported to Python by Vasileios Karakasis, CSCS

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def diffusion(U, S, x_old, boundary, options):
    dxs = 1000. * options.dx * options.dx
    alpha = options.alpha
    nx = options.nx
    ny = options.ny
    bndN = boundary.north
    bndS = boundary.south
    bndE = boundary.east
    bndW = boundary.west
    iend  = nx - 1
    jend  = ny - 1

    x_old = x_old.reshape((nx, ny))
    U = U.reshape((nx, ny))
    S = S.reshape((nx, ny))

    # the interior grid points
    for i in range(1, iend):
        for j in range(1, jend):
            S[i, j] = (-(4. + alpha)*U[i, j] + U[i-1, j] + U[i+1, j] + U[i, j-1] +
                       U[i, j+1] + alpha*x_old[i, j] + dxs*U[i, j]*(1 - U[i, j]))

    # the east boundary
    i = iend
    for j in range(1, jend):
        S[i, j] = (-(4. + alpha) * U[i, j] + U[i-1, j] + U[i, j-1] + U[i, j+1] +
                   alpha*x_old[i, j] + bndE[j] + dxs * U[i, j] * (1.0 - U[i, j]))

    # the west boundary
    i = 0
    for j in range(1, jend):
        S[i, j] = (-(4. + alpha) * U[i, j] + U[i-1, j] + U[i, j-1] + U[i, j+1] +
                   alpha*x_old[i, j] + bndW[j] + dxs * U[i, j] * (1.0 - U[i, j]))

    # the NW corner
    j, i = jend, 0
    S[i, j] = (-(4. + alpha) * U[i, j] + U[i+1, j] + U[i, j-1] + alpha *
               x_old[i, j] + bndW[j] + bndN[i] + dxs * U[i, j] * (1.0 - U[i, j]))

    # the north boundary
    for i in range(1, iend):
        S[i, j] = (-(4. + alpha) * U[i, j] + U[i-1, j] + U[i+1, j] + U[i, j-1] +
                   alpha*x_old[i, j] + bndN[i] + dxs * U[i, j] * (1.0 - U[i, j]))

    # the NE corner
    i = iend
    S[i, j] = (-(4. + alpha) * U[i, j] + U[i-1, j] + U[i, j-1] + alpha *
               x_old[i, j] + bndE[j] + bndN[i] + dxs * U[i, j] * (1.0 - U[i, j]))

    # the SW corner
    j, i = 0, 0
    S[i, j] = (-(4. + alpha) * U[i, j] + U[i+1, j] + U[i, j+1] + alpha *
               x_old[i, j] + bndW[j] + bndS[i] + dxs * U[i, j] * (1.0 - U[i, j]))

    # the south boundary
    for i in range(1, iend):
        S[i, j] = (-(4. + alpha) * U[i, j] + U[i-1, j] + U[i+1, j] + U[i, j+1] +
                   alpha * x_old[i, j] + bndS[i] + dxs * U[i, j] * (1.0 - U[i, j]))

    # the SE corner
    i = iend
    S[i, j] = (-(4. + alpha) * U[i, j] + U[i-1, j] + U[i, j+1] + alpha *
               x_old[i, j] + bndE[j] + bndS[i] + dxs * U[i, j] * (1.0 - U[i, j]))
