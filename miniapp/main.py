#
# CSCS-USI SummerSchool miniapp ported to Python
#
#   A small benchmark app that solves the 2D fisher equation using second-order
#   finite differences.
#
#   Originally developed in C++ by Ben Cumming, CSCS
#   Ported to Python by Vasileios Karakasis, CSCS

import collections
import math
import matplotlib
import numpy as np
import os
import pylab
import sys
from datetime import datetime

import linalg
import operators


Discretization = collections.namedtuple(
    'Discretization', ['nx', 'ny', 'N', 'nt', 'dt', 'dx', 'alpha']
)

Boundary = collections.namedtuple(
    'Boundary', ['north', 'south', 'east', 'west']
)

Solution = collections.namedtuple('Solution', ['old', 'new'])


def usage():
    print(f'Usage: {sys.argv[0]} nx ny nt t', file=sys.stderr)
    print(f'  nx  number of gridpoints in x-direction', file=sys.stderr)
    print(f'  ny  number of gridpoints in y-direction', file=sys.stderr)
    print(f'  nt  number of timesteps', file=sys.stderr)
    print(f'  t   total simulated time', file=sys.stderr)


def parse_arg(v, argname, fn):
    try:
        v = fn(v)
    except ValueError as e:
        print(f'{sys.argv[0]}: could not parse argument {argname}: {e}')
        usage()
        sys.exit(1)

    return v


def main():
    if len(sys.argv) < 5:
        print(f'{sys.argv[0]}: too few arguments', file=sys.stderr)
        usage()
        sys.exit(1)

    nx, ny, nt, t, *_ = sys.argv[1:]

    def is_positive(x):
        if x <= 0:
            raise ValueError(f'value must be positive: {x}')

        return x

    nx = parse_arg(nx, 'nx', lambda x: is_positive(int(x)))
    ny = parse_arg(ny, 'ny', lambda x: is_positive(int(x)))
    nt = parse_arg(nt, 'nt', lambda x: is_positive(int(x)))
    t  = parse_arg(t,  't',  lambda x: is_positive(float(x)))

    # calculate timestep
    dt = t / nt

    # compute the distance between grid points
    # assume that x dimension has length 1.0
    dx = 1. / (nx - 1)

    # set alpha, assume diffusion coefficient D is 1
    alpha = (dx*dx) / dt

    options = Discretization(nx, ny, nx*ny, nt, dt, dx, alpha)

    # set iteration parameters
    max_cg_iters = 200
    max_newton_iters = 50
    tolerance = 1.e-6

    print(f'========================================================================')
    print(f'                      Welcome to mini-stencil!')
    print(f'version   :: Python')
    print(f'mesh      :: {nx} * {ny} dx = {dx}')
    print(f'time      :: {nt} time steps from 0 .. {nt*dt}')
    print(f'iteration :: CG {max_cg_iters}, Newton {max_newton_iters}, '
          f'tolerance {tolerance}')
    print(f'========================================================================')

    # initialize fields
    x_new = np.zeros(nx*ny)
    x_old = np.zeros(nx*ny)
    solution = Solution(x_old, x_new)

    b = np.zeros(nx*ny)
    deltax = np.zeros(nx*ny)

    # set dirichlet boundary conditions to 0 all around
    bndN  = np.zeros(nx)
    bndS  = np.zeros(nx)
    bndE  = np.zeros(ny)
    bndW  = np.zeros(ny)
    boundary = Boundary(bndN, bndS, bndE, bndW)

    # set the initial condition
    # a circle of concentration 0.1 centred at (xdim/4, ydim/4) with radius
    # no larger than 1/8 of both xdim and ydim
    xc = 1.0 / 4.0
    yc = (ny - 1) * dx / 4
    radius = min(xc, yc) / 2.0
    x_new = x_new.reshape(nx, ny)
    for j in range(ny):
        y = (j - 1) * dx
        for i in range(nx):
            x = (i - 1) * dx
            if (x-xc) * (x-xc) + (y-yc) * (y-yc) < radius*radius:
                x_new[j, i] = 0.1

    # reshape x_new back
    x_new = x_new.reshape(nx*ny)

    flops_bc = 0
    flops_diff = 0
    flops_blas1 = 0
    iters_cg = 0
    iters_newton = 0
    timespent = datetime.now()

    # main timeloop
    for timestep in range(1, nt+1):
        np.copyto(x_old, x_new)
        converged = False
        for it in range(max_newton_iters):
            operators.diffusion(x_new, b, boundary, options, solution)
            residual = math.sqrt(b @ b)
            if residual < tolerance:
                converged = True
                break

            cg_convered, iters = linalg.cg(
                deltax, b, max_cg_iters, tolerance, boundary, options, solution
            )
            iters_cg += iters
            if not cg_convered:
                break

            x_new -= deltax

        iters_newton += it + 1
        if not converged:
            print(f'step {timestep} '
                  f'ERROR : nonlinear iterations failed to converge')
            break

    # get times
    timespent = (datetime.now() - timespent).total_seconds()
    print('----------------------------------------'
          '----------------------------------------')
    print(f'simulation took {timespent} seconds')
    print(f'{iters_cg} conjugate gradient iterations, at rate of '
          f'{iters_cg/timespent} iters/second')
    print(f'{iters_newton} newton iterations')
    print(f'Goodbye!')

    # plot final output
    if 'DISPLAY' not in os.environ:
        matplotlib.use('Agg')

    xspace = np.linspace(0, 1, nx)
    yspace = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xspace, yspace)

    # number of contours we wish to see
    V = [-0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    pylab.contourf(X, Y, x_new.reshape((nx, ny)), V, alpha=.75, cmap='jet')
    pylab.axes().set_aspect('equal')

    outfile = f'output_{nx}x{ny}_t={t}_steps={nt}.png'
    print(f'saving solution in "{outfile}" ...')
    pylab.savefig(outfile, dpi=72)
    if 'DISPLAY'  in os.environ:
        pylab.show()


if __name__ == '__main__':
    main()
