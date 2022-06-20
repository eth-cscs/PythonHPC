import math
import numba
import numba.cuda as cuda
import numpy as np
import sys
from timing import time_region, time_region_cuda


def print_usage():
    print(f'Usage: {sys.argv[0]} <arraydim> <version>', file=sys.stderr)


@numba.njit(cache=True, parallel=True)
def gemv_v1(alpha, A, x, beta, y):
    N, M = A.shape
    y_ret = np.empty(N)
    for i in numba.prange(N):
        prod = 0.0
        for j in numba.prange(M):
            prod += A[i, j]*x[j]

        y_ret[i] = alpha*prod + beta*y[i]

    return y_ret


def gemv_v2(alpha, A, x, beta, y):
    return alpha*(A @ x) + beta*y


@cuda.jit('void(float64, Array(float64, 2, "F"), Array(float64, 1, "F"), '
          'float64, Array(float64, 1, "F"))')
# @cuda.jit
def _gemv_cuda(alpha, A, x, beta, y):
    i = cuda.grid(1)
    N, M = A.shape
    if i >= N:
        return

    prod = 0.0
    for j in range(M):
        prod += A[i, j]*x[j]

    y[i] = alpha*prod + beta*y[i]


BLOCK_SIZE = 128


@cuda.jit('void(float64, Array(float64, 2, "F"), Array(float64, 1, "F"), '
          'float64, Array(float64, 1, "F"))')
def _gemv_cuda_shared(alpha, A, x, beta, y):
    i = cuda.grid(1)
    N, M = A.shape
    if i >= N:
        return

    lx = cuda.shared.array(shape=BLOCK_SIZE, dtype=numba.float64)
    bsize = cuda.blockDim.x
    tid = cuda.threadIdx.x
    num_blocks = cuda.gridDim.x

    prod = 0.0
    for b in range(num_blocks):
        lx[tid] = x[tid + b*bsize]
        cuda.syncthreads()

        for j in range(BLOCK_SIZE):
            prod += A[i, j + b*bsize]*lx[j]

        cuda.syncthreads()

    y[i] = alpha*prod + beta*y[i]


def gemv_v3(alpha, A, x, beta, y):
    # Works only for square matrices
    with time_region_cuda() as t_xfer:
        d_A = cuda.to_device(A)
        d_x = cuda.to_device(x)
        d_y = cuda.to_device(y)

    N = A.shape[0]
    NB = N // BLOCK_SIZE
    if N % BLOCK_SIZE:
        NB += 1

    with time_region_cuda() as t_kernel:
        _gemv_cuda[NB, BLOCK_SIZE](alpha, d_A, d_x, beta, d_y)

    with time_region_cuda(t_xfer.elapsed_time()) as t_xfer:
        y_ret = d_y.copy_to_host()

    print(f'  CUDA transfer times: {t_xfer.elapsed_time()}')
    print(f'  CUDA kernel time: {t_kernel.elapsed_time()}')
    print(f'  Consumed memory bandwidth: '
          f'{1e-9*8*N*(N+2)/t_kernel.elapsed_time()} GB/s')
    return y_ret


def gemv_v4(alpha, A, x, beta, y):
    global _gemv_cuda

    _gemv_cuda = _gemv_cuda_shared
    return gemv_v3(alpha, A, x, beta, y)


@cuda.jit
def _vecadd_cuda(z, x, y):
    i = cuda.grid(1)
    N = x.shape[0]
    if i >= N:
        return

    z[i] = x[i] + y[i]


def vecadd(x, y):
    ret = np.empty(x.shape)
    with time_region_cuda() as t_xfer:
        d_x = cuda.to_device(x)
        d_y = cuda.to_device(y)
        d_z = cuda.device_array_like(x)

    num_blocks = N // BLOCK_SIZE
    if N % BLOCK_SIZE:
        num_blocks += 1

    with time_region_cuda() as t_kernel:
        _vecadd_cuda[num_blocks, BLOCK_SIZE](d_z, d_x, d_y)

    with time_region_cuda(t_xfer.elapsed_time()) as t_xfer:
        d_z.copy_to_host(ret)

    print(f'  CUDA transfer times: {t_xfer.elapsed_time()}')
    print(f'  CUDA kernel time: {t_kernel.elapsed_time()}')
    return ret


def validate(found, expected):
    return np.allclose(found, expected)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'{sys.argv[0]}: ERROR: too few arguments', file=sys.stderr)
        print_usage()
        sys.exit(1)

    try:
        N = int(sys.argv[1])
    except ValueError:
        print(f'{sys.argv[0]}: ERROR: array dimension must be an integer',
              file=sys.stderr)
        sys.exit(1)

    version = sys.argv[2]
    try:
        kernel = globals()['gemv_' + version]
    except KeyError:
        print(f'{sys.argv[0]}: ERROR: no such kernel version: {version}',
              file=sys.stderr)
        versions = []
        for name in list(globals().keys()):
            if not name.startswith('gemv_'):
                continue

            try:
                version = name.split('_', maxsplit=1)[1]
            except IndexError:
                continue

            versions.append(version)

        print(f'  Available versions: {", ".join(versions)}', file=sys.stderr)
        sys.exit(1)

    if N <= 0:
        print(
            f'{sys.argv[0]}: ERROR: array dimension must be a positive integer',
            file=sys.stderr
        )
        sys.exit(1)

    rng = np.random.default_rng()
    A = np.asarray(rng.random((N, N)), order='F')
    x = rng.random(N)
    y_orig = np.ones(N)
    alpha = 0.2
    beta = 1

    with time_region() as t_kernel:
        y = kernel(alpha, A, x, beta, y_orig)
        # y = vecadd(x, x)

    cuda.profile_stop()

    print(f'Elapsed time: {t_kernel.elapsed_time()} s')
    y_ref = alpha*(A @ x) + beta*y_orig
    # y_ref = x + x
    if not validate(y, y_ref):
        print(f'{sys.argv[0]}: ERROR: could not validate solution')
        print(f'    found = {y}')
        print(f'    expected = {y_ref}')
        sys.exit(1)
