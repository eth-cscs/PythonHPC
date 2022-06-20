import numba.cuda as cuda
import numpy as np
import sys
from timing import time_region, time_region_cuda


def print_usage():
    print(f'Usage: {sys.argv[0]} <arraydim>', file=sys.stderr)


@cuda.jit('void(Array(float64, 1, "C"), Array(float64, 1, "C"), '
          'Array(float64, 1, "C"))')
def _vecadd_cuda(z, x, y):
    i = cuda.grid(1)
    N = x.shape[0]
    if i >= N:
        return

    z[i] = x[i] + y[i]


def vecadd(x, y):
    with time_region_cuda() as t_xfer:
        d_x = cuda.to_device(x)
        d_y = cuda.to_device(y)
        d_z = cuda.device_array_like(x)

    block_size = 128
    num_blocks = N // block_size
    if N % block_size:
        num_blocks += 1

    with time_region_cuda() as t_kernel:
        _vecadd_cuda[num_blocks, block_size](d_z, d_x, d_y)

    with time_region_cuda(t_xfer.elapsed_time()) as t_xfer:
        ret = cuda.pinned_array(N)
        d_z.copy_to_host(ret)

    print(f'  CUDA xfer overheads: {t_xfer.elapsed_time()} s')
    print(f'  CUDA kernel time:    {t_kernel.elapsed_time()} s')
    return ret


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'{sys.argv[0]}: ERROR: too few arguments', file=sys.stderr)
        print_usage()
        sys.exit(1)

    try:
        N = int(sys.argv[1])
    except ValueError:
        print(f'{sys.argv[0]}: ERROR: array dimension must be an integer',
              file=sys.stderr)
        sys.exit(1)

    if N <= 0:
        print(
            f'{sys.argv[0]}: ERROR: array dimension must be a positive integer',
            file=sys.stderr
        )
        sys.exit(1)

    rng = np.random.default_rng()
    x = rng.random(N)
    y = rng.random(N)
    with time_region() as t_gpu:
        z = vecadd(x, y)

    with time_region() as t_ref:
        z_ref = x + y

    cuda.profile_stop()

    print(f'Total time (Numba CUDA): {t_gpu.elapsed_time()} s')
    print(f'Total time (NumPy):      {t_ref.elapsed_time()} s')
    if not np.allclose(z, z_ref):
        print(f'{sys.argv[0]}: ERROR: could not validate solution')
        print(f'    found = {z}')
        print(f'    expected = {z_ref}')
        sys.exit(1)
