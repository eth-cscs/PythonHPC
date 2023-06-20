from cython cimport nogil
from cython.parallel cimport parallel
from libc.stdio cimport printf

from openmp cimport omp_get_num_threads, omp_get_thread_num


def hello_omp():
    printf("---Hello from serial region---\n")

    with nogil, parallel():
        printf("Hello from %2d/%2d \n", omp_get_thread_num(),
               omp_get_num_threads())
