from cython cimport nogil
from cython.parallel cimport parallel
from libc.stdio cimport printf

from openmp cimport (omp_get_num_threads, omp_get_thread_num,
                     omp_get_num_procs)


def hello_omp():
    print("Inside serial region")
    print("Number of Processors:", omp_get_num_procs())
    print()

    with nogil, parallel():
        printf("Hello from thread %d out of %d\n", omp_get_thread_num(),
               omp_get_num_threads())
