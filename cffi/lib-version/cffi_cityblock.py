import os
import cffi


ffi = cffi.FFI()

with open('cityblock.h') as f:
    ffi.cdef(f.read())

ffi.set_source("_cityblock",
               '#include "cityblock.h"',
               extra_compile_args=["-fopenmp"],
               extra_link_args=["-lgomp"],
               libraries=["cityblock"],
               library_dirs=[os.getcwd()])

ffi.compile(verbose=True)
