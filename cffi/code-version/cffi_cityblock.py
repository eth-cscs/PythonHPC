import cffi


ffi = cffi.FFI()

with open('cityblock.h') as f:
    ffi.cdef(f.read())

with open('cityblock.c') as f:
    code = f.read()
    
ffi.set_source("_cityblock",
               code,
               extra_compile_args=["-fopenmp"],
               extra_link_args=["-lgomp"])

ffi.compile(verbose=True)
