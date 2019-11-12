# Python Binding with cffi a Cityblock Distance Matrix function written in C

Here we implement a C function to compute the Cityblock distance matrix and use cffi to write a python binding to
it.

The C code consists of the source and header files `cityblock.c` and `cityblock.h`. The file `cffi_cityblock.py` is used to build a `.so` file that can be imported within python.

## How it works
The idea is
 1. Compile the C code into a dynamical library `lib<name>.so`.
    This library can't be imported in python.
 2. With CFFI, create a python-importable library that act as glue between the
    `lib<name>.so` containing our function and python.
 3. Import the python-importable library in pyhton and use it!

## How to build it
The following steps describe how to build the `.so` file:
 1. Build the `libcityblock.so` library that contains the function `cbdm`:
    ```bash
    cc -c -fpic cityblock.c
    cc -shared cityblock.o -o libcityblock.so
    ```
    Other compilation flags can be added, for instance `-fopenmp -O3` to add OpenMP support and compiler optimizations with GCC.
    On the [Code compilation](https://user.cscs.ch/computing/compilation/) page of CSCS User portal, it can be found
    the equivalent flags for all compilers supported on Piz Daint.
 2. Build the python-importable `.so` file:
    ```bash
    python cffi_cityblock.py
    ```
    This will first generate the file `_cityblock.c` that contains the C-code that act as glue between our function on `cityblock.c` and python.
    This file is then compiled into `_cityblock.cpython-36m-x86_64-linux-gnu.so` which can be imported in python.

All these steps are put together on the `Makefile` file. Just run
```bash
make pylib
```
to build the libraries.

## Importing and using the function within python    
The script `test_cityblock.py` shows how to import and work with the C function within Python.

>>The python-importable `.so` file is linked against `libcityblock.so`.
    Because of this, at run time, it is necessary to export the directory where the library `libcityblock.so` is,
    to the `LD_LIBRARY_PATH`. For a quick test, if the library is on
    the same directory where the script will be run, the following command `export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH` wil do it.
    
The C function needs to be imported as
```bash
from _cityblock.lib import cbdm
```

The function accepts as argument python's `int` type, but NumPy arrays need to be cast to `ctypes.data`:
```bash
from cffi import FFI


x = np.random.random([10, 3])

ffi = FFI()

x_c = ffi.cast('double *', x.ctypes.data)
```
This doesn't create new memory allocations. `x_c` corresponds to the same memory block where `x` is stored.
This means that modifying `x_c` will be reflected on `x`. Also, note that `x_c` is a continuous block of memory
and even if it comes from a `(10, 3)` NumPy array, in C it is seen as an one-dimensional array.
In this case a `double x[30]`.

Finally, the function can be called like this
```bash
cbdm(x_c, x_c, r_c, nsamples, nfeat)
```
Note that this function doesn't return an array with the distance matrix. Instead an array, casted
to `ctypes.data` (`r_c`), in which the matrix will be written, needs to be passed as argument to the function .

All this have been put together in `test_cityblock.py`.
