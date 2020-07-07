# Implementation in Fortran90 of the Euclidean and Cityblock Distance Matrices

The code is under `./distance-f90` and `./distance-f90-pythonized`. Both have the same algorithms, but we write the code of `./distance-f90-pythonized` in such a way that F2PY is able to produce pythonic functions (with optional arguments and returning values, opposed to Fortran subroutines).
There are the files `cityblock.f90`, `euclidean.f90`, which are the sources, and `setup.py`, which can be used to compile the code into `.so` files that can be imported in python.

## Build with `setup.py`
The following command can be used to build the `.so` files
```bash
python setup.py build_ext -i
```
This will produce the folder `metrics` that contains `cbdm.cpython-36m-x86_64-linux-gnu.so` and `edm.cpython-36m-x86_64-linux-gnu.so`.
In general, the location of the installed modules is controled by command line options like `--prefix`:
```bash
python setup.py install --prefix /some/dir
```

## Build with F2PY by hand
Building with `setup.py` produces the command
```bash
f2py -c cityblock.f90 -m cbdm --f90flags='-fopenmp -O3' -lgomp
```
The `*so` files may as well be compiled directly with that command. Using the `setup.py` file is not mandatory.

```bash
f2py -c euclidean.f90 -m edm --f90flags='-fopenmp -O3' -lgomp
mkdir metrics
mv edm.cpy* metrics

f2py -c cityblock.f90 -m cbdm --f90flags='-fopenmp -O3' -lgomp
mkdir metrics
mv cityblock.cpy* metrics
```

## Conclusions
1. Implementing functions in Fortran and binding them to python with F2PY might result in significant speedups.
2. Move only small compute-intensive bits of a python program to Fortran90. This will result in cleaner code.
3. Keep in mind the types and order of the multidimensional arrays passed to Fortran functions.
