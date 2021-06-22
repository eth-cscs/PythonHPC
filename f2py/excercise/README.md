## Exercise: Matrix-Matrix multiplication with Fortran90 and F2PY

 1. Complete the missing pieces on `mm.f90`:
    - Add the arguments and their definition to the subroutine based on what the code does.
    - Make two implementations, one using single precision floats (`real` in Fortran) and one using double precision (`double precision` in Fortran).
 2. Create a Python module using the `f2py` command.
 3. Check the doc string of the wrapper function.
 4. Create a script to test the function. How do you deal in Python with the two implementations with different precission? Which NumPy function can be used to check that our Fortran implementation is correct?

You can find the solution on the folders `solution-*`.
