

# Make sure to use the same python we use on the notebooks' kernel!
# If you are outside jupyter, you may replace `python` on the
# by the path to the python executable used in the kernel launcher.
PYTHON = python

lib:
	$(CC) -c -O3 -fopenmp -fpic cityblock.c
	$(CC) -shared -lgomp cityblock.o -o libcityblock.so

pylib: lib
	$(PYTHON) cffi_cityblock.py
	rm _cityblock.c _cityblock.o cityblock.o

clean:
	rm -f *.so *.o _*c
