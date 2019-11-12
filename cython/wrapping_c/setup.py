from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("my_primes",
sources=["my_primes.pyx", "primes.c"])

setup(
    name="my_primes",
    ext_modules = cythonize([ext],
                            compiler_directives={
                                'language_level' : '3'
                            })
)
