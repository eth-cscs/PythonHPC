from numpy.distutils.core import Extension, setup


compiler_flags = ['-O3', '-fopenmp']  # , '-m64', '-march=native', '-fPIC'
linker_flags = ['-lgomp']
f2py_options = ['--verbose']
language = 'f90'

edm = Extension(name='edm',
                sources=['euclidean.f90'],
                extra_f90_compile_args=compiler_flags,
                extra_f77_compile_args=compiler_flags,
                extra_compile_args=compiler_flags,
                extra_link_args=linker_flags,
                language=language,
                f2py_options=f2py_options)

cityblock = Extension(name='cbdm',
                      sources=['cityblock.f90'],
                      extra_f90_compile_args=compiler_flags,
                      extra_f77_compile_args=compiler_flags,
                      extra_compile_args=compiler_flags,
                      extra_link_args=linker_flags,
                      language=language,
                      f2py_options=f2py_options)

setup(name='metrics',
      ext_package='metrics',
      ext_modules=[edm, cityblock])
