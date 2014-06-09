from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'Numpy Utility Functions',
    ext_modules = cythonize("numpyutil.pyx"),
)

