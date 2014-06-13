from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

#setup(
#    name = 'Numpy Utility Functions',
#    ext_modules = cythonize("numpyutil.pyx",
#			    include_path = [np.get_include()])
#)

extensions = [
    Extension("numpyutil", ["numpyutil.pyx"],
	       include_dirs = [np.get_include()])

]
setup(
    name = 'Numpy Utility Functions',
    ext_modules = cythonize(extensions),
)
