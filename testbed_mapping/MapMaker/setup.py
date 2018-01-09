#call with: python setup.py build_ext --inplace

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
    name = 'binning',
    ext_modules = [Extension('binning', 
                             ['binning.pyx'], 
                             include_dirs=[np.get_include()],
                             extra_compile_args=['-O3','-fopenmp'],
                             extra_link_args=['-fopenmp'])],
    cmdclass = {'build_ext': build_ext}
)

