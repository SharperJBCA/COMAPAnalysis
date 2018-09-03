from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

sourcefiles = ['chistogram.pyx']
ext_modules = [Extension('chistogram', sourcefiles, include_dirs=[np.get_include()])]

setup(name = 'chistogram',
    cmdclass = {'build_ext':build_ext}, ext_modules=cythonize(ext_modules))