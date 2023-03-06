from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [Extension("loops", ["src/loops.pyx"],
                         include_dirs=[numpy.get_include()])]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
