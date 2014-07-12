#!/usr/bin/python

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = "speedup",
  ext_modules = cythonize("src/*.pyx", include_path = [numpy.get_include()])
)

