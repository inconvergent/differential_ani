#!/usr/bin/python

from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

ext = cythonize("src/*.pyx", include_path = [numpy.get_include()])

setup(
  name = "speedup",
  cmdclass={'build_ext' : build_ext},
  include_dirs = [numpy.get_include()],
  ext_modules = ext
)
