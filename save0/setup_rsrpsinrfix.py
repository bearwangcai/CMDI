#!python2
#coding=utf-8
# RUNME: python setup.py build_ext --inplace


import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("rsrpsinrfix", ["rsrpsinrfix.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['/O2'],  # FAs: turn asm on
    )
]
setup(
    ext_modules = cythonize(extensions),
)
