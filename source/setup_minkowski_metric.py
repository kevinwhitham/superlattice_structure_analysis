from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy

setup(
    name = 'Minkowski Structure Metric',
    ext_modules = cythonize("minkowski_metric.pyx"),
    include_dirs=[numpy.get_include()]
)