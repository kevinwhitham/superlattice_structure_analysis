from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'Generate spherical harmonic values',
    ext_modules = cythonize("generate_sph_harm.pyx"),
    include_dirs=[numpy.get_include()]
)
