__author__ = 'kevin'

# to turn off bounds checking
cimport cython

# for numpy arrays
import numpy as np
cimport numpy as np

from scipy.special import sph_harm

# single gives ~6 significant digits
# double gives ~15 significant digits
DTYPE = np.float32

# the array length = ceil(2*pi*10**SIG_FIG)
# here SIG_FIG = 4
DEF ALEN = 62832

ctypedef np.float32_t DTYPE_t

# cdef extern from "complexobject.h":
#
#     struct Py_complex:
#         double real
#         double imag
#
#     ctypedef class __builtin__.complex [object PyComplexObject]:
#         cdef Py_complex cval

@cython.boundscheck(False)
cpdef generate_sph_harm_values(unsigned int l):

    # calculate the real part of the spherical harmonic function
    # in the real plane (polar angle = pi/2)
    # only need to calculate for all positive even or positive odd m for even or odd l
    # because only the sign of the imaginary part changes for negative values of m
    # only need to iterate 2*pi*10^sig_fig times

    # one row for each angle
    # one column for each m value * 2 for real, imag
    cdef unsigned int col_len = <unsigned int>(np.ceil(l/2))
    cdef np.ndarray[DTYPE_t, ndim=2] values = np.empty((ALEN,2*col_len), dtype=DTYPE)
    cdef unsigned int row, col, m
    cdef unsigned int col_offset = <unsigned int>(l % 2)
    #cdef complex complex_result

    for row in range(ALEN):
        for col in range(0,2*col_len,2):
            m = (2-col_offset+col)
            complex_result = sph_harm(m,l,theta=2.0*np.pi*(<double>row/<double>ALEN),phi=np.pi/2.0).item()
            values[<unsigned int>row, <unsigned int>col] = complex_result.real
            values[<unsigned int>row, <unsigned int>col+1] = complex_result.imag

    np.save('sph_harm_'+str(l), values)