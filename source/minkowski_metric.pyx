# minkowski_metric.pyx
# Calculates the Minkowski structure metrics for an array of points
# Reference: Mickel, Walter, et al. "Shortcomings of the bond orientational order parameters for the analysis of disordered particulate matter." The Journal of chemical physics (2013)
# 20140902 Kevin Whitham, Cornell University

import numpy as np
cimport numpy as np
cimport cython
from math import acos
from cpython cimport bool
from scipy.special import sph_harm

DUBTYPE = np.double
SINTYPE = np.float32

ctypedef np.double_t DUBTYPE_t
ctypedef np.float32_t SINTYPE_t

# returns the angle in radians of the interior angle made by 3 points
cdef inline double angle(double x1, double y1, double x2, double y2, double x3, double y3):

    cdef double len1,len2

    # change coordinates to put pt2 at the origin
    x1 = x1-x2
    y1 = y1-y2
    x3 = x3-x2
    y3 = y3-y2

    x2 = 0
    y2 = 0


    inner_product = x1*x3 + y1*y3
    len1 = np.sqrt(np.abs(x1)**2 + np.abs(y1)**2)
    len2 = np.sqrt(np.abs(x3)**2 + np.abs(y3)**2)
    if len1 > 0 and len2 > 0:
        return acos(inner_product/(len1*len2))
    else:
        return 0

# calculates the Minkowski structure metric of order l for each Voronoi cell in vor
# ignoring cells with vertices less than zero or greater than limits
@cython.boundscheck(False)
cpdef minkowski(np.ndarray[DUBTYPE_t, ndim=2] vor_vertices, vor_regions, int l, limits):
    
    cdef double x_max, y_max
    x_max = limits[0]
    y_max = limits[1]

    msm = []

    # Cython static type definitions
    cdef unsigned int region_index, region_vert_index, vert_count, facet_index
    cdef int m
    cdef double x1, y1, x2, y2, x3, y3, rotation, cell_perimeter, sum1_real, sum1_imag, sum2, zero_sum
    cdef bool out_of_bounds

    # load the spherical harmonic values from a file
    cdef np.ndarray[SINTYPE_t, ndim=2] sph_harm_values
    sph_harm_values = np.load('sph_harm_'+str(l)+'.npy')

    # for indexing the spherical harmonic value array by angle value
    # 62832 values gives 4 significant figures of precision
    # could do this dynamically: rows_per_radian = sph_harm_values.shape[0]/(2.0*np.pi)
    # but it will be off due to rounding/trunction error
    cdef DUBTYPE_t rows_per_radian = 10**4

    # make a 1-D arrays to hold information about each facet
    # assume all regions have fewer than 20 facets
    cdef np.ndarray[DUBTYPE_t, ndim=1] facet_lengths       = np.zeros(20,dtype=DUBTYPE)
    cdef np.ndarray[DUBTYPE_t, ndim=1] facet_normal_angles = np.zeros(20,dtype=DUBTYPE)
    cdef np.ndarray[DUBTYPE_t, ndim=1] interior_angles     = np.zeros(20,dtype=DUBTYPE)
    cdef np.ndarray[DUBTYPE_t, ndim=1] region              = np.zeros(20,dtype=DUBTYPE)

    # get the vertices for each Voronoi cell
    # region contains the indices of the vertices of the cell
    for region_index in range(len(vor_regions)):

        vert_count = len(vor_regions[region_index])

        assert vert_count <= 20

        # copy from vor_regions list to region array
        for region_vert_index in range(vert_count):
            region[region_vert_index] = vor_regions[region_index][region_vert_index]

        # clear the perimeter value
        cell_perimeter = 0

        # skip infinite cells and empty regions
        if vert_count and np.all(np.not_equal(region[:vert_count],-1)):

            for region_vert_index in range(vert_count):

                x1,y1 = vor_vertices[region[region_vert_index]]
                x2,y2 = vor_vertices[region[<unsigned int>((region_vert_index+1) % vert_count)]]
                x3,y3 = vor_vertices[region[<unsigned int>((region_vert_index+2) % vert_count)]]

                # check if any points are off the image
                if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0 and x1 <= x_max and x2 <= x_max and y1 <= y_max and y2 <= y_max:

                    out_of_bounds = False


                    # euclidean distance
                    facet_lengths[region_vert_index] = np.sqrt(np.abs(x1-x2)**2 + np.abs(y1-y2)**2)

                    # find the angle of the facets relative to the first facet

                    interior_angles[region_vert_index] = angle(x1,y1,x2,y2,x3,y3)

                    # the angle of the facet vertex2 to vertex3
                    # relative to the facet vertex1 to vertex2
                    # is 180-90-interior_angle + the sum of all previous interior angles
                    #if (region_vert_index+1) < vert_count:
                    rotation = np.pi-interior_angles[region_vert_index]
                    facet_normal_angles[<unsigned int>((region_vert_index+1) % vert_count)] = ((facet_normal_angles[region_vert_index]+rotation) % (2.0*np.pi))

                    # add to the cell perimeter
                    cell_perimeter += facet_lengths[region_vert_index]

                else:

                    out_of_bounds = True
                    break

            # seems logical to make the facet angles relative to the facet with the largest length?
            #facet_normal_angles -= facet_normal_angles[np.argmax(facetLengths)]

            if not out_of_bounds:

                sum1_real   = 0.0
                sum1_imag   = 0.0
                sum2        = 0.0
                zero_sum    = 0.0

                # handle the m = 0 case
                for facet_index in range(vert_count):
                    zero_sum += facet_lengths[facet_index]/cell_perimeter * <double> (np.real(sph_harm(0,l,0,np.pi/2)).item())

                sum2 += zero_sum**2

                # calculate only the even values of m (2...l-2,l)
                # the sph_harm_values array has 2 cols for each m value (real and imaginary)
                # e.g. for l = 6:
                #   col 0,1 are real and im values for m = 2
                #   col 2,3 are real and im values for m = 4
                #   col 4,5 are real and im values for m = 6
                for col in range(0,<unsigned int>sph_harm_values.shape[1],2):
                    for facet_index in range(vert_count):
                        row_index = <unsigned int> np.ceil(facet_normal_angles[facet_index]*rows_per_radian)

                        # wrap-around, this shouldn't happen except maybe at 62832
                        row_index = row_index % 62832

                        # row is the angle, col is the m-number
                        sum1_real += facet_lengths[facet_index]/cell_perimeter * sph_harm_values[row_index,col]
                        sum1_imag += facet_lengths[facet_index]/cell_perimeter * sph_harm_values[row_index,col+1]

                    # factor of 2 because the negative m values are the same but with a sign change on the imaginary part
                    sum2 += 2.0*(sum1_real*sum1_real - sum1_imag*(-sum1_imag))
                    sum1_real = 0
                    sum1_imag = 0

                msm.append([region_index,np.sqrt(4*np.pi/(2*l+1) * sum2)])
    return msm
