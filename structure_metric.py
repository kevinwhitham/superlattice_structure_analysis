# structure_metric.py
# Calculates the Minkowski structure metrics for an array of points
# Reference: Mickel, Walter, et al. "Shortcomings of the bond orientational order parameters for the analysis of disordered particulate matter." The Journal of chemical physics (2013)
# 20140902 Kevin Whitham

from math import hypot
from math import acos
import numpy as np
from scipy import special as sp
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
#from scipy.spatial import Delaunay
#from scipy.spatial import delaunay_plot_2d

# returns the angle in radians of the interior angle made by 3 points
def angle(pt1, pt2, pt3):
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3
    
    # change coordinates to put pt2 at the origin
    x1 = x1-x2
    y1 = y1-y2
    x3 = x3-x2
    y3 = y3-y2

    x2 = 0
    y2 = 0

    
    inner_product = x1*x3 + y1*y3
    len1 = hypot(x1, y1)
    len2 = hypot(x3, y3)
    if len1 > 0 and len2 > 0:
        return acos(inner_product/(len1*len2))
    else:
        return 0

def minkowski_structure_metric(vor,l,img):
    # vor
    # calculate the Minkowski structure metric of order l for each Voronoi cell
    msm = []
    
    regionIndex = 0

    # for each Voronoi cell
    for region in vor.regions:
        
        # for storing in msm
        regionIndex += 1
    
        # clear the perimeter value
        cellPerimeter = 0
        cell_strength = 0
    
        # skip infinite cells and empty regions
        if len(region) and np.all(np.not_equal(region,-1)):
                
            # check if any points are off the image
            polygon = np.asarray([vor.vertices[vert] for vert in region])
            
            if not (np.any(np.less(polygon,0)) or np.any(np.greater(polygon[:,0],xmax)) or np.any(np.greater(polygon[:,1],ymax))):
                
                # make a 1-D arrays to hold information about each facet
                facet_lengths = np.zeros(len(region))
                facetNormalAngles = np.zeros(len(region))
                interior_angles = np.zeros(len(region))
                bond_strengths = np.zeros(len(region))
                
                # region contains the indices of the points which make the vertices of the simplex
                for regionVertIndex in range(len(region)):
                    # euclidean distance
                    vertex1 = vor.vertices[region[regionVertIndex]]
                    vertex2 = vor.vertices[region[(regionVertIndex+1) % len(region)]]
                    facet_lengths[regionVertIndex] = distance_matrix([vertex1],[vertex2],p=2)
                    
                    # this is crude...lots of assumptions here
                    # wont work well unless the img array is nearly binary because long facets
                    # will always contribute more than small ones even if there is little
                    # material there
                    row_dist = np.abs(vertex2[0]-vertex1[0])+1
                    col_dist = np.abs(vertex2[1]-vertex1[1])+1
                    range_len = np.max((row_dist,col_dist))
                    row_range = np.linspace(vertex1[0],vertex2[0],num=range_len,dtype='i4')
                    col_range = np.linspace(vertex1[1],vertex2[1],num=range_len,dtype='i4')
                    
                    bond_strengths[regionVertIndex] = np.sum(img[row_range,col_range])

                    # find the angle of the facets relative to the first facet
                    vertex3 = vor.vertices[region[(regionVertIndex+2) % len(region)]]
                    interior_angles[regionVertIndex] = angle(vertex1,vertex2,vertex3)

                    # the angle of the facet vertex2 to vertex3
                    # relative to the facet vertex1 to vertex2
                    # is 180-90-interior_angle + the sum of all previous interior angles
                    #if (regionVertIndex+1) < len(region):
                    rotation = np.pi-interior_angles[regionVertIndex]
                    facetNormalAngles[(regionVertIndex+1) % len(region)] = (facetNormalAngles[regionVertIndex]+rotation) % (2*np.pi)

                    # add to the cell perimeter
                    cellPerimeter += facet_lengths[regionVertIndex]
                    cell_strength += bond_strengths[regionVertIndex]

                # make the facetNormalAngles actually the normal angles, not the facet angles
                facetNormalAngles += 3*np.pi/2

                # seems logical to make the facet angles relative to the facet with the largest length?
                #facetNormalAngles -= facetNormalAngles[np.argmax(facetLengths)]

                sum1 = 0
                sum2 = 0

                for m in (-l,l):
                    for facet_length,normalAngle in zip(facet_lengths,facetNormalAngles):
                        sum1 += facet_length/cellPerimeter * sp.sph_harm(m,l,theta=normalAngle,phi=np.pi/2)

                    sum2 += np.abs(sum1)**2 #sum1*sum1.conjugate()
                    sum1 = 0

                #msm.append([np.sqrt(4*np.pi/(2*l+1) * sum2),regionIndex-1])
                msm.append([cell_strength,regionIndex-1])
    return msm

# import the points to analyze
pts = np.loadtxt("T6_79_2_004_XY.txt",skiprows=1,usecols=(1,2),ndmin=2)

# ideal square grid test case
#x = np.linspace(-0.5, 2.5, 5)
#y = np.linspace(-0.5, 2.5, 5)
#xx, yy = np.meshgrid(x, y)
#xy = np.c_[xx.ravel(), yy.ravel()]
#pts = xy

vor = Voronoi(pts)

plt.figure(1)

im = plt.imread("T6_79_2_004.tif")
implot = plt.imshow(im)

# minkowski_structure_metric returns a list with metric,regionIndex
msm = minkowski_structure_metric(vor,4,im.shape[1],im.shape[0],np.asarray(im))

# get a color map for mapping metric values to colors of some color scale
colormap = plt.get_cmap('RdYlGn_r')

patches = []
for metric,regionIndex in msm:
    regionIndex = int(regionIndex)
    region = vor.regions[regionIndex]
    verts = np.asarray([vor.vertices[index] for index in region])
    # could use metric to set the color here
    patches.append(Polygon(verts,closed=True,facecolor=colormap(metric),edgecolor='k'))

#test = np.asarray([vor.vertices[index] for index in vor.regions[20]])
#print(test)

pc = PatchCollection(patches,match_original=False,cmap=colormap,alpha=0.5)
pc.set_array(np.array([metric for metric,_ in msm]))
plt.gca().add_collection(pc)

# show the colorbar
plt.colorbar(pc)

# set the x axis range
plt.gca().set_xlim(0, im.shape[1])

# set the y-axis range and flip the y-axis
plt.gca().set_ylim(im.shape[0], 0)

plt.savefig('map.pdf',bbox_inches='tight')

plt.figure(2)
plt.hist([metric for metric,_ in msm],bins=len(msm)/4)
plt.savefig('hist.png', bbox_inches='tight')
#plt.show()