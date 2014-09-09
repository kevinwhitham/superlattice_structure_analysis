# structure_metric.py
# Calculates the Minkowski structure metrics for an array of points
# Reference: Mickel, Walter, et al. "Shortcomings of the bond orientational order parameters for the analysis of disordered particulate matter." The Journal of chemical physics (2013)
# 20140902 Kevin Whitham

# for calculating facet angles
from math import hypot
from math import acos

# general math
import numpy as np
from scipy import special as sp
from scipy.spatial import distance_matrix

# plotting
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap

# Voronoi
from scipy.spatial import Voronoi

# for finding particle centers, diameters, etc.
from skimage.measure import regionprops
from skimage.filter import threshold_otsu,threshold_adaptive
from skimage.morphology import watershed, remove_small_objects
from skimage.feature import peak_local_max
from scipy import ndimage

# for command line interface
import argparse
from skimage import io as skimio
from os import path

# for matching the scale bar
from skimage.feature import match_template

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

# calculates the Minkowski structure metric of order l for each Voronoi cell in vor
# ignoring cells with vertices less than zero or greater than limits
def minkowski_structure_metric(vor,l,limits):

    x_max = limits[0]
    y_max = limits[1]
    
    msm = []
    
    region_index = 0

    # for each Voronoi cell
    for region in vor.regions:
        
        # for storing in msm
        region_index += 1
    
        # clear the perimeter value
        cellPerimeter = 0
    
        # skip infinite cells and empty regions
        if len(region) and np.all(np.not_equal(region,-1)):
            
            # check if any points are off the image
            polygon = np.asarray([vor.vertices[vert] for vert in region])
            
            if not (np.any(np.less(polygon,0)) or np.any(np.greater(polygon[:,0],x_max)) or np.any(np.greater(polygon[:,1],y_max))):

                # make a 1-D arrays to hold information about each facet
                facet_lengths = np.zeros(len(region))
                facetNormalAngles = np.zeros(len(region))
                interior_angles = np.zeros(len(region))
                
                # region contains the indices of the points which make the vertices of the simplex
                for region_vert_index in range(len(region)):
                    # euclidean distance
                    vertex1 = vor.vertices[region[region_vert_index]]
                    vertex2 = vor.vertices[region[(region_vert_index+1) % len(region)]]
                    facet_lengths[region_vert_index] = distance_matrix([vertex1],[vertex2],p=2)

                    # find the angle of the facets relative to the first facet
                    vertex3 = vor.vertices[region[(region_vert_index+2) % len(region)]]
                    interior_angles[region_vert_index] = angle(vertex1,vertex2,vertex3)

                    # the angle of the facet vertex2 to vertex3
                    # relative to the facet vertex1 to vertex2
                    # is 180-90-interior_angle + the sum of all previous interior angles
                    #if (region_vert_index+1) < len(region):
                    rotation = np.pi-interior_angles[region_vert_index]
                    facetNormalAngles[(region_vert_index+1) % len(region)] = (facetNormalAngles[region_vert_index]+rotation) % (2*np.pi)

                    # add to the cell perimeter
                    cellPerimeter += facet_lengths[region_vert_index]

                # make the facetNormalAngles actually the normal angles, not the facet angles
                facetNormalAngles += 3*np.pi/2

                # seems logical to make the facet angles relative to the facet with the largest length?
                #facetNormalAngles -= facetNormalAngles[np.argmax(facetLengths)]

                sum1 = 0
                sum2 = 0

                for m in range(-l,l+1):
                    for facet_length,normalAngle in zip(facet_lengths,facetNormalAngles):
                        sum1 += facet_length/cellPerimeter * sp.sph_harm(m,l,theta=normalAngle,phi=np.pi/2)

                    sum2 += np.abs(sum1)**2 #sum1*sum1.conjugate()
                    sum1 = 0

                msm.append([np.sqrt(4*np.pi/(2*l+1) * sum2),region_index-1])
    return msm

def make_binary_image(im, white_background, min_feature_size):

    if white_background:
        im = np.abs(im-np.max(im))

    # the block size should be large enough to include 4 to 9 particles
    binary = threshold_adaptive(im,block_size=10*min_feature_size)

    binary = remove_small_objects(binary,min_size=min_feature_size-1)

    # dilation of the binary image helps congeal large particles with low contrast
    # that get broken up by threshold
    #binary = ndimage.binary_dilation(binary,iterations=1)

    # 3 iterations is better for large particles with low contrast
    binary = ndimage.binary_closing(binary,iterations=1)

    return binary

def get_particle_centers(im, white_background, pixels_per_nm):

    if pixels_per_nm == 0:
        # default value
        pixels_per_nm = 5

    min_feature_size = int(pixels_per_nm*2)

    binary = make_binary_image(im, white_background, min_feature_size)

    # create a distance map to find the particle centers
    # as the points with maximal distance to the background
    distance = ndimage.distance_transform_edt(binary)

    # dilate the distance map to merge close peaks (merges multiple peaks in one particle)
    distance = ndimage.grey_dilation(distance,size=min_feature_size)

    # min_distance=5 for large particles
    local_maxi = peak_local_max(distance,indices=False,min_distance=min_feature_size)
    markers = ndimage.label(local_maxi)[0]

    labels = watershed(-distance, markers, mask=binary)

    # DEBUG
    # plt.figure(2)
    # plt.imshow(binary)
    # plt.figure(3)
    # plt.imshow(distance)
    # #plt.figure(4)
    # plt.imshow(markers,cmap=plt.cm.spectral,alpha=0.5)
    # plt.figure(5)
    # plt.imshow(labels,cmap=plt.cm.prism)
    # plt.show()

    # get the particle centroids
    regions = regionprops(labels)
    pts = []
    for props in regions:
        # centroid is [row, col] we want [col, row] aka [X,Y]
        # so reverse the order
        pts.append(props.centroid[::-1])

    return np.asarray(pts)

def get_image_scale(im):

    input_path = '../test_data/input/'

    # images of scale bars to match with the input image
    # second element is the scale in units of pixels/nm
    scale_bars = []
    scale_bars.append([skimio.imread(input_path+'Scale_1p425px_nm.tif',as_grey=True),1.425])
    scale_bars.append([skimio.imread(input_path+'Scale_3p44px_nm.tif',as_grey=True),3.44])
    scale_bars.append([skimio.imread(input_path+'Scale_5p14px_nm.tif',as_grey=True),5.14])
    scale_bars.append([skimio.imread(input_path+'Scale_2px_nm.tif',as_grey=True),2])
    scale_bars.append([skimio.imread(input_path+'Scale_9p4px_nm.tif',as_grey=True),9.4])

    match_score = []

    for scale_bar,pixels_per_nm in scale_bars:

        result = match_template(im,template=scale_bar)

        ij = np.unravel_index(np.argmax(result), result.shape)
        row, col = ij

        match_score.append(result[row][col])

    return scale_bars[np.argmax(match_score)][1]

parser = argparse.ArgumentParser()
parser.add_argument("img_file",help="image file to analyze")
parser.add_argument("pts_file",nargs='?',help="XY point data file",default='')
parser.add_argument("-b","--black", help="black background", action='store_true')
args = parser.parse_args()

im = skimio.imread(args.img_file,as_grey=True)
implot = plt.imshow(im)
implot.set_cmap('gray')

# import the points to analyze
#input_data_path = '../test_data/input/'
output_data_path = path.dirname(args.img_file)
filename = str.split(path.basename(args.img_file),'.')[0]

pixels_per_nm = get_image_scale(im)

background = 1
if args.black:
    background = 0

if len(args.pts_file) == 0:
    # find the centroid of each particle in the image
    pts = get_particle_centers(im,background,pixels_per_nm)

    # save the input points to a file
    header_string = 'Particle centroids\nX (pixel)\tY (pixel)'
    np.savetxt(output_data_path+'/'+filename+'_centroids.txt',pts,fmt='%.4e',delimiter='\t',header=header_string)
else:
    print("User specified points")
    pts = np.loadtxt(args.pts_file,skiprows=1,usecols=(1,2),ndmin=2)

# ideal square grid test case
#x = np.linspace(-0.5, 2.5, 5)
#y = np.linspace(-0.5, 2.5, 5)
#xx, yy = np.meshgrid(x, y)
#xy = np.c_[xx.ravel(), yy.ravel()]
#pts = xy

vor = Voronoi(pts)

plt.figure(1)
#plt.subplot(231)

bond_order = 4

# minkowski_structure_metric returns a list with metric,region_index
msm = minkowski_structure_metric(vor,bond_order,(im.shape[1],im.shape[0]))

# get a color map for mapping metric values to colors of some color scale
value_rgb_pairs = []
rgb_array = np.asarray([[0,0,0],[255,0,0],[255,50,34],[255,109,59],[255,177,102],[255,220,125],[255,245,160],[255,245,192],[255,255,255],[212,251,255],[160,253,255],[120,226,255],[81,177,255],[55,127,255],[31,81,255],[0,13,255]],dtype='f4')
rgb_array /= 255
rgb_list_norm = []
    
for value, color in zip(np.linspace(0,1,16),rgb_array):
    value_rgb_pairs.append((value,color))
    
custom_colormap = LinearSegmentedColormap.from_list(name="custom", colors=value_rgb_pairs, N=16)

symmetry_colormap = plt.get_cmap('RdBu_r')
    
cell_patches = []
for metric,region_index in msm:
    region_index = int(region_index)
    region = vor.regions[region_index]
    verts = np.asarray([vor.vertices[index] for index in region])
    cell_patches.append(Polygon(verts,closed=True,facecolor=symmetry_colormap(metric),edgecolor='k'))

pc = PatchCollection(cell_patches,match_original=False,cmap=symmetry_colormap,alpha=0.5)
pc.set_array(np.array([metric for metric,_ in msm]))
plt.gca().add_collection(pc)

# add the colorbar
plt.colorbar(pc)

# set the limits for the plot
# set the x axis range
plt.gca().set_xlim(0, im.shape[1])

# set the y-axis range and flip the y-axis
plt.gca().set_ylim(im.shape[0], 0)

# save this plot to a file
plt.gca().set_axis_off()
plt.gca().set_title('q'+str(bond_order))
plt.savefig(output_data_path+'/'+filename+'_q'+str(bond_order)+'_map.pdf',bbox_inches='tight')

# plot a histogram of the Minkowski structure metrics
plt.figure(2)
plt.hist([metric for metric,_ in msm],bins=len(msm)/4)
plt.savefig(output_data_path+'/'+filename+'_q'+str(bond_order)+'_hist.png', bbox_inches='tight')

# save the metrics to a file
header_string =     str(bond_order)+'-fold Minkowski metric (q'+str(bond_order)+')\n'
header_string +=    'Reference: Mickel, Walter, et al. The Journal of Chemical Physics (2013)\n'
header_string +=    'q'+str(bond_order)+'\tindex'
np.savetxt(output_data_path+'/'+filename+'_q'+str(bond_order)+'_data.txt',msm,fmt='%.4e',delimiter='\t',header=header_string)


# Calculate and plot the "bond strengths"
plt.figure(3)

binary_im = make_binary_image(im,background,0.306*pixels_per_nm)

# add the TEM image as the background
implot = plt.imshow(binary_im)
implot.set_cmap('gray')

bond_color_map = custom_colormap

# bond_strenths is a 2D symmetric array where bond_widths[i,j] is the bond strength
# between input point i and input point j
bond_widths = np.zeros((len(pts),len(pts)))
line_segments = []
bond_width_list = []
nn_dist = []

# to draw the bonds between points
for ridge_vert_indices,input_pair_indices in zip(vor.ridge_vertices,vor.ridge_points):
    
    if np.all(np.not_equal(ridge_vert_indices,-1)):
    
        # get the region enclosing this point
        vertex1 = vor.vertices[ridge_vert_indices[0]]
        vertex2 = vor.vertices[ridge_vert_indices[1]]
        
        x_vals = zip(vertex1,vertex2)[0]
        y_vals = zip(vertex1,vertex2)[1]
        
        if np.all((np.greater(x_vals,0),np.greater(y_vals,0),np.less(x_vals,im.shape[1]),np.less(y_vals,im.shape[0]))):
            # this is crude...lots of assumptions here
            # may not work well unless the im array is nearly binary because long facets
            # will always contribute more than small ones even if there is little material
            # would be better to use a binary image and assume all non-zero pixels are connected
            # or fit a function to find the width of the bridge
            facet_x_dist = np.abs(vertex2[0]-vertex1[0])+1
            facet_y_dist = np.abs(vertex2[1]-vertex1[1])+1
            range_len = np.max((facet_x_dist,facet_y_dist))
            x_range = np.linspace(vertex1[0],vertex2[0],num=range_len,dtype='i4')
            y_range = np.linspace(vertex1[1],vertex2[1],num=range_len,dtype='i4')

            # save the nearest neighbor distance
            input_pt1 = pts[input_pair_indices[0]]
            input_pt2 = pts[input_pair_indices[1]]
            nn_x_dist = np.abs(input_pt1[0]-input_pt2[0])
            nn_y_dist = np.abs(input_pt1[1]-input_pt2[1])
            nn_dist.append(np.sqrt(nn_x_dist**2 + nn_y_dist**2)/pixels_per_nm)

            # should fit a square function to get the actual width
            bond_widths[input_pair_indices[0],input_pair_indices[1]] = np.sum(binary_im[y_range,x_range])/pixels_per_nm
            bond_widths[input_pair_indices[1],input_pair_indices[0]] = bond_widths[input_pair_indices[0],input_pair_indices[1]]
    
            # make the lines
            line_segments.append(np.asarray([pts[input_pair_indices[0]],pts[input_pair_indices[1]]]))
            bond_width_list.append(bond_widths[input_pair_indices[0],input_pair_indices[1]])

bond_threshold = threshold_otsu(bond_widths)

bond_width_list_filtered = []
line_segments_filtered = []
nn_dist_filtered = []

for width,segment,nn in zip(bond_width_list,line_segments,nn_dist):
    if width >= bond_threshold:
        bond_width_list_filtered.append(width)
        line_segments_filtered.append(segment)
        nn_dist_filtered.append(nn)

lc = LineCollection(line_segments_filtered,cmap=bond_color_map)
lc.set_array(np.asarray(bond_width_list_filtered))
lc.set_linewidth(1)
plt.gca().add_collection(lc)
plt.colorbar(lc)

# set the x axis range
plt.gca().set_xlim(0, im.shape[1])

# set the y-axis range and flip the y-axis
plt.gca().set_ylim(im.shape[0], 0)

plt.savefig(output_data_path+'/'+filename+'_bond_map.pdf',bbox_inches='tight')

# save a version of the bond map without the image background
plt.figure(4)
plt.gca().add_collection(lc)
plt.colorbar(lc)
plt.savefig(output_data_path+'/'+filename+'_bond_map_nobg.pdf',bbox_inches='tight')

# make a map of the nn distances
plt.figure(5)
nn_dist_cmap = plt.get_cmap('RdBu_r')
implot = plt.imshow(im)
implot.set_cmap('gray')
nn_lc = LineCollection(line_segments_filtered,cmap=nn_dist_cmap)
nn_lc.set_array(np.asarray(nn_dist_filtered))
nn_lc.set_linewidth(1)
plt.gca().add_collection(nn_lc)
plt.colorbar(nn_lc)
plt.savefig(output_data_path+'/'+filename+'_nn_dist_map.pdf',bbox_inches='tight')

# plot a histogram of the "bond strengths"
plt.figure(6)

plt.hist(bond_width_list_filtered,bins=len(bond_width_list_filtered)/4)
plt.ylabel('Count')
plt.xlabel('Width (nm)')
plt.gca().set_title('Bond Widths (filtered)')
plt.savefig(output_data_path+'/'+filename+'_bond_hist.png', bbox_inches='tight')

header_string =     'Bond widths\n'
header_string +=    'width (nm)'
np.savetxt(output_data_path+'/'+filename+'_bond_data.txt',bond_width_list,fmt='%.4e',delimiter='\t',header=header_string)

# plot a histogram of the nearest neighbor distances
plt.figure(7)
plt.hist(nn_dist,bins=len(nn_dist)/4)
plt.ylabel('Count')
plt.xlabel('Neighbor Distance (nm)')
plt.savefig(output_data_path+'/'+filename+'_nn_dist_hist.png', bbox_inches='tight')

header_string =     'Nearest neighbor distances\n'
header_string +=     'distance (nm)'
np.savetxt(output_data_path+'/'+filename+'_nn_dist_data.txt',nn_dist,fmt='%.4e',delimiter='\t',header=header_string)

# show it all
#plt.show()