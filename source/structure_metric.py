# structure_metric.py
# Calculates the Minkowski structure metrics for an array of points
# Reference: Mickel, Walter, et al. "Shortcomings of the bond orientational order parameters for the analysis of disordered particulate matter." The Journal of chemical physics (2013)
# 20140902 Kevin Whitham, Cornell University

# general math
import numpy as np
from scipy import stats # for stats.mode

# for data structures
import collections
import scipy.sparse as sparse

# for disabling annoying warnings
import warnings

# plotting
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable, Size

# Voronoi
from scipy.spatial import Voronoi

# for finding particle centers, diameters, etc.
from skimage.measure import regionprops
from skimage.filter import threshold_otsu,threshold_adaptive, threshold_isodata
from skimage.morphology import watershed, remove_small_objects, binary_erosion, binary_dilation, binary_closing, binary_opening
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.draw import circle
from skimage.morphology import disk
from skimage.filter.rank import tophat

# for command line interface
import argparse
from skimage import io as skimio
from os import path

# for matching the scale bar
from skimage.feature import match_template

# minkowski structure metric function written and compiled with Cython
from minkowski_metric import minkowski
#import pyximport; pyximport.install()
#import minkowski_metric


def make_binary_image(im, white_background, min_feature_size, adaptive):

    image = im.copy()

    if white_background:
        image = np.abs(image-np.max(image))

    # get rid of large background patches before local thresholding
    # do a global threshold
    thresh = threshold_isodata(image)
    binary = image < thresh
    
    # debug
    plt.figure(0)
    plt.imshow(binary)
    plt.gca().set_title('Global thresh')
    plt.show()
    
    # remove noise in the binary distance image
    # this gets rid of junk in the gap areas
    fill_size = 2
    binary_closing(binary, selem=np.ones((fill_size,fill_size)), out=binary)
    
    # debug
    plt.figure(0)
    plt.imshow(binary)
    plt.gca().set_title('Global thresh, smalls removed')
    plt.show()

    # make a distance map of the inverted image
    distance = ndimage.distance_transform_edt(binary)

    # do a global threshold on the distance map to select the biggest objects
    # larger than a minimum size to prevent masking of particles in images with no empty patches
    dist_thresh = threshold_isodata(distance)
    mask = distance < dist_thresh
    
    # debug
    plt.figure(0)
    plt.imshow(mask)
    plt.gca().set_title('Distance based mask')
    plt.show()
    
    # remove areas of background smaller than a certain size in the mask
    # this fills in small pieces of mask between particles where the voronoi
    # vertices will end up
    dilation_size = max(int(1),int(2.5*min_feature_size))
    binary_closing(mask, selem=np.ones((dilation_size,dilation_size)), out=mask)
    
    # debug
    plt.figure(0)
    plt.imshow(mask)
    plt.gca().set_title('Modified mask')
    plt.show()
    
    #image = image * mask
    
    # adaptive was here

    binary = (1-binary) * mask

    # get rid of speckle
    binary = remove_small_objects(binary,min_size=max(min_feature_size,2))
    
    # 3 iterations is better for large particles with low contrast
    binary = ndimage.binary_closing(binary,iterations=1)

    # DEBUG
    plt.figure(2)
    plt.imshow(binary)
    plt.gca().set_title('Masked Binary')
    plt.show()

    return binary, mask
    
def adaptive_binary_image(im, mask, white_background, min_feature_size):

    image = im.copy()

    if white_background:
        image = np.abs(image-np.max(image))
    
    # the block size should be large enough to include 4 to 9 particles
    local_size = 10*min_feature_size
    binary = threshold_adaptive(image,block_size=local_size)
    
    # debug
    plt.figure(0)
    plt.imshow(binary)
    plt.gca().set_title('Adaptive thresh')
    plt.show()
    
    # 3 iterations is better for large particles with low contrast
    binary = ndimage.binary_closing(binary,iterations=1)
    
    # debug
    plt.figure(0)
    plt.imshow(binary)
    plt.gca().set_title('closed adap. thresh')
    plt.show()
    
    return binary

def morphological_threshold(im, white_background, mean_radius, min_feature_size, mask):

    # debug
    #print('Mean radius (px): '+str(mean_radius))
    #plt.figure(6)
    #plt.hist(radii*pixels_per_nm,bins=len(radii)/4)
    #plt.xlabel('particle radius (pixels)')
    #plt.show()

    if white_background:
        im_mod = np.abs(im-np.max(im)).copy()
    else:
        im_mod = im.copy()

    # set large areas of background to zero using the mask
    im_mod = im_mod * mask

    # debug
    #plt.figure(8)
    #plt.imshow(im_mod)
    #plt.gca().set_title('im_mod after masking')

    #topped_im = tophat(im_mod,disk(mean_radius),mask=mask)
    matched_im = match_template(im_mod,template=np.pad(disk(int(mean_radius/2)),pad_width=int(mean_radius), mode='constant',constant_values=0),pad_input=True)

    thresh = threshold_otsu(matched_im)
    matched_im_bin = matched_im > thresh
    
    matched_im_bin *= mask

    distance = ndimage.distance_transform_edt(matched_im_bin)

    # dilate the distance map to merge close peaks (merges multiple peaks in one particle)
    distance = ndimage.grey_dilation(distance,size=2*min_feature_size)

    local_maxi = peak_local_max(distance,indices=False,min_distance=min_feature_size)
    markers = ndimage.label(local_maxi)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels_th = watershed(-distance, markers, mask=matched_im_bin)

    # debug
    plt.figure(9)
    plt.imshow(matched_im)
    plt.gca().set_title('template match')
    plt.figure(10)
    plt.imshow(matched_im_bin)
    plt.gca().set_title('bin')
    plt.figure(11)
    plt.imshow(labels_th,cmap=plt.cm.prism)
    plt.gca().set_title('labels')
    plt.figure(12)
    plt.imshow(distance)
    plt.gca().set_title('distance')
    plt.show()

    return labels_th


def get_particle_centers(im, white_background, pixels_per_nm, morph):

    if pixels_per_nm == 0:
        # default value
        pixels_per_nm = 1.0

    # minimum size object to look for
    min_feature_size = int(3) #int(3*pixels_per_nm)

    binary, mask = make_binary_image(im, white_background, min_feature_size, adaptive=0)

    # create a distance map to find the particle centers
    # as the points with maximal distance to the background
    distance = ndimage.distance_transform_edt(binary)

    # dilate the distance map to merge close peaks (merges multiple peaks in one particle)
    distance = ndimage.grey_dilation(distance,size=min_feature_size)

    # min_distance=5 for large particles
    local_maxi = peak_local_max(distance,indices=False,min_distance=min_feature_size)
    markers = ndimage.label(local_maxi)[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels = watershed(-distance, markers, connectivity=None, offset=None, mask=binary)

    # get the particle radii
    regions = regionprops(labels)
    radii = []
    
    for props in regions:
        # define the radius as half the average of the major and minor diameters
        radii.append(((props.minor_axis_length+props.major_axis_length)/4)/pixels_per_nm)

    # minimum size object to look for
    mean_radius = int(np.mean(radii)*pixels_per_nm)
    mode_radius = int(stats.mode(radii,axis=None)[0].item()*pixels_per_nm)
    
    feature_size = mode_radius
    
    # don't get stuck with 1.0 pixel features
    if mode_radius <= min_feature_size:
        feature_size = max(mean_radius, min_feature_size)
    
    print('Mean radius global threshold (px): %(rad).2f' % {'rad':np.mean(radii)*pixels_per_nm})
    print('Mode radius global threshold (px): %(rad).2f' % {'rad':mode_radius})
    
    binary = adaptive_binary_image(im, mask, white_background, feature_size)

    # create a distance map to find the particle centers
    # as the points with maximal distance to the background
    distance = ndimage.distance_transform_edt(binary)

    # dilate the distance map to merge close peaks (merges multiple peaks in one particle)
    distance = ndimage.grey_dilation(distance,size=feature_size)

    local_maxi = peak_local_max(distance,indices=False,min_distance=feature_size)
    markers = ndimage.label(local_maxi)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels = watershed(-distance, markers, connectivity=None, offset=None, mask=binary)

    regions = regionprops(labels)
    radii = []
    
    for props in regions:
        # define the radius as half the average of the major and minor diameters
        radii.append(((props.minor_axis_length+props.major_axis_length)/4)/pixels_per_nm)

    # minimum size object to look for
    mean_radius = int(np.mean(radii)*pixels_per_nm)
    print('Mean radius adaptive threshold (px): %(rad).2f' % {'rad':np.mean(radii)*pixels_per_nm})
    
    # remove areas of background smaller than a certain size in the mask
    # this fills in small pieces of mask between particles where the voronoi
    # vertices will end up
    dilation_size = max(int(1),int(4*mean_radius))
    binary_closing(mask, selem=np.ones((dilation_size,dilation_size)), out=mask)
    
    # debug
    plt.figure(0)
    plt.imshow(mask)
    plt.gca().set_title('closed mask adap. thresh')
    plt.show()
    
    # morphological thresholding
    if morph:
        labels = morphological_threshold(im, white_background, mean_radius, mean_radius/2, mask)

    # DEBUG
    # plt.figure(2)
    # plt.imshow(binary)
    # plt.figure(3)
    # plt.imshow(distance)
    # plt.figure(4)
    # plt.imshow(markers,cmap=plt.cm.spectral,alpha=0.5)
    # plt.figure(5)
    # plt.imshow(labels,cmap=plt.cm.prism)
    # plt.gca().set_title('distance_mapped_labels')
    # plt.show()

    # get the particle centroids again, this time with better thresholding
    regions = regionprops(labels)
    pts = []
    radii = []

    for props in regions:
        # centroid is [row, col] we want [col, row] aka [X,Y]
        # so reverse the order
        pts.append(props.centroid[::-1])

        # define the radius as half the average of the major and minor diameters
        radii.append(((props.minor_axis_length+props.major_axis_length)/4)/pixels_per_nm)

    # debug
    if morph:
        print('Mean radius morphological threshold (px): %(rad).2f' % {'rad':np.mean(radii)*pixels_per_nm})
    
    return np.asarray(pts,dtype=np.double), np.asarray(radii,dtype=np.double).reshape((-1,1)), mask

def get_image_scale(im):

    scale = 0.0
    bar_width = 0

    input_path = '../test_data/input/'

    # images of scale bars to match with the input image
    # second element is the scale in units of pixels/nm
    scale_bars = []
    scale_bars.append([skimio.imread(input_path+'Scale_0p654px_nm.tif',as_grey=True),0.654])
    scale_bars.append([skimio.imread(input_path+'Scale_1p425px_nm.tif',as_grey=True),1.425])
    scale_bars.append([skimio.imread(input_path+'Scale_2p88px_nm.tif',as_grey=True),2.88])
    scale_bars.append([skimio.imread(input_path+'Scale_3p44px_nm.tif',as_grey=True),3.44])
    scale_bars.append([skimio.imread(input_path+'Scale_5p14px_nm.tif',as_grey=True),5.14])
    scale_bars.append([skimio.imread(input_path+'Scale_2px_nm.tif',as_grey=True),2])
    scale_bars.append([skimio.imread(input_path+'Scale_7p258px_nm.tif',as_grey=True),7.258])
    scale_bars.append([skimio.imread(input_path+'Scale_9p4px_nm.tif',as_grey=True),9.4])
    scale_bars.append([skimio.imread(input_path+'Scale_4p51px_nm.tif',as_grey=True),4.51])

    match_score = []

    for scale_bar,pixels_per_nm in scale_bars:

        result = match_template(im,template=scale_bar)

        ij = np.unravel_index(np.argmax(result), result.shape)
        row, col = ij

        match_score.append(result[row][col])

    # the match score should be about 0.999999
    if np.max(match_score) > 0.99:

        scale = scale_bars[np.argmax(match_score)][1]
        bar_width = scale_bars[np.argmax(match_score)][0].shape[1]
        
        print('Scale: '+str(scale)+' pixels/nm, Score: '+str(np.max(match_score)))

    else:
        print('!!!!!!!!!!!!!!!!!!!! No scale bar found !!!!!!!!!!!!!!!!!!!!!')

    return [np.double(scale),((985,1369-bar_width),(1025,1369))]

def create_custom_colormap():

    # get a color map for mapping metric values to colors of some color scale
    value_rgb_pairs = []
    rgb_array = np.asarray([[0,0,0],[255,0,0],[255,50,34],[255,109,59],[255,177,102],[255,220,125],[255,245,160],[255,245,192],[255,255,255],[212,251,255],[160,253,255],[120,226,255],[81,177,255],[55,127,255],[31,81,255],[0,13,255]],dtype='f4')
    rgb_array /= 255
    rgb_list_norm = []

    for value, color in zip(np.linspace(0,1,16),rgb_array):
        value_rgb_pairs.append((value,color))

    return LinearSegmentedColormap.from_list(name="custom", colors=value_rgb_pairs, N=16)

def plot_symmetry(im,msm,bond_order,symmetry_colormap, mask, outline, map_edge_particles):
    
    cell_patches = []
    metric_list = []
    for region_index,metric in msm:
        plot_this_cell = 0
        region_index = int(region_index)
        region = vor.regions[region_index]
        verts = np.asarray([vor.vertices[index] for index in region])
        
        # don't plot cells inside masked off regions of the image (blank patches)
        int_verts = np.asarray(verts,dtype='i4')
        if map_edge_particles:
            if np.any(mask[int_verts[:,1],int_verts[:,0]] == 1):
                plot_this_cell = 1
        else:
            if np.all(mask[int_verts[:,1],int_verts[:,0]] > 0):
                plot_this_cell = 1
                
        if plot_this_cell:
            if outline:
                cell_patches.append(Polygon(verts,closed=True,facecolor='none',edgecolor='r'))
            else:
                cell_patches.append(Polygon(verts,closed=True,edgecolor='none'))
        
            metric_list.append(metric)
            
    if outline:
        pc = PatchCollection(cell_patches,match_original=True,alpha=1)
    else:
        pc = PatchCollection(cell_patches,match_original=False, cmap=symmetry_colormap, edgecolor='k', alpha=1)
        pc.set_array(np.asarray(metric_list))
        
    plt.gca().add_collection(pc)

    # set the limits for the plot
    # set the x axis range
    plt.gca().set_xlim(0, im.shape[1])

    # set the y-axis range and flip the y-axis
    plt.gca().set_ylim(im.shape[0], 0)

    # save this plot to a file
    plt.gca().set_axis_off()

    if not outline:
        # add the colorbar
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes(position='right', size='5%', pad = 0.05)
        cbar = plt.colorbar(pc, cax=cax)
        cax.set_xlabel('$\Psi_'+str(bond_order)+'$', fontsize=18)
        cax.xaxis.set_label_position('top')
        cax.xaxis.set_label_coords(0.5, 1.04)
    
    plt.savefig(output_data_path+'/'+filename+'_Psi'+str(bond_order)+'_map.png',bbox_inches='tight',dpi=300)

    # plot a histogram of the Minkowski structure metrics
    plt.figure(2)
    plt.hist(metric_list,bins=len(msm)/4)
    plt.xlabel('$\Psi_'+str(bond_order)+'$')
    plt.ylabel('Count')
    plt.savefig(output_data_path+'/'+filename+'_Psi'+str(bond_order)+'_hist.png', bbox_inches='tight')

def plot_bonds(im,line_segments,bond_list):

    # add the TEM image as the background
    implot = plt.imshow(im)
    implot.set_cmap('gray')

    bond_color_map = create_custom_colormap()

    lc = LineCollection(line_segments,cmap=bond_color_map)
    lc.set_array(np.asarray(bond_list))
    lc.set_linewidth(1)
    plt.gca().add_collection(lc)
    bond_color_bar = plt.colorbar(lc)

    # set the x axis range
    plt.gca().set_xlim(0, im.shape[1])

    # set the y-axis range and flip the y-axis
    plt.gca().set_ylim(im.shape[0], 0)

    bond_color_bar.ax.set_ylabel('Bond Width (nm)')
    plt.gca().set_axis_off()

def plot_nn_distance(im,line_segments,nn_distance_list):

    nn_dist_cmap = plt.get_cmap('RdBu_r')

    # show the background image
    implot = plt.imshow(im)
    implot.set_cmap('gray')

    nn_lc = LineCollection(line_segments,cmap=nn_dist_cmap)
    nn_lc.set_array(np.asarray(nn_distance_list))
    nn_lc.set_linewidth(1)
    plt.gca().add_collection(nn_lc)
    nn_color_bar = plt.colorbar(nn_lc)
    nn_color_bar.ax.set_ylabel('NN Dist. (nm)')
    plt.gca().set_axis_off()
    plt.gca().set_title('Neighbor Distance')




parser = argparse.ArgumentParser()
parser.add_argument('-b','--black', help='black background', action='store_true')
parser.add_argument('-n','--noplot', help='do not plot data', action='store_true')
parser.add_argument('-m','--morph', help='use morphological filtering', action='store_true')
parser.add_argument('-o','--outline', help='draw Voronoi cell outlines only', action='store_true')
parser.add_argument('-e','--edge', help='plot the symmetry metric of particles at superlattice edge', action='store_true')
parser.add_argument('-mc','--montecarlo', help='caclulate NN dist, Bond width, etc.', action='store_true')
parser.add_argument('order',help='order of the symmetry function', type=int, default=4)
parser.add_argument('img_file',help='image file to analyze')
parser.add_argument('pts_file',nargs='?',help='XY point data file',default='')
parser.add_argument('pix_per_nm',nargs='?',help='scale in pixels per nm',default=0.0,type=float)
args = parser.parse_args()

im = skimio.imread(args.img_file,as_grey=True)
im_original = np.empty_like(im)
np.copyto(im_original,im)

# import the points to analyze
#input_data_path = '../test_data/input/'
output_data_path = path.dirname(args.img_file)
filename = str.split(path.basename(args.img_file),'.')[0]

background = 1
if args.black:
    background = 0
    print('User specified black background')
    
if args.morph:
    print('Using morphological filtering')
    
if args.edge:
    print('Plotting metric of edge particles')

pixels_per_nm = args.pix_per_nm

if pixels_per_nm == 0:

    pixels_per_nm,bar_corners = get_image_scale(im)

    if not pixels_per_nm == 0:

        # remove the scalebar from the image
        if background:
            im[bar_corners[0][0]:bar_corners[1][0]+1,bar_corners[0][1]:bar_corners[1][1]+1] = np.max(im)
        else:
            im[bar_corners[0][0]:bar_corners[1][0]+1,bar_corners[0][1]:bar_corners[1][1]+1] = 0

else:
    print("User specified scale: "+str(pixels_per_nm)+" pixels per nm")

if len(args.pts_file) == 0:
    # find the centroid of each particle in the image
    pts, radii, mask = get_particle_centers(im,background,pixels_per_nm,args.morph)

    assert len(pts) == len(radii)

    particle_data = np.hstack((pts,radii))

    # save the input points to a file
    header_string = 'Particle centroids in pixel units\n'
    header_string += 'Particle radii - half of the average of the major and minor diameters of an ellipse fit to the particle area\n'
    header_string += 'total particles: '+str(len(pts))+'\n'
    header_string += 'mean radius: %(rad_nm).2g (nm) %(rad_px).2g (pixels)\n' % {'rad_nm':np.mean(radii), 'rad_px':np.mean(radii)*pixels_per_nm}
    header_string += 'X (pixel)\tY (pixel)\tradius (nm)'
    np.savetxt(output_data_path+'/'+filename+'_particles.txt',particle_data,fmt='%.4e',delimiter='\t',header=header_string)
    np.savez(output_data_path+'/'+filename+'_particles.npz',pixels_per_nm=pixels_per_nm, centroids=pts, radii=radii)

else:
    print("User specified points")
    pts = np.loadtxt(args.pts_file,skiprows=1,usecols=(1,2),ndmin=2)
    radii = []
    mask = np.ones(im.shape, dtype='i4')

# ideal square grid test case
#x = np.linspace(-0.5, 2.5, 5)
#y = np.linspace(-0.5, 2.5, 5)
#xx, yy = np.meshgrid(x, y)
#xy = np.c_[xx.ravel(), yy.ravel()]
#pts = xy

if not args.noplot:
    plt.figure(0)
    plt.hist(radii,bins=len(radii)/4)
    plt.gca().set_title('Radius')
    plt.xlabel('radius (nm)')
    plt.ylabel('Count')
    plt.savefig(output_data_path+'/'+filename+'_radius_hist.png', bbox_inches='tight')

vor = Voronoi(pts)

bond_order = args.order

if not bond_order > 0:
    raise ValueError('order parameter should be > 0')

# minkowski_structure_metric returns a list with region_index, metric
msm = minkowski(vor.vertices,vor.regions,bond_order,(im.shape[1],im.shape[0]))

if np.any(np.isnan(np.asarray(msm)[:,1])):
    raise ValueError('nan found in structure metric array')

if not args.noplot:
    plt.figure(1)
    plt.subplot(111)
    implot = plt.imshow(im_original)
    implot.set_cmap('gray')

    symmetry_colormap = plt.get_cmap('RdBu_r')

    plot_symmetry(im,msm,bond_order,symmetry_colormap,mask, args.outline, args.edge)


# save the metrics to a file
header_string =     str(bond_order)+'-fold metric (Psi'+str(bond_order)+')\n'
header_string +=    'References: Mickel, W., J. Chem. Phys. (2013), Escobedo, F., Soft Matter (2011)\n'
header_string +=    'length: '+str(len(msm))+'\n'
header_string +=    'region_index\tPsi'+str(bond_order)
np.savetxt(output_data_path+'/'+filename+'_Psi'+str(bond_order)+'_data.txt',msm,fmt=('%u','%.3f'),delimiter='\t',header=header_string)

if args.montecarlo:
    # Calculate the "bond strengths"
    binary_im,_ = make_binary_image(im,background,2*pixels_per_nm,adaptive=1)
    nn_dist_list = []
    bond_list = []

    # graphs for random access, Monte Carlo?
    bond_graph = sparse.lil_matrix((len(pts),len(pts)),dtype=np.double)
    distance_x_graph = sparse.lil_matrix((len(pts),len(pts)),dtype=np.double)
    distance_y_graph = sparse.lil_matrix((len(pts),len(pts)),dtype=np.double)

    # keep track of boundary particles
    # naively there is a maximum of len(pts) particles that could be on the boundary
    # since we don't want to allocate all of them or append a site more than once to a list, use a sparse matrix
    boundary_sites = sparse.lil_matrix((len(pts),1),dtype=np.int8)

    # for saving edge data to file
    edges = []

    # lists for plotting
    bond_list_filtered = []
    nn_dist_list_filtered= []
    line_segments = []
    line_segments_filtered = []

    # to draw the bonds between points
    for ridge_vert_indices,input_pair_indices in zip(vor.ridge_vertices,vor.ridge_points):
    
        if np.all(np.not_equal(ridge_vert_indices,-1)):
    
            # get the region enclosing this point
            vertex1 = vor.vertices[ridge_vert_indices[0]]
            vertex2 = vor.vertices[ridge_vert_indices[1]]
        
            x_vals = zip(vertex1,vertex2)[0]
            y_vals = zip(vertex1,vertex2)[1]

            input_pt1 = pts[input_pair_indices[0]]
            input_pt2 = pts[input_pair_indices[1]]
        
            # check if the voronoi region vertices are within the image bounds
            if np.all((np.greater(x_vals,0),np.greater(y_vals,0),np.less(x_vals,im.shape[1]),np.less(y_vals,im.shape[0]))):

                # get the nearest neighbor distance
                # the directional x and y distance to move from pt1 to pt2
                nn_x_dist = (input_pt2[0]-input_pt1[0])/pixels_per_nm
                nn_y_dist = (input_pt2[1]-input_pt1[1])/pixels_per_nm

                # if the interparticle distance computes to zero
                # it is because of the resolution of the image
                # therefore set the distance to the uncertainty of the measurement
                # this avoids dropping of zero-valued elements later in sparse matrix operations
                # this should only be an issue for a few particles with the lowest magnification images
                if nn_x_dist == 0.0:
                    nn_x_dist = 1.0/pixels_per_nm

                if nn_y_dist == 0.0:
                    nn_y_dist = 1.0/pixels_per_nm

                nn_distance = np.sqrt(nn_x_dist**2 + nn_y_dist**2)

                # distance_..._graph[i,j] is the distance to move from point i to point j
                distance_x_graph[input_pair_indices[0],input_pair_indices[1]] = nn_x_dist
                distance_x_graph[input_pair_indices[1],input_pair_indices[0]] = -nn_x_dist
                distance_y_graph[input_pair_indices[0],input_pair_indices[1]] = nn_y_dist
                distance_y_graph[input_pair_indices[1],input_pair_indices[0]] = -nn_y_dist
                nn_dist_list.append(nn_distance)

                # get the bond width
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

                bond_width = np.sum(binary_im[y_range,x_range])/pixels_per_nm
                bond_graph[input_pair_indices[0],input_pair_indices[1]] = bond_width
                bond_graph[input_pair_indices[1],input_pair_indices[0]] = bond_width
                bond_list.append(bond_width)

                # make the line segments for plotting bonds, neighbor distances, whatever
                line_segments.append(np.asarray([pts[input_pair_indices[0]],pts[input_pair_indices[1]]]))

                edges.append([input_pair_indices[0],input_pair_indices[1],nn_distance,bond_width])

                if not bond_width == 0:
                    bond_list_filtered.append(bond_width)
                    nn_dist_list_filtered.append(nn_distance)
                    line_segments_filtered.append(np.asarray([pts[input_pair_indices[0]],pts[input_pair_indices[1]]]))
    
            else:
                # at least one ridge vertex is off the image
                # these two input points are boundary sites
                boundary_sites[input_pair_indices,0] = np.int(1)

    if not args.noplot:

        plt.figure(3)
        plot_bonds(im,line_segments_filtered,bond_list_filtered)
        plt.savefig(output_data_path+'/'+filename+'_bond_map.pdf',bbox_inches='tight')

        plt.figure(4)
        plot_bonds(binary_im,line_segments,bond_list)
        plt.savefig(output_data_path+'/'+filename+'_bond_map_binary.pdf',bbox_inches='tight')

        # make a map of the nn distances
        plt.figure(5)
        plot_nn_distance(im_original,line_segments_filtered,nn_dist_list_filtered)
        plt.savefig(output_data_path+'/'+filename+'_nn_dist_map.pdf',bbox_inches='tight')

        # plot a histogram of the "bond strengths"
        plt.figure(6)
        plt.hist(bond_list_filtered,bins=len(bond_list_filtered)/4)
        plt.ylabel('Count')
        plt.xlabel('Width (nm)')
        plt.gca().set_title('Bond Widths (filtered)')
        plt.savefig(output_data_path+'/'+filename+'_bond_hist.png', bbox_inches='tight')

        # plot a histogram of the nearest neighbor distances
        plt.figure(7)
        plt.hist(nn_dist_list,bins=len(nn_dist_list)/4)
        plt.ylabel('Count')
        plt.xlabel('Neighbor Distance (nm)')
        plt.savefig(output_data_path+'/'+filename+'_nn_dist_hist.png', bbox_inches='tight')

    # save edge data to file
    header_string =     'pt1 and pt2 are the indices of the points between which the distance and bond width are given\n'
    header_string +=    'total edges: '+str(len(edges))+'\n'
    header_string +=    'pt1\tpt2\tdistance (nm)\tbond width (nm)'
    np.savetxt(output_data_path+'/'+filename+'_edges.txt',np.asarray(edges),fmt=('%u','%u','%.3f','%.3f'),delimiter='\t',header=header_string)

    # show it all
    #plt.show()

    # save the graphs to files to use in Monte Carlo
    bond_graph_csr = bond_graph.tocsr()
    distance_x_graph_csr = distance_x_graph.tocsr()
    distance_y_graph_csr = distance_y_graph.tocsr()
    boundary_sites_csc = boundary_sites.tocsc()
    radii = np.asarray(radii.reshape((-1,)),dtype=np.double)
    np.savez(output_data_path+'/'+filename+'_bond_graph',bond_graph=bond_graph_csr)
    np.savez(output_data_path+'/'+filename+'_x_distance_graph',data=distance_x_graph_csr.data,indices=distance_x_graph_csr.indices,indptr=distance_x_graph_csr.indptr)
    np.savez(output_data_path+'/'+filename+'_y_distance_graph',data=distance_y_graph_csr.data,indices=distance_y_graph_csr.indices,indptr=distance_y_graph_csr.indptr)
    np.savez(output_data_path+'/'+filename+'_boundary_graph',data=boundary_sites_csc.data,indices=boundary_sites_csc.indices,indptr=boundary_sites_csc.indptr)
    np.savez(output_data_path+'/'+filename+'_site_data',radii=radii,pts=pts,pixels_per_nm=np.asarray(pixels_per_nm),box_x_size=np.asarray(im.shape[1]/pixels_per_nm))