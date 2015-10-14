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

# for fitting the radius distribution
from scipy.optimize import curve_fit
from scipy.stats import linregress

# for disabling annoying warnings
import warnings

# plotting
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable, Size

# Voronoi
from scipy.spatial import Voronoi

# for finding particle centers, diameters, etc.
from skimage.measure import regionprops
from skimage.filters import threshold_otsu,threshold_adaptive, threshold_isodata
from skimage.morphology import watershed, remove_small_objects, binary_erosion, binary_dilation, binary_closing, binary_opening
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.draw import circle
from skimage.morphology import disk

# for command line interface
import argparse
from skimage import io as skimio
from os import path
from os import walk

# regular expressions for parsing scalebar filenames
import re

# for matching the scale bar
from skimage.feature import match_template

# minkowski structure metric function written and compiled with Cython
from minkowski_metric import minkowski
#import pyximport; pyximport.install()
#import minkowski_metric

# globals
DEBUG_OUTPUT = False


def make_binary_image(im, white_background, min_feature_size, adaptive):

    image = im.astype('float32')

    if white_background:
        image = np.abs(image-np.max(image))

    # get rid of large background patches before local thresholding
    # do a global threshold
    thresh = threshold_isodata(image)
    binary = image > thresh
    
    # get rid of speckle
    # this is not good for very small particles
    binary_closing(binary, selem=disk(min_feature_size), out=binary)

    # make a map of the distance to a particle to find large background patches
    # this is a distance map of the inverse binary image
    distance = ndimage.distance_transform_edt(1-binary)
    
    # dilate the distance map to expand small voids
    #distance = ndimage.grey_dilation(distance,size=2*min_feature_size)
    
    # do a global threshold on the distance map to select the biggest objects
    # larger than a minimum size to prevent masking of particles in images with no empty patches
    dist_thresh = threshold_isodata(distance)
    mask = distance < dist_thresh  
    
    # remove areas of background smaller than a certain size in the mask
    # this fills in small pieces of mask between particles where the voronoi
    # vertices will end up
    # this gets rid of junk in the gap areas
    binary_opening(mask, selem=disk(min_feature_size), out=mask)
    
    binary = binary * mask

    # get rid of speckle
    binary = remove_small_objects(binary,min_size=max(min_feature_size,2))
    
    # 3 iterations is better for large particles with low contrast
    binary = ndimage.binary_closing(binary,iterations=1)

    if DEBUG_OUTPUT:
        fig, ax = plt.subplots(2, 2)
        ax[0,0].imshow(binary)
        ax[0,0].set_title('Global threshold')
        
        ax[0,1].imshow(distance)
        ax[0,1].set_title('Distance map')
        
        ax[1,0].imshow(mask)
        ax[1,0].set_title('Modified Distance Based Mask')
        
        ax[1,1].imshow(binary)
        ax[1,1].set_title('Masked Binarized Image')
        plt.show()

    return binary, mask
    
def adaptive_binary_image(im, white_background, min_feature_size, std_dev, mask):

    image = im.astype('float32')

    if white_background:
        image = np.abs(image-np.max(image))
    
    # the block size should be large enough to include 4 to 9 particles
    local_size = 40*min_feature_size
    binary = threshold_adaptive(image,block_size=local_size)

    # close any small holes in the particles
    # 3 iterations is better for large particles with low contrast
    #binary = ndimage.binary_closing(binary,iterations=1)
    binary_closing(binary, selem=disk(int(max((0.414*(min_feature_size-std_dev)),2)/2.0)), out=binary)
    
    # remove speckle from background areas in the binary image
    binary = binary * mask #binary_erosion(mask, selem=disk(int(min_feature_size)))
    binary_opening(binary, selem=disk(int(max((min_feature_size-3.0*std_dev),2)/2.0)), out=binary)

    # make a distance map of the inverted image
    distance = ndimage.distance_transform_edt((1-binary))

    # do a global threshold on the distance map to select the biggest objects
    # larger than a minimum size to prevent masking of particles in images with no empty patches
    dist_thresh = threshold_isodata(distance)
    new_mask = distance < dist_thresh
    
    # remove areas of background in the mask smaller than a certain size
    # this fills in small pieces of mask between particles where the voronoi
    # vertices will end up
    dilation_size = max(int(1),int(min_feature_size))
    new_mask = binary_closing(new_mask, selem=np.ones((dilation_size,dilation_size)))
    
    if DEBUG_OUTPUT:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(binary)
        ax[0].set_title('Adaptive thresh')
        
        ax[1].imshow(binary)
        ax[1].set_title('masked adaptive thresh')
        
        ax[2].imshow(new_mask)
        ax[2].set_title('adaptive mask')
        plt.show()
    
    return binary, new_mask

def morphological_threshold(im, white_background, mean_radius, min_feature_size, mask):

    if DEBUG_OUTPUT:
        print('Morphological threshold mean radius (px): '+str(mean_radius))
    
    im_mod = np.array(im, dtype='float64')

    if white_background:
        im_mod = np.abs(im_mod-np.max(im_mod))
        
    # subtract the mean before running match_template
    # not sure this works quite right
    im_mod = im_mod - np.mean(im_mod)

    # set large areas of background to zero using the mask
    im_mod = im_mod * mask
    
    if DEBUG_OUTPUT:
        print(np.max(im_mod))
        print(np.min(im_mod))        

    #topped_im = tophat(im_mod,disk(mean_radius),mask=mask)
    template_matrix = np.pad(disk(int(mean_radius/4)),pad_width=int(mean_radius), mode='constant',constant_values=0)
    
    matched_im = match_template(im_mod,template=template_matrix,pad_input=True)
    
    if DEBUG_OUTPUT:
        print(np.max(matched_im))
        print(np.min(matched_im))

    thresh = threshold_isodata(matched_im)
    matched_im_bin = matched_im > thresh
    
    matched_im_bin *= mask

    distance = ndimage.distance_transform_edt(matched_im_bin)

    # dilate the distance map to merge close peaks (merges multiple peaks in one particle)
    distance = ndimage.grey_dilation(distance,size=min_feature_size)

    local_maxi = peak_local_max(distance,indices=False,min_distance=min_feature_size)
    markers = ndimage.label(local_maxi)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels_th = watershed(-distance, markers, mask=matched_im_bin)

    if DEBUG_OUTPUT:
        f, ax = plt.subplots(2, 2)
        ax[0,0].imshow(matched_im)
        ax[0,0].set_title('template match')
        
        ax[0,1].imshow(matched_im_bin)
        ax[0,1].set_title('bin')
        
        ax[1,0].imshow(distance)
        ax[1,0].set_title('distance')
        
        ax[1,1].imshow(labels_th,cmap=plt.cm.prism)
        ax[1,1].set_title('labels')
        plt.show()

    return labels_th, matched_im_bin


def get_particle_centers(im, white_background, pixels_per_nm, morph):

    if pixels_per_nm == 0:
        # default value
        pixels_per_nm = 1.0

    # minimum size object to look for
    min_feature_size = int(3) #int(3*pixels_per_nm)

    global_binary, mask = make_binary_image(im, white_background, min_feature_size, adaptive=0)

    # create a distance map to find the particle centers
    # as the points with maximal distance to the background
    distance = ndimage.distance_transform_edt(global_binary)

    # dilate the distance map to merge close peaks (merges multiple peaks in one particle)
    distance = ndimage.grey_dilation(distance,size=min_feature_size)

    # min_distance=5 for large particles
    local_maxi = peak_local_max(distance,indices=False,min_distance=min_feature_size)
    markers = ndimage.label(local_maxi)[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels = watershed(-distance, markers, connectivity=None, offset=None, mask=global_binary)

    # get the particle radii
    global_regions = regionprops(labels)
    gobal_radii = []
    
    for props in global_regions:
        # define the radius as half the average of the major and minor diameters
        gobal_radii.append(((props.minor_axis_length+props.major_axis_length)/4)/pixels_per_nm)

    # minimum size object to look for
    global_mean_radius = np.mean(gobal_radii)*pixels_per_nm
    global_radii_sd = np.std(gobal_radii)*pixels_per_nm
    
    print('Mean radius global threshold (px): %(rad).2f SD: %(sd).2g' % {'rad':global_mean_radius, 'sd':global_radii_sd})
    
    feature_size = int(max(global_mean_radius, min_feature_size))
    std_dev = int(max(global_radii_sd, min_feature_size))
    
    adaptive_binary, adaptive_mask = adaptive_binary_image(im, white_background, feature_size, std_dev, mask)

    # create a distance map to find the particle centers
    # as the points with maximal distance to the background
    distance = ndimage.distance_transform_edt(adaptive_binary)

    # dilate the distance map to merge close peaks (merges multiple peaks in one particle)
    distance = ndimage.grey_dilation(distance,size=feature_size)

    local_maxi = peak_local_max(distance,indices=False,min_distance=feature_size)
    markers = ndimage.label(local_maxi)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels = watershed(-distance, markers, connectivity=None, offset=None, mask=adaptive_binary)

    adaptive_regions = regionprops(labels)
    adaptive_radii = []
    
    for props in adaptive_regions:
        # define the radius as half the average of the major and minor diameters
        adaptive_radii.append(((props.minor_axis_length+props.major_axis_length)/4)/pixels_per_nm)

    # minimum size object to look for
    adaptive_radii_sd = np.std(adaptive_radii)
    adaptive_mean_radius = np.mean(adaptive_radii)
    print('Mean radius adaptive threshold (px): %(rad).2f SD: %(sd).2g' % {'rad':adaptive_mean_radius*pixels_per_nm, 'sd':adaptive_radii_sd*pixels_per_nm})
    
    
    # morphological thresholding
    if morph:
        print('Using morphological threshold')
        labels, morph_binary = morphological_threshold(im, white_background, int(adaptive_mean_radius*pixels_per_nm), int(adaptive_mean_radius*pixels_per_nm)/2, adaptive_mask)
        regions = regionprops(labels)
        binary = morph_binary
    elif global_radii_sd/global_mean_radius < adaptive_radii_sd/adaptive_mean_radius:
        print('Using global threshold')
        regions = global_regions
        binary = global_binary
    else:
        print('Using adaptive threshold')
        regions = adaptive_regions
        binary = adaptive_binary

#     if DEBUG_OUTPUT:
#         fig, ax = plt.subplots(2, 2)
#         ax[0,0].imshow(binary)
#         ax[0,0].set_title('Binarized Image')
#         
#         ax[0,1].imshow(distance)
#         ax[0,1].set_title('Distance Map')
# 
#         ax[1,0].imshow(markers,cmap=plt.cm.spectral,alpha=0.5)
#         ax[1,0].set_title('Region Markers')
#         
#         ax[1,1].imshow(labels,cmap=plt.cm.prism)
#         ax[1,1].set_title('Labels')
#         plt.show()

    # get the particle centroids again, this time with better thresholding
    pts = []
    radii = []

    for props in regions:
        # centroid is [row, col] we want [col, row] aka [X,Y]
        # so reverse the order
        pts.append(props.centroid[::-1])

        # define the radius as half the average of the major and minor diameters
        radii.append(((props.minor_axis_length+props.major_axis_length)/4)/pixels_per_nm)

    if morph:
        print('Mean radius morphological threshold (px): %(rad).2f SD: %(sd).2g' % {'rad':np.mean(radii)*pixels_per_nm, 'sd':np.std(radii)*pixels_per_nm})
        
        
    return np.asarray(pts,dtype=np.double), np.asarray(radii,dtype=np.double).reshape((-1,1)), adaptive_mask, binary

def get_image_scale(im):

    scale = 0.0
    topleft = (0,0)
    bottomright = (0,0)

    input_path = path.normpath('../resources/scalebars')

    # load all files in scalebars directory and sub-directories
    scalebar_filesnames = []
    for (dirpath, dirnames, filesnames) in walk(input_path):
        scalebar_filesnames.extend(filesnames)

    scale_bars = []
    ipart = 1
    fpart = 0.0
    for filename in scalebar_filesnames:
        reg_result = re.search('Scale_(\d*)p*(\d*)',filename)

        if reg_result.group(1):
            ipart = int(reg_result.group(1))

        if reg_result.group(2):
            fpart = float('0.'+reg_result.group(2))

        px_per_nm = ipart+fpart

        # images of scale bars to match with the input image
        # second element is the scale in units of pixels/nm
        scale_bars.append([skimio.imread(input_path+'/'+filename,as_grey=True,plugin='matplotlib'),px_per_nm])

    match_score = []

    for scale_bar,pixels_per_nm in scale_bars:

        result = match_template(im,template=scale_bar)

        ij = np.unravel_index(np.argmax(result), result.shape)
        row, col = ij

        match_score.append(result[row][col])

        # the match score should be about 0.999999
        if result[row][col] > 0.99:

            topleft = (row,col)
            bottomright = (row+scale_bar.shape[0],col+scale_bar.shape[1])

            scale = pixels_per_nm

            print('Scale: '+str(scale)+' pixels/nm, Score: '+str(np.max(match_score)))
            break # we found it, stop looking

    if np.max(match_score) < 0.99:
        print('!!!!!!!!!!!!!!!!!!!! No scale bar found !!!!!!!!!!!!!!!!!!!!!')

    return [np.double(scale),(topleft,bottomright)]

def create_custom_colormap():

    # get a color map for mapping metric values to colors of some color scale
    value_rgb_pairs = []
    rgb_array = np.asarray([[0,0,0],[255,0,0],[255,50,34],[255,109,59],[255,177,102],[255,220,125],[255,245,160],[255,245,192],[255,255,255],[212,251,255],[160,253,255],[120,226,255],[81,177,255],[55,127,255],[31,81,255],[0,13,255]],dtype='f4')
    rgb_array /= 255
    rgb_list_norm = []

    for value, color in zip(np.linspace(0,1,16),rgb_array):
        value_rgb_pairs.append((value,color))

    return LinearSegmentedColormap.from_list(name="custom", colors=value_rgb_pairs, N=16)

def plot_symmetry(im, msm, vor, bond_order, symmetry_colormap, mask, no_fill, map_edge_particles):
    
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
            if no_fill:
                cell_patches.append(Polygon(verts,closed=True,facecolor='none',edgecolor='r'))
            else:
                cell_patches.append(Polygon(verts,closed=True,edgecolor='none'))
        
            metric_list.append(metric)
            
    if no_fill:
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

    if not no_fill:
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
    
def plot_particle_outlines(im, pts, radii, pixels_per_nm):
    
    cell_patches = []
    for center,radius in zip(pts,radii):
        cell_patches.append(Circle(center,radius*pixels_per_nm,facecolor='none',edgecolor='r'))
            
    pc = PatchCollection(cell_patches,match_original=True,alpha=1)
        
    plt.gca().add_collection(pc)

    # set the limits for the plot
    # set the x axis range
    plt.gca().set_xlim(0, im.shape[1])

    # set the y-axis range and flip the y-axis
    plt.gca().set_ylim(im.shape[0], 0)

    # save this plot to a file
    plt.gca().set_axis_off()
    
    plt.savefig(output_data_path+'/'+filename+'_particle_map.png',bbox_inches='tight',dpi=300)

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

def gaussian(x, amp, mean, std):
    return amp/(np.sqrt(2.0*np.pi)*std)*np.exp(-(x-mean)**2/(2.0*std**2))

def square(x, amp, center, width):
    out = np.zeros(len(x))
    peaks = np.linspace(center-width,center+width,10)
    for peak in peaks:
        out += gaussian(x,amp*np.sqrt(2.0*np.pi)*width/10,peak,width/10)

    return out



parser = argparse.ArgumentParser()
parser.add_argument('-b','--black', help='black background', action='store_true')
parser.add_argument('-np','--noplot', help='do not plot data', action='store_true')
parser.add_argument('-nd', '--nodata', help='do not output data files (images only)', action='store_true')
parser.add_argument('-m','--morph', help='use morphological filtering', action='store_true')
parser.add_argument('-o','--outline', help='draw particle outlines on the image', action='store_true')
parser.add_argument('-v','--voronoi', help='draw the voronoi diagram on the image', action='store_true')
parser.add_argument('-e','--edge', help='plot the symmetry metric of particles at superlattice edge', action='store_true')
parser.add_argument('-mc','--montecarlo', help='caclulate NN dist, Bond width, etc.', action='store_true')
parser.add_argument('-d','--debug', help='turn on debugging output', action='store_true')
parser.add_argument('order',nargs='?', help='order of the structure metric', type=int, default=0)
parser.add_argument('img_file',help='image file to analyze')
parser.add_argument('pts_file',nargs='?',help='XY point data file',default='')
parser.add_argument('pix_per_nm',nargs='?',help='scale in pixels per nm',default=0.0,type=float)
args = parser.parse_args()

im = skimio.imread(args.img_file,as_grey=True,plugin='matplotlib')
im_original = np.empty_like(im)
np.copyto(im_original,im)

if args.debug:
    DEBUG_OUTPUT = True
    print('Debugging output turned on')

if DEBUG_OUTPUT:
    implot = plt.imshow(im)
    implot.set_cmap('gray')
    plt.gca().set_title('Imported Image')
    plt.show()

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
        pixels_per_nm = 1

else:
    print("User specified scale: "+str(pixels_per_nm)+" pixels per nm")

if len(args.pts_file) == 0:
    # find the centroid of each particle in the image
    pts, radii, mask, binary = get_particle_centers(im,background,pixels_per_nm,args.morph)

    assert len(pts) == len(radii)
    
    # fit a normal distribution function to the data at the mean +/- 3 std.
    radii_hist, radii_bins = np.histogram(radii, bins=len(radii)/4, range=(np.mean(radii)-3.0*np.std(radii),np.mean(radii)+3.0*np.std(radii)))
    # trapezoidal approximation
    radii_bins += (radii_bins[1]-radii_bins[0])/2
    popt, pcov = curve_fit(gaussian, radii_bins[0:(len(radii_bins)-1)], radii_hist, p0=(np.max(radii_hist),np.mean(radii),np.std(radii)))
    fit_amp, fit_mean, fit_std = popt
    radii_mean = np.mean(radii)
    radii_std = np.std(radii)

    particle_data = np.hstack((pts,radii))

    # save the input points to a file
    if not args.nodata:
        header_string = 'Particle centroids in pixel units\n'
        header_string += 'Particle radii - the average of the major and minor radii of an ellipse fit to the particle area\n'
        header_string += 'total particles: '+str(len(pts))+'\n'
        header_string += 'mean radius (raw data): %(rad_nm).2g Std.Dev. %(sd_nm).2g (nm), %(rad_px).2g Std.Dev. %(sd_px).2g (pixels)\n' % {'rad_nm':radii_mean, 'sd_nm':radii_std, 'rad_px':radii_mean*pixels_per_nm, 'sd_px':radii_std*pixels_per_nm}
        header_string += 'mean radius (gaussian fit): %(rad_nm).2g Std.Dev. %(sd_nm).2g (nm), %(rad_px).2g Std.Dev. %(sd_px).2g (pixels)\n' % {'rad_nm':fit_mean, 'sd_nm':fit_std, 'rad_px':fit_mean*pixels_per_nm, 'sd_px':fit_std*pixels_per_nm}
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
    ax = plt.figure(0)
    plt.hist(2.0*radii,range=(2.0*(max(0,np.mean(radii)-3.0*np.std(radii))),2.0*(np.mean(radii)+3.0*np.std(radii))),bins=len(radii)/4)
    fit_hist = gaussian(2.0*radii_bins, 2.0*fit_amp, 2.0*fit_mean, 2.0*fit_std)
    plt.plot(2.0*radii_bins, fit_hist, 'r-', linewidth=3)
    label_string = '%(count)i particles, Diameter: %(mean).3g (nm), $\sigma$: %(sd).2g (nm), %(percent).2g%%' % {'count':len(radii), 'mean':2.0*fit_mean, 'sd':2.0*fit_std, 'percent':100.0*fit_std/fit_mean }
    plt.gca().set_title(label_string)
    plt.xlabel('Diameter (nm)')
    plt.ylabel('Count')
    plt.savefig(output_data_path+'/'+filename+'_diameter_hist.png', bbox_inches='tight')

vor = Voronoi(pts)

bond_order = args.order

if bond_order < 0:
    raise ValueError('order parameter should be > 0')

if bond_order > 0:
    # minkowski_structure_metric returns a list with region_index, metric
    msm = minkowski(vor.vertices,vor.regions,bond_order,(im.shape[1],im.shape[0]))

    if np.any(np.isnan(np.asarray(msm)[:,1])):
        raise ValueError('nan found in structure metric array')
else:
    print("No bond order given, only the particle locations will be found")

if not args.noplot:

    if args.outline:
        plt.figure(1)
        plt.subplot(111)
        implot = plt.imshow(im_original)
        implot.set_cmap('gray')
        plot_particle_outlines(im, pts, radii, pixels_per_nm)
    
    if bond_order > 0:
        plt.figure(1)
        plt.clf()
        plt.subplot(111)
        implot = plt.imshow(im_original)
        implot.set_cmap('gray')
        symmetry_colormap = plt.get_cmap('RdBu_r')
        plot_symmetry(im, msm, vor, bond_order, symmetry_colormap, mask, args.voronoi, args.edge)


# save the metrics to a file
if bond_order > 0:
    if not args.nodata:
        header_string =     str(bond_order)+'-fold metric (Psi'+str(bond_order)+')\n'
        header_string +=    'References: Mickel, W., J. Chem. Phys. (2013), Escobedo, F., Soft Matter (2011)\n'
        header_string +=    'length: '+str(len(msm))+'\n'
        header_string +=    'region_index\tPsi'+str(bond_order)
        np.savetxt(output_data_path+'/'+filename+'_Psi'+str(bond_order)+'_data.txt',msm,fmt=('%u','%.3f'),delimiter='\t',header=header_string)

# the rest of this should be moved to another file
if bond_order > 0:
    if args.montecarlo:

        # make sure that im is not altered
        if background:
            # invert a light background image
            im_greyscale = np.abs(im_original-np.max(im_original))
        else:
            im_greyscale = im_original

        max_image_intensity = np.max(im_original)

        # Calculate the "bond strengths"
        binary_im = binary # use the binary from get_particles above instead of: make_binary_image(im,background,2*pixels_per_nm,adaptive=1)
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
        bond_line_segments = []
        bond_line_segments_filtered = []
        bond_width_segments = []

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
                    x_range = np.linspace(vertex1[0],vertex2[0],num=range_len)
                    y_range = np.linspace(vertex1[1],vertex2[1],num=range_len)

                    x_range = x_range.astype(int)
                    y_range = y_range.astype(int)

                    # the old, fast, simple way:
                    # need to do some trig here...
                    #bond_width = np.sum(binary_im[y_range,x_range])/pixels_per_nm

                    bond_width = 0

                    # the new, "improved" way, instead of counting the number of non-background pixels
                    # along the facet line in a binary image, fit some profile to the greyscale image

                    # find the length of the facet in pixel units
                    facet_length = np.sqrt(float(facet_y_dist)**2 + float(facet_x_dist)**2)

                    # try extending the line in both directions a little to try to find some background pixels
                    extension_factor = 0.05 # extend this much in both directions
                    phi = np.arctan(float(facet_y_dist)/float(facet_x_dist))

                    x_facet_start = int(x_range[0] + np.sign(x_range[0]-x_range[-1]) * np.cos(phi) * (facet_length * extension_factor))
                    if x_facet_start < 0:
                        x_facet_start = 0
                    elif x_facet_start > im_greyscale.shape[1]-1:
                        x_facet_start = im_greyscale.shape[1]-1

                    y_facet_start = int(y_range[0] + np.sign(y_range[0]-y_range[-1]) * np.sin(phi) * (facet_length * extension_factor))
                    if y_facet_start < 0:
                        y_facet_start = 0
                    elif y_facet_start > im_greyscale.shape[0]-1:
                        y_facet_start = im_greyscale.shape[0]-1

                    x_facet_end = int(x_range[-1] + np.sign(x_range[-1]-x_range[0]) * np.cos(phi) * (facet_length * extension_factor))
                    if x_facet_end < 0:
                        x_facet_end = 0
                    elif x_facet_end > im_greyscale.shape[1]-1:
                        x_facet_end = im_greyscale.shape[1]-1

                    y_facet_end = int(y_range[-1] + np.sign(y_range[-1]-y_range[0]) * np.sin(phi) * (facet_length * extension_factor))
                    if y_facet_end < 0:
                        y_facet_end = 0
                    elif y_facet_end > im_greyscale.shape[0]-1:
                        y_facet_end = im_greyscale.shape[0]-1

                    range_len = max(np.abs(x_facet_start-x_facet_end), np.abs(y_facet_start-y_facet_end))

                    # overwrite x_range, y_range
                    x_range = np.linspace(x_facet_start,x_facet_end,num=range_len, dtype='int')
                    y_range = np.linspace(y_facet_start,y_facet_end,num=range_len, dtype='int')

                    # overwrite the length now that we've extended it
                    facet_length = facet_length + 2.0 * extension_factor * facet_length


                    # don't try to fit a flat line or a very small line
                    # need at least 3 points to fit a function with 3 parameters
                    if int(range_len) > 9:

                        # now we get the values to fit
                        # average some values to the left and right of the facet line
                        averaging_steps = np.linspace(-0.1 * facet_length, 0.1 * facet_length, num=2*0.1*facet_length+1)
                        facet_pixel_values = np.zeros(range_len)
                        theta = np.arctan2( (y_range[-1]-y_range[0]) , (x_range[-1] - x_range[0]) )

                        bond_area = np.empty((0,2))

                        for step in averaging_steps:
                            # move perpendicular to the facet line
                            # a vertical line will only shift in x
                            # a horizontal line will only shift in y
                            x_shift = int(step * np.sin(theta))
                            y_shift = -int(step * np.cos(theta))

                            # for plotting in debug code
                            bond_area = np.vstack((bond_area,np.vstack((x_range+x_shift, y_range+y_shift)).transpose()))

                            facet_pixel_values += im_greyscale[y_range + y_shift, x_range + x_shift]

                        facet_pixel_values = facet_pixel_values/len(averaging_steps)

                        # normalize the values to help the fitting
                        facet_pixel_values = (facet_pixel_values - np.min(facet_pixel_values))/np.max(facet_pixel_values)

                        # fit a square function to that vector of intensity values
                        # square(x, amp, center, width)
                        # gaussian(x, amp, mean, std)
                        # width is the extent of the square function on either side of center
                        width_guess = facet_length/4.0
                        center_guess = facet_length/2.0
                        amp_guess = facet_pixel_values.mean()
                        params = (amp_guess, center_guess, width_guess)

                        # debug to find problematic values
                        # print(params,len(facet_pixel_values))

                        fit_func = square
                        # fit_func = gaussian

                        try:
                            fit_window = np.linspace(0,facet_length, num=len(facet_pixel_values))
                            popt, pcov = curve_fit(fit_func, fit_window, facet_pixel_values, p0=params)

                            if fit_func == square:
                                bond_width = popt[2]*2.0/pixels_per_nm
                                bond_center = popt[1]
                                fit_curve = square(fit_window, popt[0], popt[1], popt[2])

                            elif fit_func == gaussian:
                                # FWHM from gaussian
                                # might need to modify this because the background isn't always near 0 if the image is not binary
                                bond_width = 2.0*np.sqrt(2.0*np.log(2))*popt[2]/pixels_per_nm
                                bond_center = popt[1]
                                fit_curve = gaussian(fit_window, popt[0], popt[1], popt[2])


                            # do a linear regression to find the R**2 value for the fit
                            # bad fits will be thrown out
                            s, i, r, p, se = linregress(facet_pixel_values,fit_curve)


                            # for plotting the bond
                            x_bond_start = int(x_range[0] + np.sign(x_range[-1]-x_range[0]) * np.cos(phi) * (bond_center - bond_width/2*pixels_per_nm))
                            y_bond_start = int(y_range[0] + np.sign(y_range[-1]-y_range[0]) * np.sin(phi) * (bond_center - bond_width/2*pixels_per_nm))

                            x_bond_end = int(x_range[0] + np.sign(x_range[-1]-x_range[0]) * np.cos(phi) * (bond_center + bond_width/2*pixels_per_nm))
                            y_bond_end = int(y_range[0] + np.sign(y_range[-1]-y_range[0]) * np.sin(phi) * (bond_center + bond_width/2*pixels_per_nm))


                            if DEBUG_OUTPUT:
                                # use this to look at the fitting of bond width
                                # plot the pixel values and the fit curve
                                plt.subplot(1, 2, 1)
                                plt.scatter(np.arange(len(x_range)),facet_pixel_values)
                                plt.plot(np.arange(len(x_range)),fit_curve,'r-')

                                # plot the image where the bond is, overlay the facet line and the bond width
                                plt.subplot(1, 2, 2)
                                plt.imshow(im_greyscale, cmap='gray', interpolation='none')
                                plt.plot(x_range,y_range,'b-')
                                plt.scatter(bond_area[:,0], bond_area[:,1], edgecolor='none', alpha=0.1)

                                # plot a line showing the fit bond_width
                                bond_len = int(max(np.abs(x_bond_start-x_bond_end),np.abs(y_bond_start-y_bond_end)))
                                bond_x = np.linspace(x_bond_start, x_bond_end, num=bond_len)
                                bond_y = np.linspace(y_bond_start, y_bond_end, num=bond_len)

                                plt.plot(bond_x, bond_y, 'r-')

                                # just show the part of the image where the bond is
                                plt.gca().set_xlim(np.min(x_range)-range_len, np.max(x_range)+range_len)
                                plt.gca().set_ylim(np.min(y_range)-range_len, np.max(y_range)+range_len)
                                plt.gca().set_title('Facet Length: %(facet)i R^2: %(rs).3f\n Bond Width: %(bond)i (pixels), Center: %(center)i' % {'facet':facet_length, 'rs':r**2, 'bond':bond_width*pixels_per_nm, 'center':bond_center})
                                plt.show()

                            # sanity checks
                            if bond_width*pixels_per_nm > facet_length:
                                # we'll allow the width to be a little more than
                                # the fitting window, because the window doesn't
                                # always span the entire bond, but the fit can sometimes
                                # be a good estimate based on the curvature of the bond
                                bond_width = 0

                            if bond_center < 0 or bond_center > facet_length:
                                # if the center of the bond is outside of the fitting window,
                                # it's probably a bad fit, throw it out
                                bond_width = 0

                            if r**2 < 0.5:
                                # bad fit
                                bond_width = 0

                            # save the bond locations as line segments for output in an image
                            if bond_width:
                                bond_width_segments.append(np.asarray([(x_bond_start,y_bond_start),(x_bond_end,y_bond_end)]))

                        except RuntimeError:
                            # this happens if the curve_fit method doesn't converge
                            # the bond width will be zero in this case
                            pass


                    bond_graph[input_pair_indices[0],input_pair_indices[1]] = bond_width
                    bond_graph[input_pair_indices[1],input_pair_indices[0]] = bond_width
                    bond_list.append(bond_width)

                    # make the line segments for plotting bonds, neighbor distances, whatever
                    bond_line_segments.append(np.asarray([pts[input_pair_indices[0]],pts[input_pair_indices[1]]]))

                    edges.append([input_pair_indices[0],input_pair_indices[1],nn_distance,bond_width])

                    if not bond_width == 0:
                        bond_list_filtered.append(bond_width)
                        nn_dist_list_filtered.append(nn_distance)
                        bond_line_segments_filtered.append(np.asarray([pts[input_pair_indices[0]],pts[input_pair_indices[1]]]))
    
                else:
                    # at least one ridge vertex is off the image
                    # these two input points are boundary
                    # boundary_sites is an Nx1 matrix
                    boundary_sites[input_pair_indices,0] = np.int(1)
            else:
                # why are these no good?
                # are they on the boundary?
                boundary_sites[input_pair_indices,0] = np.int(1)

        if not args.noplot:

            plt.figure(3)
            plot_bonds(im_original,bond_line_segments_filtered,bond_list_filtered)
            plt.savefig(output_data_path+'/'+filename+'_bond_map.png',bbox_inches='tight', dpi=300)

            plt.figure(4)
            plot_bonds(im_original,bond_width_segments,bond_list_filtered)
            plt.savefig(output_data_path+'/'+filename+'_bond_width_map.png',bbox_inches='tight', dpi=300)

            # make a map of the nn distances
            plt.figure(5)
            plot_nn_distance(im_original,bond_line_segments_filtered,nn_dist_list_filtered)
            plt.savefig(output_data_path+'/'+filename+'_nn_dist_map.png',bbox_inches='tight', dpi=300)

            # plot a histogram of the "bond strengths"
            # fit a gaussian to the histogram data
            bond_hist_window = 3.0*np.std(bond_list_filtered)
            bond_hist, bond_hist_bins = np.histogram(bond_list_filtered, bins=len(bond_list_filtered)/4, range=(np.mean(bond_list_filtered)-bond_hist_window,np.mean(bond_list_filtered)+bond_hist_window))

            # shift bin locations from edge to center
            bond_hist_bins += (bond_hist_bins[1]-bond_hist_bins[0])/2
            popt, pcov = curve_fit(gaussian, bond_hist_bins[0:len(bond_hist_bins)-1], bond_hist, p0=(np.max(bond_hist),np.mean(bond_list_filtered),np.std(bond_list_filtered)))
            fit_amp, fit_mean, fit_std = popt

            plt.figure(6)
            plt.hist(bond_list_filtered,bins=len(bond_list_filtered)/4)
            fit_curve = gaussian(bond_hist_bins,fit_amp,fit_mean,fit_std)
            plt.plot(bond_hist_bins, fit_curve, 'r-', linewidth=3)
            label_string = '%(count)i bonds, Width: %(mean).3g (nm), $\sigma$: %(sd).2g (nm), %(percent).2g%%' % {'count':len(bond_list_filtered), 'mean':fit_mean, 'sd':fit_std, 'percent':100.0*fit_std/fit_mean }
            plt.gca().set_title(label_string)
            plt.ylabel('Count')
            plt.xlabel('Connection Width (nm)')
            plt.gca().set_title(label_string)
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
