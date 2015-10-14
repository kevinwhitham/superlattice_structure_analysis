__author__ = 'kevin'
# 20151009 Kevin Whitham, Cornell University
# License: GNU Public License (GPL) v.3.0

import numpy as np
import argparse
import matplotlib.pyplot as plt
import skimage.io as skimio
from os import path
import seaborn as sb
import scipy.sparse as sparse

from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.colors import LinearSegmentedColormap

def create_custom_colormap(categories):

    # get a color map for mapping metric values to colors of some color scale
    value_rgb_pairs = []
    rgb_array = np.asarray([[255,0,10],[255,50,34],[255,109,59],[255,177,102],[255,220,125],[255,245,160],[255,245,192],[255,200,230],[212,251,255],[160,253,255],[120,226,255],[81,177,255],[55,127,255],[31,100,255],[0,13,255]],dtype='f4')
    rgb_array /= 255
    rgb_list_norm = []

    for value, color in zip(np.linspace(0,1,categories),rgb_array[np.linspace(0,rgb_array.shape[0]-1,num=categories,dtype='i4')]):
        value_rgb_pairs.append((value,color))

    return LinearSegmentedColormap.from_list(name="custom", colors=value_rgb_pairs, N=categories)




# load edge data
parser = argparse.ArgumentParser()
parser.add_argument('-m','--map', help='draw a map showing the number of bonds at each site', action='store_true')
parser.add_argument('order_parameter', nargs='?', help='number of nearest neighbors (4 for square lattice, etc.)', type=int, default=4)
parser.add_argument('img_file', help='path to image file to analyze')
args = parser.parse_args()

im = skimio.imread(args.img_file, as_grey=True, plugin='matplotlib')
data_directory = path.dirname(args.img_file)+'/'
base_filename = str.split(path.basename(args.img_file),'.')[0]
edge_data = np.loadtxt(data_directory+base_filename+'_edges.txt')

# load data about the particles at the edge of the image
boundary_data_file = np.load(data_directory+base_filename+'_boundary_graph.npz')

# boundary_graph is an Nx1 sparse matrix
# to test if site j is on the boundary, check if boundary_graph[j,0] == 1
boundary_graph = sparse.csc_matrix((boundary_data_file['data'], boundary_data_file['indices'],boundary_data_file['indptr']))


# make a histogram to count the number of non-zero width bonds per site
all_site_indices = np.vstack((edge_data[:,0],edge_data[:,1]))

# expected columns in data file are pt1, pt2, distance, bond width
connected_site_indices = all_site_indices[np.nonzero(np.vstack((edge_data[:,3],edge_data[:,3])))]

#bonds_per_site, unique_site_indices = np.histogram(connected_site_indices,bins=len(np.unique(connected_site_indices)))
unique_site_indices, bonds_per_site = np.unique(connected_site_indices, return_counts=True)

non_boundary_site_indices = []
total_bonds = 0
connections_to_boundary_sites = 0
# filter out the boundary sites
for index, site in zip(range(len(unique_site_indices)),unique_site_indices):
    # there are no bonds between boundary sites saved in the edges data file
    # so include the bonds between boundary sites and interior sites
    total_bonds += bonds_per_site[index]

    if not boundary_graph[site,0]:
        non_boundary_site_indices.append(index)
    else:
        connections_to_boundary_sites += bonds_per_site[index]

interior_site_indices = unique_site_indices[non_boundary_site_indices]
bonds_per_interior_site = bonds_per_site[non_boundary_site_indices]

# make a histogram of the number of bonds
bond_numbers, bonds_per_bond_number = np.unique(bonds_per_interior_site, return_counts=True)

plt.figure(1)
sb.barplot(bond_numbers, bonds_per_bond_number, hue=np.zeros(bond_numbers.shape))
plt.xlabel('Bonds per Site')
plt.ylabel('Count')
plt.savefig(data_directory+base_filename+'_bond_count_hist.png', bbox_inches='tight')


points_file = np.load(data_directory+base_filename+'_particles.npz')
pts = points_file['centroids']
pixels_per_nm = points_file['pixels_per_nm']
radii = points_file['radii']

"""
# calculate the number of missing bonds
# maximum number of bonds is non-boundary sites * order number (4 for square, 6 for hexagonal, etc.)
#   divide by two for double counting, add in the connections between interior and boundary sites / 2 (double count)
# missing bonds is the maximum number of bonds minus the actual number of non-zero width bonds
# Example: a 4x4 square lattice. There are 16 sites, 12 boundary sites, 4 interior sites
# bonds between boundary sites are not included in the input file, so the perfect network would look like:
# 0 0 0 0
#   | |
# 0-0-0-0
#   | |
# 0-0-0-0
#   | |
# 0 0 0 0
#
# maximum number of bonds: (16 - 12) * 4/2 + 8/2 = 12
# if one bond is missing, then total_bonds will be 22 (because we double counted)
#   and the missing percentage is 100 * (12-11)/12 = 8.3%
# This won't work if there are sites with more than the theoretical maximum number of bonds
# (i.e. 5 bonds in a square lattice)
"""

# use this line for debugging
# print(pts.shape,boundary_graph.nnz,args.order_parameter,total_bonds,connections_to_boundary_sites)

maximum_possible_bonds = (pts.shape[0] - boundary_graph.nnz) * args.order_parameter / 2 + connections_to_boundary_sites/2
missing_bond_percentage = 100.0 * (float(maximum_possible_bonds) - float(total_bonds/2))/float(maximum_possible_bonds)

header_string  = 'Missing connections: %(mp).2f%%\n' % {'mp':missing_bond_percentage}
header_string += 'Bonds\tCount'
np.savetxt(data_directory+base_filename+'_bond_count_hist.txt',np.vstack((bond_numbers,bonds_per_bond_number)).transpose(),fmt='%d',header=header_string)


# Extra features
# plot a map showing the number of bonds on each site
if args.map:
    plt.figure(2)
    implot = plt.imshow(im)
    implot.set_cmap('gray')

    # make a color map with enough colors to identify sites with different bond numbers
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import matplotlib.cm as cmx

    color_palette = create_custom_colormap(max(args.order_parameter,len(bond_numbers)))
    color_norm = mcolors.Normalize(vmin=0, vmax=max(args.order_parameter,len(bond_numbers)))
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='jet')

    patches = []
    for site_number, site_index in zip(range(len(interior_site_indices)),interior_site_indices):
        patches.append(Circle(pts[site_index],radii[site_index]*pixels_per_nm,edgecolor='none', facecolor=scalar_map.to_rgba(bonds_per_interior_site[site_number]), alpha=0.5))

    pc = PatchCollection(patches, match_original=True)
    plt.gca().add_collection(pc)

    # plot bonds
    line_segments = []
    bond_list = []
    filtered_edges = edge_data[np.nonzero(edge_data[:,3])]
    for pt1_index, pt2_index, distance, bond_width in filtered_edges:
        line_segments.append(np.asarray([pts[pt1_index],pts[pt2_index]]))
        bond_list.append(bond_width)

    lc = LineCollection(line_segments,colors='w')

    # could color code the bond lines here
    # lc.set_array(np.asarray(bond_list))

    lc.set_linewidth(1)
    plt.gca().add_collection(lc)

    # add some patches for the legend instead of using a color bar
    legend_patches = []
    for bond_number in bond_numbers:
        legend_patches.append(mpatches.Patch(color=scalar_map.to_rgba(bond_number), label=str(bond_number)))

    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)

    # bond_color_bar = plt.colorbar(lc)
    # bond_color_bar.ax.set_ylabel('Bond Width (nm)')

    # set the limits for the plot
    # set the x axis range
    plt.gca().set_xlim(0, im.shape[1])

    # set the y-axis range and flip the y-axis
    plt.gca().set_ylim(im.shape[0], 0)

    # save this plot to a file
    plt.gca().set_axis_off()

    # add the colorbar
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes(position='right', size='5%', pad = 0.05)
    # cbar = plt.colorbar(pc, cax=cax, ticks=range(np.max(bond_numbers)+1))
    # cbar.ax.set_yticklabels(map(str,range(np.max(bond_numbers)+1)))
    # cax.set_xlabel('Bonds')
    # cax.xaxis.set_label_coords(0.5, 1.04)

    plt.savefig(data_directory+base_filename+'_bond_count_map.png',bbox_inches='tight',dpi=300)