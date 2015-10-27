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

from scipy.optimize import curve_fit

def create_custom_colormap(categories):

    # get a color map for mapping metric values to colors of some color scale
    value_rgb_pairs = []
    rgb_array = np.asarray([[255,0,10],[255,50,34],[255,109,59],[255,177,102],[255,220,125],[255,245,160],[255,245,192],[255,200,230],[212,251,255],[160,253,255],[120,226,255],[81,177,255],[55,127,255],[31,100,255],[0,13,255]],dtype='f4')
    rgb_array /= 255
    rgb_list_norm = []

    for value, color in zip(np.linspace(0,1,categories),rgb_array[np.linspace(0,rgb_array.shape[0]-1,num=categories,dtype='i4')]):
        value_rgb_pairs.append((value,color))

    return LinearSegmentedColormap.from_list(name="custom", colors=value_rgb_pairs, N=categories)

def gaussian(x, amp, mean, std):
    return amp * np.exp(-(x-mean)**2 / (2.0 * std**2))


# load edge data
parser = argparse.ArgumentParser()
parser.add_argument('-m','--map', help='draw a map showing the number of bonds at each site', action='store_true')
parser.add_argument('order_parameter', nargs='?', help='number of nearest neighbors (4 for square lattice, etc.)', type=int, default=4)
parser.add_argument('img_file', help='path to image file to analyze')
args = parser.parse_args()

im = skimio.imread(args.img_file, as_grey=True, plugin='matplotlib')
data_directory = path.dirname(args.img_file)+'/'
base_filename = str.split(path.basename(args.img_file),'.')[0]

# the edge_data matrix contains a row for each edge
# the columns are: pt1_index, pt2_index, distance (nm), bond_width (nm)
edge_data = np.loadtxt(data_directory+base_filename+'_edges.txt')

# load data about the particles at the edge of the image
boundary_data_file = np.load(data_directory+base_filename+'_boundary_graph.npz')

# boundary_graph is an Nx1 sparse matrix
# to test if site j is on the boundary, check if boundary_graph[j,0] == 1
boundary_graph = sparse.csc_matrix((boundary_data_file['data'], boundary_data_file['indices'],boundary_data_file['indptr']))

# load the facet lengths of the voronoi graph
facet_length_file = np.load(data_directory+base_filename+'_facet_length_graph.npz')
facet_length_graph = sparse.csc_matrix((facet_length_file['data'], facet_length_file['indices'], facet_length_file['indptr']))

# load the bond widths
bond_graph_file = np.load(data_directory+base_filename+'_bond_graph_csc.npz')
bond_width_graph = sparse.csc_matrix((bond_graph_file['data'], bond_graph_file['indices'], bond_graph_file['indptr']))

# load the particle data to get the locations for overlay on the image
points_file = np.load(data_directory+base_filename+'_particles.npz')
pts = points_file['centroids']
pixels_per_nm = points_file['pixels_per_nm']
radii = points_file['radii']

# find the real bonds
# for each edge, find the largest (args.order_parameter) faceted bonds
all_site_indices = np.vstack((edge_data[:,0],edge_data[:,1]))
unique_site_indices = np.unique(all_site_indices)

# print('Edge matrix shape: '+str(edge_data.shape))
# print('Indices in bond_width_graph: '+str(bond_width_graph.indices))
# print('Indptr in bond_width_graph: '+str(bond_width_graph.shape))

line_segments = []
line_colors = []
connected_neighbors = 0
unconnected_neighbors = 0

for pt1_index in unique_site_indices:
    facet_lengths = facet_length_graph.data[facet_length_graph.indptr[pt1_index]:facet_length_graph.indptr[pt1_index+1]]
    neighbor_indices = facet_length_graph.indices[facet_length_graph.indptr[pt1_index]:facet_length_graph.indptr[pt1_index+1]]
    #bond_widths = bond_width_graph.data[bond_width_graph.indptr[pt1_index]:bond_width_graph.indptr[pt1_index+1]]

    sorted_facet_indices = np.argsort(facet_lengths)[::-1]

    # if there are more facets than the order_parameter (e.g. more than 4 for a square lattice, etc.)
    # just take the biggest ones
    if len(facet_lengths) > args.order_parameter:
        extra_dest_indices = neighbor_indices[sorted_facet_indices[args.order_parameter:len(facet_lengths)]]

        # get rid of extra connections that exceed the order_parameter
        extra_src_indices = np.ones(len(extra_dest_indices)) * pt1_index
        facet_length_graph[extra_dest_indices,extra_src_indices] = 0




facet_length_graph.eliminate_zeros()

# now check the edges left in facet_length_graph to see if they are connected or missing bonds
for pt1_index in np.arange(len(facet_length_graph.indptr)-1):
    neighbor_indices = facet_length_graph.indices[facet_length_graph.indptr[pt1_index]:facet_length_graph.indptr[pt1_index+1]]

    for pt2_index in neighbor_indices:

        if facet_length_graph[pt1_index,pt2_index]:

            line_segments.append(np.asarray([pts[pt1_index],pts[pt2_index]]))

            if bond_width_graph[min(pt2_index,pt1_index),max(pt1_index,pt2_index)]:
                connected_neighbors += 1
                line_colors.append((0,1,0))
            else:
                unconnected_neighbors += 1
                line_colors.append((1,0,0))

            # prevent double counting
            facet_length_graph[pt2_index,pt1_index] = 0


# determine how many bonds are at each site
# make a histogram to count the number of non-zero width bonds per site

# expected columns in data file are pt1, pt2, distance, bond width
connected_site_indices = all_site_indices[np.nonzero(np.vstack((edge_data[:,3],edge_data[:,3])))]

unique_connected_site_indices, bonds_per_site = np.unique(connected_site_indices, return_counts=True)

non_boundary_site_indices = []

# filter out the boundary sites
for index, site in zip(range(len(unique_connected_site_indices)),unique_connected_site_indices):
    if not boundary_graph[site,0]:
        non_boundary_site_indices.append(index)

interior_site_indices = unique_connected_site_indices[non_boundary_site_indices]
bonds_per_interior_site = bonds_per_site[non_boundary_site_indices]

# make a histogram of the number of bonds
bond_numbers, bonds_per_bond_number = np.unique(bonds_per_interior_site, return_counts=True)

plt.figure(1)
sb.barplot(bond_numbers, bonds_per_bond_number, hue=np.zeros(bond_numbers.shape))
plt.xlabel('Bonds per Site')
plt.ylabel('Count')
plt.savefig(data_directory+base_filename+'_bond_count_hist.png', bbox_inches='tight')

# find the number of missing bonds
# it's impossible to calculate the number of missing bonds
# from the number of sites and bonds because of the boundary sites and
# because there are edges between next-nearest neighbors, not just nearest neighbors

connectivity = 100.0 * connected_neighbors/(unconnected_neighbors + connected_neighbors)

header_string  = 'Number of sites: '+str(pts.shape[0])+'\n'
header_string += 'Number of boundary sites: '+str(boundary_graph.nnz)+'\n'
header_string += 'Connected neighbors: '+str(connected_neighbors)+'\n'
header_string += 'Unconnected neighbors: '+str(unconnected_neighbors)+'\n'
header_string += 'Connectivity: %(con).2f%%\n' % {'con':connectivity}
header_string += 'Coordination_Number\tCount'
np.savetxt(data_directory+base_filename+'_bond_count_hist.txt',np.vstack((bond_numbers,bonds_per_bond_number)).transpose(),fmt='%d',header=header_string)


# Extra features
# plot a map showing the number of bonds on each site
if args.map:
    plt.figure(2)
    implot = plt.imshow(im, interpolation='none')
    implot.set_cmap('gray')

    # make a color map with enough colors to identify sites with different bond numbers
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import matplotlib.cm as cmx

    color_palette = create_custom_colormap(max(args.order_parameter,len(bond_numbers)))
    color_norm = mcolors.Normalize(vmin=0, vmax=max(args.order_parameter,len(bond_numbers)))
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='jet')

    patches = []
    for site_number, pt1_index in zip(range(len(interior_site_indices)),interior_site_indices):
        patches.append(Circle(pts[pt1_index],radii[pt1_index]*pixels_per_nm/2.0,edgecolor='none', facecolor=scalar_map.to_rgba(bonds_per_interior_site[site_number]), alpha=0.5))

    pc = PatchCollection(patches, match_original=True)
    plt.gca().add_collection(pc)

    # make a list of line segments to plot
    # filtered_edges = edge_data[np.nonzero(edge_data[:,3])]
    # for pt1_index, pt2_index, distance, bond_width in filtered_edges:
    #     line_segments.append(np.asarray([pts[pt1_index],pts[pt2_index]]))

    # plot bonds
    lc = LineCollection(line_segments,colors=line_colors, linewidths=0.25)
    plt.gca().add_collection(lc)

    # add some patches for the legend instead of using a color bar
    legend_patches = []
    for bond_number in bond_numbers:
        legend_patches.append(mpatches.Patch(color=scalar_map.to_rgba(bond_number), label=str(bond_number)))

    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0, title='Connections')

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

    plt.savefig(data_directory+base_filename+'_bond_count_map.pdf',bbox_inches='tight',dpi=300)