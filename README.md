# README #

This is the readme document accompanying software to perform analysis on images of particles, currently called Superlattice Structure Analysis. This software is distributed under the GNU Public License (GPL) v.3.0.

### What is this repository for? ###

This software performs analysis of images containing particles, connected or unconnected, to measure:

radii
centroid
nearest neighbor distance
structure metric
width of "bridge" or "bond" connecting nearest neighbors

The data is output as text files, numpy array files, and images.

### How do I get set up? ###

1. Install dependencies:
    python
    python packages:
        numpy
        scipy
        matplotlib
        scikit-image
        
    There are many ways to get python depending on your operating system. If you have OS X,
    you already have python and maybe the matplotlib, numpy and scipy packages. In that case,
    to get other packages such as scikit-image, use a package installer such as pip or easy_install.
    
    Open a terminal and type
        sudo pip scikit-image
        or
        sudo easy_install scikit-image
        
    If you don't have python or want a different version, either use a package installer 
    (like MacPorts on OS X) or use a distribution such as Anaconda or Canopy.


2. clone the repository to your machine
    The easiest way to keep up with bug fixes, new development, etc. is to 
    use an SVN client like SourceTree (www.sourcetreeapp.com)
    
    Otherwise you can just download the latest version as a zip file
    
3. run using the command:

python <path to structure_metric.py> [options] N <path to image file> [parameters]

N is the desired order of the structure metric (e.g. 4, 6, etc.)
filepath is the complete path to the image file (tif, png, etc.)

prefix options are:
-h                  help
-b, --black         specify the image has a dark background with lighter colored particles
-n, --noplot        do not create any image files, only text and numpy array files (saves time)
-e, --edge          plot the structure metric of particles at the edge of a region
-m, --morph         use morphological filtering during particle search
-o, --outline       plot the Voronoi cells as outlines without the structure metric color
-mc, --montecarlo   compute and output data that could be used for MC (NN distance, bonds, etc.)

post-fix optional parameters are:
pts_file         the path to a text file with two columns (X and Y) specifying the centroid location of each particle in pixel units.
                    you may use this option to bypass the algorithm that attempts to find the particles. Useful when the image quality is
                    poor, such as when there is low contrast between particles and background. You can generate this file for example 
                    with ImageJ.

pix_per_nm  floating point number specifying the image scaling in units of pixels/nm. Use this if the image does not have one
                     of the scale bars contained in the /input directory. To use this option you must also specify the pts_file
                     parameter, use a blank path '' if you are not using a file of points

### Dependencies: ###
Python (tested with 2.7.8) with packages matplotlib, numpy, scipy, scikit-image.

### Who do I talk to? ###
Author:
k (as in Kilo) w (as in Whiskey) 242 (at) cornell (dot) edu