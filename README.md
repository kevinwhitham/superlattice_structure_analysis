# README #

This is the readme document accompanying software to perform analysis on images of particles, currently called Superlattice Structure Analysis. This software is distributed under the GNU Public License (GPL) v.3.0.

### What is this repository for? ###

* Quick summary
This software performs analysis of images containing particles, connected or unconnected, to measure:

radii
centroid
nearest neighbor distance
Minkowski structure metric
width of "bridge" or "bond" connecting nearest neighbors

The data is output as text files, numpy array files, and images.

### How do I get set up? ###

Install dependencies, clone the repository to your machine, run using the command:

python structure_metric.py N filepath

N is the desired order of the Minkowski structure metric.
filepath is the complete path to the image file (tif, png, etc.)

Currently only 4th and 6th order are implemented, but others may be generated using generate_sph_harm.pyx

prefix options are:
-h                help
-b, --black   specify the image has a dark background with lighter colored particles
-n, --noplot  do not plot any histograms or output image files, only text and numpy array files (saves time, drawing is slow)

post-fix optional parameters are:
pts_file         the path to a text file with two columns (X and Y) specifying the centroid location of each particle in pixel units.
                    you may use this option to bypass the algorithm that attempts to find the particles. Useful when the image quality is
                    poor, such as when there is low contrast between particles and background. You can generate this file for example 
                    with ImageJ.

pix_per_nm  floating point number specifying the image scaling in units of pixels/nm. Use this if the image does not have one
                     of the scale bars contained in the /input directory.

* Configuration
Tested on Mac OS X 10.9.4

* Dependencies
Python (tested with 2.7) with packages matplotlib, numpy, scipy, scikit-image.

### Who do I talk to? ###
Author:
kw242@cornell.edu