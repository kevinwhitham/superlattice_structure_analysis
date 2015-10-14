# README #

This is the readme document accompanying software to perform analysis on images of particles, currently called Superlattice Structure Analysis. This software is distributed under the GNU Public License (GPL) v.3.0.

### What is this repository for? ###

This software performs analysis of images containing particles, connected or unconnected, to measure:

radius
centroid
nearest neighbor distance
structure metric
width and number of connections to nearest neighbors

The data is output as text files, numpy array files, and images.

### How do I get set up? ###

 1. Install dependencies:
    python
    python packages:
        numpy
        scipy
        matplotlib
        scikit-image
        Cython
        
    There are many ways to get python depending on your operating system. If you have OS X,
    you already have python and maybe the matplotlib, numpy and scipy packages. In that case,
    to get other packages such as scikit-image, use a package installer such as pip or easy_install.
    
    Open a terminal and type
        sudo pip scikit-image
        or
        sudo easy_install scikit-image
        
    If you don't have python or want a different version, either use a package installer 
    (like MacPorts or homebrew on OS X) or use a distribution such as Anaconda or Canopy.


 2. clone the repository to your machine
    The easiest way to keep up with bug fixes, new development, etc. is to 
    use an SVN client like SourceTree (www.sourcetreeapp.com)
    
    Otherwise you can just download the latest version as a zip file
    

### How do I run it? ###

Navigate to the /source directory in your console.

To run the structure, size, and connection analysis type:
```
#!

python structure_metric.py [options] <N> <path to image file> [parameters]
```

N is the desired order of the structure metric (e.g. 4, 6, etc.)
filepath is the complete path to the image file (tif, png, etc.)

prefix options are:
```
#!

-h                  help
-b, --black         specify the image has a dark background with lighter colored particles
-np, --noplot       do not create any image files, only text and numpy array files (saves time)
-nd, --nodata       do not output data files (if you only want the images and plots)
-e, --edge          plot the structure metric or outline of particles at the edge of a blank region
-m, --morph         use morphological filtering during particle search
-o, --outline       plot the Voronoi cells as outlines without the structure metric color
-mc, --montecarlo   compute and output data that could be used for MC (NN distance, bonds, etc.)
-d, --debug         turns on debugging output, you can see how well it is thresholding your image
```

post-fix optional parameters are:
```
#!
pts_file         the path to a text file with two columns (X and Y) specifying the centroid location of each particle in pixel units.
                    you may use this option to bypass the algorithm that attempts to find the particles. Useful when the image quality is
                    poor, such as when there is low contrast between particles and background. You can generate this file for example 
                    with ImageJ.

pix_per_nm  floating point number specifying the image scaling in units of pixels/nm. Use this if the image does not have one
                     of the scale bars contained in the /input directory. To use this option you must also specify the pts_file
                     parameter, use a blank path '' if you are not using a file of points

```


To run the radial distribution function analysis (after running the structure analysis to get the particle locations):

```
#!
python fit_rdf.py <N> <model> <distance> <path to image file>
```

N is the order of the symmetry you're looking for (4 or 6 for square or hexagonal)
model can be either 'para' or 'uniform' to fit using a paracrystalline disorder model, or a uniform (uncorrelated) model
distance is the maximum distance to calculate and fit the radial distribution function in units of the nearest neighbor distance

To analyze the number of connections per site and the percentage of connections missing (after running structure analysis with the option -mc to output data about the connections):

```
#!
python bond_analysis.py [-m] N <path to image file>
```

options:
-m output images showing the number of connections per site and a histogram of the number of connections

parameters:
N the order of the structure for calculating the maximum number of bonds (4 for square, 6 for hexagonal, etc.)

### Examples ###
You can practice on a provided example image by running the following commands.
```
#!
python structure_metric.py -o -m -e 4 ../examples/example.tif
python fit_rdf.py 4 para 5 ../examples/example.tif
python fit_rdf.py 4 uniform 5 ../examples/example.tif
```

### Windows Notes ###
If you are running on Windows, you should buy a Mac. Until then, you might need to recompile the minkowski_metric module if you get an error about not finding the module 'minkowski_metric'. To do so you'll need to have a compiler, such as Microsoft Visual Studio (check [this](https://github.com/cython/cython/wiki/CythonExtensionsOnWindows)). Make sure you have the cython module, then run:
```
python setup_minkowski_metic.py build_ext --inplace
```

This should generate a file 'minkowski_metric.pyd'


### Dependencies: ###
Python (tested with 2.7.8) with packages matplotlib, numpy, scipy, scikit-image.

### Who do I talk to? ###
Author:
k (as in Kilo) w (as in Whiskey) 242 (at) cornell (dot) edu