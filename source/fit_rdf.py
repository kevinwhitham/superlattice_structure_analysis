# fit_rdf.py
# convenience functions for running a radial distribution function (rdf) calculation on a set of
# points and fitting the rdf to a model to quantify disorder

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotsettings
import prettyplotlib as ppl
import seaborn as sns
from paircorrelation import PairCorrelationFunction_2D
from paircorrelation import fit_hex_paracrystal_rdf, fit_square_paracrystal_rdf, fit_hex_disordered_rdf, fit_square_disordered_rdf
from paircorrelation import generate_paracrystal_rdf, generate_disordered_rdf
import argparse
from os import path

parser = argparse.ArgumentParser()
parser.add_argument('symmetry', help='4 = square, 6 = hexagonal', type=int, default=4)
parser.add_argument('model', help='crystal disorder model for fitting: \'uniform\', \'para\'', default='uniform')
parser.add_argument('distance', help='cut off distance in number of particle diameters', type=float, default=20.0)
parser.add_argument('input_file',help='Image file')
args = parser.parse_args()

output_data_path = path.dirname(args.input_file)+'/'
filename = str.split(path.basename(args.input_file),'.')[0]

pts_file = np.load(output_data_path+filename+'_particles.npz')
pts = pts_file['centroids']
pixels_per_nm = pts_file['pixels_per_nm']
radii = pts_file['radii']
nn_dist = 2.0*np.mean(radii)*pixels_per_nm

im = plt.imread(args.input_file)
box_size = np.sqrt(float(im.shape[0])*float(im.shape[1]))

g,r,x,y = PairCorrelationFunction_2D(pts[:,0],pts[:,1], box_size, min(box_size/2.0,args.distance*nn_dist), 0.5)

# change units from pixels to nm
r /= pixels_per_nm

from scipy.optimize import curve_fit

# disordered, paracrystal, mixed
fit_functions = [[fit_square_disordered_rdf, fit_square_paracrystal_rdf], # 4-fold
                 [fit_hex_disordered_rdf, fit_hex_paracrystal_rdf]]       # 6-fold

if args.symmetry == 4:
    fit_func_row = 0
    symmetry_label = 'square'
elif args.symmetry == 6:
    fit_func_row = 1
    symmetry_label = 'hex'
else:
    print('Symmetry not recognized')
    symmetry_label = ''

if args.model == 'uniform':
    fit_func_col = 0
    params = [0.05*r[np.argmax(g)], r[np.argmax(g)]]
elif args.model == 'para':
    fit_func_col = 1
    params = [0.05*r[np.argmax(g)], r[np.argmax(g)]]
else:
    print('Model not recognized')
    
fit_function = fit_functions[fit_func_row][fit_func_col]

popt, pcov = curve_fit(fit_function, r, g, p0=params)
print(popt)
print(pcov)

# fit results
# a is the length of the first basis vector
# b is the length of the second basis vector
# popt[0] is the std. dev. 

# DEBUG!!!
# you can force the values here instead of using the fit value
#print('Overriding fit results!')
#popt[1] = 6.61;
#popt[0] = 0.035*popt[1]

if args.symmetry == 4:
    a=(popt[1],0.0)
    b=(0.0,popt[1])
elif args.symmetry == 6:
    a=(popt[1],0.0)
    b=(popt[1]*-0.5,popt[1]*np.sqrt(3.0)/2.0)

if args.model == 'uniform':
    full_model_string = 'Uncorrelated Disorder'
    r_fit, g_fit = generate_disordered_rdf(a, b, popt[0], max_distance=np.max(r), resolution=r[1]-r[0])
    func_label = r'$\sum_i\/\frac{1}{2\pi r_i \sigma} exp\left(\frac{-(r-r_i)^2}{2 \sigma^2}\right)$'
elif args.model == 'para':
    full_model_string = 'Paracrystal'
    r_fit, g_fit = generate_paracrystal_rdf(a, b, popt[0], max_distance=np.max(r), resolution=r[1]-r[0])
    func_label = r'$\sum_i\/\frac{L}{2\pi r_i^2 \sigma} exp\left(\frac{-(r-r_i)^2}{2 (\sigma r_i/L)^2}\right)$'
else:
    print('Model not recognized')

pub = plotsettings.Set('Nature')
pub.set_figsize(n_columns = 1, n_rows = 1)

plt.figure(1)
#plt.plot(r,g,'r-',label='Data')
#plt.plot(r_fit, g_fit, 'k-', label=func_label+' $\sigma$ = '+'%(std).2f%%, L = %(L).2f nm' % {'std':popt[0]/popt[1]*100.0, 'L':popt[1]})

sns.set_style("white")
sns.despine()

# make text editable in .pdf files
mpl.rc('pdf', fonttype=42)

ppl.plot(r,g,label='Data')
#ppl.plot(r_fit, g_fit, label=func_label+' $\sigma$ = '+'%(std).2f%%, L = %(L).2f nm' % {'std':popt[0]/popt[1]*100.0, 'L':popt[1]})
ppl.plot(r_fit, g_fit, label=full_model_string)
plt.xlabel('r (nm)')
plt.ylabel('g(r)')

mpl.rc('pdf', fonttype=3)

plt.gca().legend()
plt.savefig(output_data_path+filename+'_'+args.model+'_'+symmetry_label+'_RDF.pdf', bbox_inches='tight')

header_string = 'Radial distribution function\n'
header_string += 'Model: '+args.model+' Equation: '+func_label+'\n'
header_string += 'sigma = '+'%(std).2f%%, L = %(L).2f nm\n' % {'std':popt[0]/popt[1]*100.0, 'L':popt[1]}
header_string += 'r (nm)\tg(r)'
np.savetxt(output_data_path+filename+'_'+args.model+'_'+symmetry_label+'_RDF.txt',zip(r,g),header=header_string)