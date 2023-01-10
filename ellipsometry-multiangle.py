#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to analyze the ellipsometry data recorded on the Picometer Light ellipsometer available at the 
Partneship for soft condensed matter at the Institut Laue-Langevin, Grenoble. The script uses the tmm package
developed by Steven J. Byrnes (see https://pypi.org/project/tmm/ for full details) and the lmfit package
 (https://lmfit.github.io/lmfit-py/). 
@author: Leonardo Chiappisi
"""
date = '2023.01.10'
version = '0.3.1'

import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
import itertools 
import matplotlib.patches as mpl_patches
import numpy as np
from lmfit import minimize, Parameters, fit_report
from ellipsometry_multiangle_functions import ell, ell_xy
sys.path.insert(0, './tmm-0.1.7')
import tmm_core as tmm
basepath = './' #Folder where the datafiles are contained. 
wl = 632.8 #Laser wavelength, given in nm


''' define the model here. For each layer, a couple of parameters d_i and n_i need to be defined. 
Set the guessed value for the parameter, its limits and if it has to be optmized or not. '''

fit_params = Parameters()
fit_params.add('d0', value= np.inf, vary=False)  #Incoming medium thickness
fit_params.add('n0', value= 1.0, vary=False)  #Incoming medium thickness
# fit_params.add('d1', value= 0.5, vary=False, min = 0.0, max = 5.0)  #1st layer thickness, nm
# fit_params.add('n1', value= 1.15, vary=True, min=1.0, max=1.55)  #1st layer refractive index
#fit_params.add('d11', value= 50.1, vary=True, min = 0.0, max = 200)  #1st layer thickness, nm
#fit_params.add('n11', value= 1.55, vary=True, min=1.2, max=1.6)  #1st layer refractive index
fit_params.add('d2', value= 130, min=50., max=150., vary=True)  #2st layer thickn1ess, nm   (SiO2)
fit_params.add('n2', value= 1.47, vary=False)  #2st layer refractive index
fit_params.add('d3', value= np.inf, vary=False)  #2st layer th1ickness, nm (Si)
# fit_params.add('d3', value= np.inf, vary=False)  #2st layer thickness, nm (Si)
fit_params.add('n3', value= 3.88, vary=False)  #2st layer refractive index
# fit_params.add('d3', value= np.inf, vary=False)  #2st layer thickness, nm (Si)

elements = list(itertools.chain(*[(key, str(key)+'_err') for key in fit_params.valuesdict()]))
fitted_params = pd.DataFrame(columns=elements) #Pandas DataFrame where all fit parameters are saved. 


''' List here the files to be analysed using the same set of initial parameters '''
Files_to_be_analysed = ['SiO2_120nm_exy.txt', 'SiO2_130nm_exy.txt']
# Files_to_be_analysed = [file for file in sorted(os.listdir(basepath)) if file.endswith('exy.txt')] #


def fcn2min_xy(pars, angle):
    x, y = ell_xy(pars,angle,wl)
    res1 = x_exp - x
    res2 = y_exp - y
    res = np.concatenate([res1, res2])
    return res

plt.clf()


with open(os.path.join(basepath,'fit_out.csv'), 'w+') as f:
    f.write('#Data analyzed with pyEll, version {} from {}. \n'.format(version, date))
    f.write('#The program was first described in: Langmuir 2020, 36, 37, 10941-10951 and freely available at https://github.com/leonardo-chiappisi/pyEllipsometry \n')
    
 

for filename in Files_to_be_analysed:
        angle, x_exp, y_exp = np.loadtxt(basepath+filename, skiprows=1, usecols=(5,0,1), unpack=True)       
        out = minimize(fcn2min_xy, fit_params, args=(angle,), method='leastsq') 
        print(fit_report(out))
        x, y = ell_xy(out.params,angle,wl)
       
        
       
        fig, ax1 = plt.subplots()
        ax1.set_title(filename)
        ax2 = ax1.twinx()
        ax1.plot(angle, x, color='red')
        ax2.plot(angle, y, color='blue')
        ax1.plot(angle, x_exp, 'ro', label='')
        ax2.plot(angle, y_exp, 'bs', label='')
        # print(filename, '\\',data[:,0], np.asarray(psi[:])*180./np.pi)
        ax1.set_xlabel('Angle / deg')
        ax1.set_ylabel('x ', color='red')
        ax2.set_ylabel('y ', color='blue')
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # ax1.legend(lines, labels, loc='center right', title='n$_3$= {:.4f} $\pm$ {:.4f} \nn$_t$ = 1.5426'.format(out.params['n3'].value, out.params['n3'].stderr) )
        fig.tight_layout()  
        
        
        plt.savefig(os.path.join(basepath, filename.split('_xy')[0] + '.pdf'))
        
        
        outfile = 'fit-' + filename.split('_exy')[0] + '.dat'
        header = 'Angle \t x \t y \t x_exp \t y_exp'
        np.savetxt(os.path.join(basepath,outfile), np.transpose([angle, x,  y, x_exp,  y_exp]), delimiter="\t", header=header)
            
        
        out_values = {}
        for key in out.params.valuesdict():
            out_values[key] = out.params[key].value
            out_values[str(key)+'_err'] = out.params[key].stderr
                    
        fitted_params.loc[filename.split('_exy')[0]] = out_values
        # fitted_params.to_csv(os.path.join(basepath,'fit_out.csv'), mode='a+')
            
        # if fit_xy is True:
        #     if filename.endswith(".txt") is False:
        #         data = np.loadtxt(basepath+filename, skiprows=1, usecols=(5,1,0), unpack=False)
        #         fcn2min(fit_params, data)
        #         out = minimize(fcn2min, fit_params, kws = {'data': data}, method='leastsq')
        #         psi, delta = ell(out.params,data[:,0],wl)
        #         x, y = psidelta_in_xy(psi, delta)
        #         print(fit_report(out))

        #         plt.figure()
        #         plt.plot(data[:,0], np.asarray(y[:]), label='y')
        #         plt.plot(data[:,0], np.asarray(x[:]), label='x')
        #         plt.xlabel('Angle / deg')
        #         plt.ylabel('x, y')
        #         plt.plot(data[:,0], data[:,1], 'ro')
        #         plt.plot(data[:,0], data[:,2], 'bs')
        #         plt.legend()
        #         plt.savefig(os.path.join(basepath, filename.split('_xy')[0] + '.pdf'))
        #         plt.close()
                
        #         outfile = 'fit-' + filename.split('_xy')[0] + '.dat'
        #         np.savetxt(os.path.join(basepath, outfile), np.transpose([data[:,0],  np.asarray(delta[:])*180./np.pi,  np.asarray(psi[:])*180./np.pi]), delimiter="\t")
                
        #         # out_values = out.params.valuesdict()
        #         out_values = {}
        #         for key in out.params.valuesdict():
        #             out_values[key] = out.params[key].value
        #             out_values[str(key)+'_err'] = out.params[key].stderr
                    

        # else:
            # if filename.endswith("epd.txt"):
            #     print(filename)
            #     angle, psi_exp, delta_exp = np.loadtxt(basepath+filename, skiprows=1, usecols=(5,0,1), unpack=True)
            #     angle, psi_exp, delta_exp = np.delete(angle, np.argwhere(np.isnan(psi_exp))), np.delete(psi_exp, np.argwhere(np.isnan(psi_exp))), np.delete(delta_exp, np.argwhere(np.isnan(psi_exp)))
                
      
   
            #     out = minimize(fcn2min, fit_params, args=(angle,), method='leastsq') 
            #     psi, delta = ell(out.params,angle,wl)
            #     print(fit_report(out))
        

            #     fig, ax1 = plt.subplots()
            #     ax1.set_title(filename)
            #     ax2 = ax1.twinx()
            #     ax1.plot(angle, np.asarray(psi[:])*180./np.pi, color='red')
            #     ax2.plot(angle, np.asarray(delta[:])*180./np.pi, color='blue')
            #     ax1.plot(angle, psi_exp*180/np.pi, 'ro', label='psi')
            #     ax2.plot(angle, delta_exp*180./np.pi, 'bs', label='delta')
            #     # print(filename, '\\',data[:,0], np.asarray(psi[:])*180./np.pi)
            #     ax1.set_xlabel('Angle / deg')
            #     ax1.set_ylabel('Psi / deg', color='red')
            #     ax2.set_ylabel('Delta / deg', color='blue')
            #     lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
            #     lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            #     fig.legend(lines, labels)
            #     fig.tight_layout()  
            #     plt.savefig(os.path.join(basepath, filename.split('_epd')[0] + '.pdf'))
            #     # plt.close()
 
            #     a_min = np.arctan(fit_params['n3'].value/fit_params['n0'].value)*180/np.pi
            #     print('The calculated brewster angle, using refractive indices {} and {} is: {:.2f} degrees.'.format(fit_params['n3'].value, fit_params['n0'].value, a_min))
            #     print('The experimentally determined Brewster angle is {} degrees.'.format(angle[np.argmin(psi_exp)]))
            #     print('The beam seems to be off by {:.2f}'.format(abs(angle[np.argmin(psi_exp)]-a_min)))
                
 
            #     outfile = 'fit-' + filename.split('_epd')[0] + '.dat'
            #     header = 'Angle \t Delta \t Psi \t Delta_exp \t Psi_exp'
            #     np.savetxt(os.path.join(basepath,outfile), np.transpose([angle, np.asarray(delta)*180./np.pi,  np.asarray(psi[:])*180./np.pi, delta_exp*180./np.pi,  psi_exp*180/np.pi]), delimiter="\t", header=header)
                
            #     out_values = {}
            #     for key in out.params.valuesdict():
            #         out_values[key] = out.params[key].value
            #         out_values[str(key)+'_err'] = out.params[key].stderr
                    
            #     fitted_params.loc[filename.split('_')[0]] = out_values
                
                # with open(os.path.join(basepath,'fit_out.out'), 'a') as f:
                    # f.write('{:s} \t {:.2f} \t {:.2f} \t {:.3f} \t {:.3f} \n'.format(filename.split('_')[0], out.params['d1'].value, out.params['d1'].stderr,  out.params['n1'].value, out.params['n1'].stderr))

# print(fitted_params)
fitted_params.to_csv(os.path.join(basepath,'fit_out.csv'), mode='a+')    