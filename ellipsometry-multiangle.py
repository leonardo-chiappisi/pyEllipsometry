#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to analyze the ellipsometry data recorded on the Picometer Light ellipsometer available at the 
Partneship for soft condensed matter at the Institut Laue-Langevin, Grenoble. The script uses the tmm package
developed by Steven J. Byrnes (see https://pypi.org/project/tmm/ for full details) and the lmfit package
 (https://lmfit.github.io/lmfit-py/). 

@author: Leonardo Chiappisi
"""
date = '2020.03.22'
version = '0.1'

import sys
import matplotlib.pyplot as plt
import os
import numpy as np
from lmfit import minimize, Parameters, fit_report
basepath = '/data/owncloud/PostDoc-ILL/Lukas/Ellipsometry/200221ellipso/' #Folder where the datafiles are contained. 
from tmm import ellips

fit_xy = False  #are the data in xy fitted? Otherwise  delta and psi will be fitted. 

degree = np.pi/180
wl = 632.8 #Laser wavelength, given in nm

''' define the model here. For each layer, a couple of parameters d_i and n_i need to be defined. 
Set the guessed value for the parameter, its limits and if it has to be optmized or not. '''

fit_params = Parameters()
fit_params.add('d0', value= np.inf, vary=False)  #Incoming medium thickness (eg air or water)
fit_params.add('n0', value= 1, vary=False)  #Incoming medium refractive index
fit_params.add('d1', value= 2.1, vary=True, min = 0.0, max = 20)  #1st layer thickness, nm
fit_params.add('n1', value= 1.55, vary=True, min=1.3, max=1.6)  #1st layer refractive index
#fit_params.add('d11', value= 50.1, vary=True, min = 0.0, max = 200)  #1st layer thickness, nm
#fit_params.add('n11', value= 1.55, vary=True, min=1.2, max=1.6)  #1st layer refractive index
fit_params.add('d2', value= 5.0, vary=False)  #2st layer thickness, nm   (SiO2)
fit_params.add('n2', value= 1.46, vary=False)  #2st layer refractive index
fit_params.add('d3', value= np.inf, vary=False)  #2st layer thickness, nm (Si)
fit_params.add('n3', value= 3.87, vary=False)  #2st layer refractive index


def ell(pars,ang,wl):
    '''takes refractive index and layer thickness and returns psi, delta'''
    vals = pars.valuesdict()
    ns = [vals[key] for key in vals if 'n' in key] #List of refractive indices, starting from air to the substrate
    ds = [vals[key] for key in vals if 'd' in key] #List of refractive indices, starting from air to the substrate
    #ds=[vals['d0'], vals['d1'], vals['d2'], vals['d3']] #List of heights of the layers, in nm. Has the same length as n_list
    psi = [ellips(ns, ds, i*degree, wl)['psi'] for i in ang]
    delta = np.pi - np.array([ellips(ns, ds, i*degree, wl)['Delta'] for i in ang]) #in nm
    return psi, delta
    

def psidelta_in_xy(psi,delta):
    ''' Converts Psi and Delta in x and y'''
    x = -np.sin(2*np.asarray(psi))*np.cos(delta)
    y = np.sin(2*np.asarray(psi))*np.sin(delta)
    return x, y
    

def fcn2min(pars, data):
    psi, delta = ell(pars,data[:,0],wl)
    if fit_xy is True:
        x, y = psidelta_in_xy(psi, delta)
        res1 = data[:,1] - psi
        res2 = data[:,2] - psi
    else:
        res1 = data[:,1] - psi
        res2 = data[:,2] - delta
    res = np.concatenate([res1, res2])
    return res

plt.clf()


with open(os.path.join(basepath,'fit_out.out'), 'w+') as f:
    f.write('#Data analyzed with pyEll, version {} from {}. \n#Filename \t d \t err_d \t n \t err_n \n'.format(version, date))
    
 

for filename in sorted(os.listdir(basepath)):
    if filename.endswith(".txt"):
        if fit_xy is True:
            if filename.endswith("epd.txt") is False:
                data = np.loadtxt(basepath+filename, skiprows=1, usecols=(5,0,1), unpack=False)
                fcn2min(fit_params, data)
                out = minimize(fcn2min, fit_params, kws = {'data': data}, method='leastsq')
                psi, delta = ell(out.params,data[:,0],wl)
                x, y = psidelta_in_xy(psi, delta)
                print(fit_report(out))
                
                plt.figure()
                plt.plot(data[:,0], np.asarray(y[:]), label='y')
                plt.plot(data[:,0], np.asarray(x[:]), label='x')
                plt.xlabel('Angle / deg')
                plt.ylabel('x, y')
                plt.plot(data[:,0], data[:,1], 'ro')
                plt.plot(data[:,0], data[:,2], 'bs')
                plt.legend()
                plt.savefig(os.path.join(basepath, filename.split('_xy')[0] + '.pdf'))
                plt.close()
                
                outfile = 'fit-' + filename.split('_xy')[0] + '.dat'
                np.savetxt(os.path.join(basepath, outfile), np.transpose([data[:,0],  np.asarray(delta[:])*180./np.pi,  np.asarray(psi[:])*180./np.pi]), delimiter="\t")
                
                
                with open(os.path.join(basepath,'fit_out.out'), 'a') as f:
                    f.write('{:s} \t {:.2f} \t {:.2f} \t {:.3f} \t {:.3f} \n'.format(filename.split('_p')[0], out.params['d1'].value, out.params['d1'].stderr,  out.params['n1'].value, out.params['n1'].stderr))
        else:
            if filename.endswith("epd.txt"):
                print(filename)
                tmp = np.loadtxt(basepath+filename, skiprows=1, usecols=(5,0,1), unpack=False)
                data = tmp[~np.isnan(tmp).any(axis=1)]  #all nanvalues are dropped. 
                fcn2min(fit_params, data)
                #print(data)
         
   
                out = minimize(fcn2min, fit_params, kws = {'data': data}, method='leastsq') 
                psi, delta = ell(out.params,data[:,0],wl)
                print(fit_report(out))

                plt.figure()
                plt.plot(data[:,0], np.asarray(psi[:])*180./np.pi)
                plt.plot(data[:,0], np.asarray(delta[:])*180./np.pi)
                plt.plot(data[:,0], data[:,1]*180./np.pi, 'ro', label='psi')
                plt.plot(data[:,0], data[:,2]*180./np.pi, 'bs', label='delta')
                plt.xlabel('Angle / deg')
                plt.ylabel('Psi, Delta / deg')
                plt.legend()
                plt.savefig(os.path.join(basepath, filename.split('_epd')[0] + '.pdf'))
                plt.close()
                
                outfile = 'fit-' + filename.split('_epd')[0] + '.dat'
                np.savetxt(os.path.join(basepath,outfile), np.transpose([data[:,0],  np.asarray(delta[:])*180./np.pi,  np.asarray(psi[:])*180./np.pi]), delimiter="\t")
                    
                
                with open(os.path.join(basepath,'fit_out.out'), 'a') as f:
                    f.write('{:s} \t {:.2f} \t {:.2f} \t {:.3f} \t {:.3f} \n'.format(filename.split('_p')[0], out.params['d1'].value, out.params['d1'].stderr,  out.params['n1'].value, out.params['n1'].stderr))
