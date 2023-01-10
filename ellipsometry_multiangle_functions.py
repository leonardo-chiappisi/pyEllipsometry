# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:36:03 2022

@author: Leonardo Chiappisi (Institut Laue-Langevin)
"""

import sys
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
sys.path.insert(0, './tmm-0.1.7')
import tmm_core as tmm
degree = np.pi/180

def ell(pars,ang,wl):
    '''takes refractive index and layer thickness and returns psi, delta'''
    vals = pars.valuesdict()
    ns = [vals[key] for key in vals if 'n' in key] #List of refractive indices, starting from air to the substrate
    ds = [vals[key] for key in vals if 'd' in key] #List of refractive indices, starting from air to the substrate
    psi = np.array([tmm.ellips(ns, ds, i*degree, wl)['psi'] for i in ang])
    delta = np.pi+np.array([tmm.ellips(ns, ds, i*degree, wl)['Delta'] for i in ang]) #in nm
    return psi, delta

def ell_xy(pars,ang,wl):
    '''takes refractive index and layer thickness and returns x and y'''
    vals = pars.valuesdict()
    ns = [vals[key] for key in vals if 'n' in key] #List of refractive indices, starting from air to the substrate
    ds = [vals[key] for key in vals if 'd' in key] #List of refractive indices, starting from air to the substrate
    x = np.array([tmm.ellips_xy(ns, ds, i*degree, wl)['x'] for i in ang])
    y = np.array([tmm.ellips_xy(ns, ds, i*degree, wl)['y'] for i in ang]) #in nm
    return x, y