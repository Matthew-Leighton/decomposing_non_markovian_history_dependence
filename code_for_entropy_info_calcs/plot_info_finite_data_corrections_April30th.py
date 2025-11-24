
# Load all the packages I might need
from __future__ import division # must be first
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import matplotlib
from matplotlib import rc
from scipy.io import loadmat
from scipy.integrate import odeint
from scipy.integrate import quad
from mpmath import gammainc
import numbers
from Bio import Entrez, SeqIO
from collections import defaultdict, Counter
import pandas as pd
import time
from math import log2
from scipy import optimize


plt.rcParams['text.usetex'] = True
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.style.use('Leighton_Style_Colors2')

# No Finite Data Corrections:
uncorrected_data = np.load('entropy_and_info_all_cg_levels.npz', allow_pickle=True)
uncorrected_results = uncorrected_data['results'].item()

corrected_data = np.load('info_cg_finitedata.npz', allow_pickle=True)
#corrected_data = np.load('info_cg_finitedata_shuffled.npz', allow_pickle=True)
corrected_results = corrected_data['results'].item()

shuffled_corrected_data = np.load('info_cg_finitedata_shuffled.npz', allow_pickle=True)
shuffled_corrected_results = shuffled_corrected_data['results'].item()


# Set up the figure with a 4x5 grid of subplots
width = 8.6
height = 7
fig, axes = plt.subplots(4, 4, figsize=(4*width/2.54, 3*height/2.54))
axes = axes.flatten()  # Flatten for easier indexing

# Define coarse-graining level
cg_levels = np.array([1,2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,4096,8192])
k_vals = np.arange(1,21)
color_list = ['black','darkviolet','purple','slateblue','blue','cyan','green','springgreen','yellow','orange','red','firebrick','brown','grey']


for i in range(len(cg_levels)):
    ax = axes[i]

    cg_level = cg_levels[i]

    #Uncorrected:
    I_k_uncorrected = uncorrected_results[cg_level]['I_k']

    #Corrected:
    I_k_corrected_mean = corrected_results[cg_level]['I_k_mean']
    I_k_corrected_std = corrected_results[cg_level]['I_k_std']

    counter=0
    for j in np.arange(1,20):
        if I_k_corrected_mean[j] - I_k_corrected_std[j]<0 or counter>0:
            I_k_corrected_mean[j]=np.nan
            I_k_corrected_std[j]=np.nan
            counter=1

    
    if i>1: #Remove corrected points if the shuffled info is significantly different from zero
        #Shuffled Corrected:
        shuffled_I_k_corrected_mean = shuffled_corrected_results[cg_level]['I_k_mean']
        shuffled_I_k_corrected_std = shuffled_corrected_results[cg_level]['I_k_std']

        pizza=0
        for j in np.arange(1,20):
            if shuffled_I_k_corrected_mean[j] - shuffled_I_k_corrected_std[j]<0 or pizza==1:
                I_k_corrected_mean[j]=np.nan
                I_k_corrected_std[j]=np.nan
                pizza=1
    
    ax.plot(k_vals, list(I_k_uncorrected.values()), 'o-',alpha=0.5, linewidth=2,clip_on=True,label='Uncorrected',color='blue')
    ax.errorbar(k_vals, I_k_corrected_mean,yerr=I_k_corrected_std, fmt='o-',alpha=0.5, linewidth=2,clip_on=True,label='Corrected',color='green')

    ax.set_xlabel(r'$k$', fontsize=14)
    ax.set_ylabel(r'$I_k$ (bits)', fontsize=14)
    ax.set_yscale('log')
    ax.set_title(f'Coarse-graining factor: {cg_level}')
    ax.set_ylim(0.00001,1)
    ax.set_xlim(1,20)


ax = axes[14]

for i in range(len(cg_levels)):
    cg = cg_levels[i]
    I_k = uncorrected_results[cg]['I_k']

    # Plot dynamical information
    ax.plot(list(I_k.keys()), list(I_k.values()), 'o-',alpha=0.5, linewidth=2,clip_on=True,label=str(cg)+'x',color=color_list[i])

    ax.set_xlabel(r'$k$', fontsize=14)
    ax.set_ylabel(r'$I_k$ (bits)', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(0.00001,1)
    ax.set_xlim(1,20)
    ax.set_title('Uncorrected')
    #ax.legend(fontsize=8)

ax = axes[15]

for i in range(len(cg_levels)):
    cg_level = cg_levels[i]
    #Corrected:
    I_k_corrected_mean = corrected_results[cg_level]['I_k_mean']
    I_k_corrected_std = corrected_results[cg_level]['I_k_std']
    # Plot dynamical information
    ax.errorbar(k_vals, I_k_corrected_mean,yerr=I_k_corrected_std, fmt='o-',alpha=0.5, linewidth=2,clip_on=True,label=str(cg)+'x',color=color_list[i])

    ax.set_xlabel(r'$k$', fontsize=14)
    ax.set_ylabel(r'$I_k$ (bits)', fontsize=14)
    ax.set_yscale('log')
    #ax.set_xscale('log')

    ax.set_ylim(0.00001,1)
    ax.set_xlim(1,20)
    ax.set_title('Corrected')


#plt.gca().xaxis.set_minor_locator(plt.NullLocator())
#plt.gca().yaxis.set_minor_locator(plt.NullLocator())



plt.tight_layout()

plt.show()