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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# Define coarse-graining level
cg_levels = np.array([1,2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,4096,8192])
k_vals = np.arange(1,11)
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from cmcrameri import cm as cmc  # pip install cmcrameri

def truncated_colors(name="viridis", N=14, lo=0.05, hi=0.75):
    cmap = cm.get_cmap(name)
    return [mcolors.to_hex(cmap(lo + (hi-lo)*i/(N-1))) for i in range(N)]
color_list = truncated_colors("tab20b", N=14, lo=0.0, hi=1)  # no yellow region


I_1_mean_list = []
I_k_sum_mean_list = []
I_k_sum_std_list = []
I_k_mean_list = np.zeros((len(cg_levels),20))
I_k_std_list = np.zeros((len(cg_levels),20))

for i in range(len(cg_levels)):

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


        for j in np.arange(1,20):
            if shuffled_I_k_corrected_mean[j] - shuffled_I_k_corrected_std[j]<0:
                I_k_corrected_mean[j]=np.nan
                I_k_corrected_std[j]=np.nan
    
    I_k_sum_mean = 0
    I_k_sum_std = 0
    
    for j in np.arange(1,10):
        I_k_sum_mean += I_k_corrected_mean[j]
        I_k_sum_std += I_k_corrected_std[j]
    

    I_k_sum_mean_list.append(I_k_sum_mean)
    I_k_sum_std_list.append(I_k_sum_std)
    I_1_mean_list.append(I_k_corrected_mean[0])

    

    I_k_mean_list[i,:] = I_k_corrected_mean
    I_k_std_list[i,:] = I_k_corrected_std



# Set up the figure
width = 8.6
height = 7
fig = plt.figure(figsize=(2*width/2.54, height/2.54))


gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1,1], wspace=0.15)


# ============================================================================
# PANEL A: coarse-graining factor
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

for k in range(10):
    if k>0:
        if k==1:
            ax1.errorbar(cg_levels, I_k_mean_list[:,k],yerr=I_k_std_list[:,k], fmt='o-',alpha=0.5, linewidth=2,clip_on=True,color=color_list[k],label=f'$k=${k+1}')
        elif k!=1:
            ax1.errorbar(cg_levels, I_k_mean_list[:,k],yerr=I_k_std_list[:,k], fmt='o-',alpha=0.5, linewidth=2,clip_on=True,color=color_list[k],label=f'{k+1}')


ax1.set_xlabel(r'Coarse-graining factor', fontsize=14)
ax1.set_ylabel(r'Dynamical Information $I_k$', fontsize=14)
#plt.ylabel(r'$I_k$ (bits)', fontsize=14)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(0.0008,0.1)
ax1.set_xlim(min(cg_levels),max(cg_levels))

ax1.legend(frameon=False,loc='best',fontsize=8)


# ============================================================================
# PANEL A: time
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

for k in range(10):
    if k>0:
        if k==1:
            ax2.errorbar(cg_levels*k*0.01, I_k_mean_list[:,k],yerr=I_k_std_list[:,k], fmt='o-',alpha=0.5, linewidth=2,clip_on=True,color=color_list[k],label=f'$k=${k+1}')
        elif k!=1:
            ax2.errorbar(cg_levels*k*0.01, I_k_mean_list[:,k],yerr=I_k_std_list[:,k], fmt='o-',alpha=0.5, linewidth=2,clip_on=True,color=color_list[k],label=f'{k+1}')


ax2.text(10,0.073,r'$\tau^*$',fontsize=14)
ax2.set_xlabel(r'Time $\tau$ (seconds)', fontsize=14)
#ax2.set_ylabel(r'Dynamical Information $I_k$', fontsize=14)
#plt.ylabel(r'$I_k$ (bits)', fontsize=14)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(0.0008,0.1)
ax2.set_xlim(0.02,1000)
ax2.vlines(7.4,0.0004,0.1,color='grey',alpha=0.7,lw=4,zorder=1)

#ax2.legend(frameon=False,loc='best',fontsize=8)


ax1.set_title('(a)',loc='left')
ax2.set_title('(b)',loc='left')

plt.tight_layout()
plt.savefig('SI_Fig_OrdervsCG_twopanel.pdf')
plt.show()
