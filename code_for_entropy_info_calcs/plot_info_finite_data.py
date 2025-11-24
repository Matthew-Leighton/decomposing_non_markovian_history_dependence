
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


# Define the quadratic function
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

I_k_corrected=[]
I_k_uncorrected=[]

# Load the data for partial data fractions
cg_level = 64
all_data = np.load(f'info_cg{cg_level}_finitedata.npz', allow_pickle=True)
results = all_data['results'].item()

# Load the full data version (reference to your entropy calculations with all flies)
data = np.load('entropy_and_info_all_cg_levels.npz', allow_pickle=True)
all_results = data['results'].item()
h_k = all_results[cg_level]['h_k']
I_k = all_results[cg_level]['I_k']


# Convert to proper dictionaries if needed
if not isinstance(h_k, dict):
    h_k = h_k.item()
if not isinstance(I_k, dict):
    I_k = I_k.item()

# Define data fractions and prepare for plotting
data_fractions = np.array(sorted([int(frac) for frac in results.keys()]))
full_data_fraction = 45  # Total number of flies

# Convert to 1/data_fraction for x-axis
inv_data_fractions = 45 / data_fractions

inv_data_fractions = np.concatenate((inv_data_fractions,np.ones(1)))

k_values = np.arange(1, 21)

# Set up the figure with a 4x5 grid of subplots
width = 8.6
height = 7
fig, axes = plt.subplots(4, 5, figsize=(3*width/2.54, 2.4*height/2.54), sharex=True)
axes = axes.flatten()  # Flatten for easier indexing

# Create plots for each k value
for i, k in enumerate(k_values):
    ax = axes[i]
    
    # Extract mean and std for this k value across all data fractions
    means = []
    stds = []

    for frac in data_fractions:
        means.append(results[frac]['I_k_mean'][k-1])
        stds.append(results[frac]['I_k_std'][k-1])

    means.append(I_k[k])
    stds.append(0.0001*I_k[k])
    
    means = np.array(means)
    stds = np.array(stds)

    # Weighted least squares fit
    # The sigma parameter incorporates the uncertainties
    params, pcov = optimize.curve_fit(
        quadratic, 
        inv_data_fractions, 
        means,
        sigma=stds,absolute_sigma=True)

    # Extract the fitted parameters
    a_fit, b_fit, c_fit = params

    I_k_corrected.append(quadratic(0, a_fit, b_fit, c_fit))
    I_k_uncorrected.append(I_k[k])

    x_fit = np.linspace(0, 6, 100)
    y_fit = quadratic(x_fit, a_fit, b_fit, c_fit)
    ax.plot(x_fit, y_fit,lw=1,ls='--',color='black',alpha=0.8)
    ax.plot(x_fit, np.ones(100)*I_k[k],lw=1,ls=':',color='black',alpha=0.8)
    
    # Plot mean with error bars for partial data
    ax.errorbar(inv_data_fractions, means, yerr=stds, fmt='o-', linewidth=2, capsize=3, clip_on=False,alpha=0.7)
    
        
    # Set title and adjust appearance
    ax.set_title(f'$k = {k}$', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Only add y-label to leftmost plots
    if i % 5 == 0:
        ax.set_ylabel('$I_k$ (bits)', fontsize=10)
    
    # Only add x-label to bottom plots
    if i >= 15:
        ax.set_xlabel('Data Fraction', fontsize=10)  # n = data fraction
    
    # Set y-scale to log for better visualization of small values
    #ax.set_yscale('log')

    min_y = min(np.min(y_fit),np.min(means-stds))
    max_y = max(np.max(y_fit),max(means+stds))
    
    ax.set_ylim(0.9*min_y,max_y*1.1)
    ax.set_xlim(0,6)
    
    # Remove minor ticks for cleaner appearance
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())


# Fine-tune the layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.3)

plt.show()




width = 8.6
height = 7
# Create a figure with two subplots
fig= plt.figure(figsize=(width/2.54,height/2.54))#, sharex=True)


plt.plot(k_values, I_k_uncorrected, 'o-',alpha=0.5, linewidth=2,label='Uncorrected',color='blue')
plt.plot(k_values, I_k_corrected, 'o-',alpha=0.5, linewidth=2,label='Corrected',color='green')



plt.xlabel(r'$k$', fontsize=14)
plt.ylabel(r'$I_k$ (bits)', fontsize=14)

plt.gca().xaxis.set_minor_locator(plt.NullLocator())
plt.gca().yaxis.set_minor_locator(plt.NullLocator())
plt.yscale('log')
plt.ylim(0.00001,1)


plt.legend(frameon=False)

plt.xticks(range(1, 21))
plt.xlim(1,20)

plt.tight_layout()
plt.show()

