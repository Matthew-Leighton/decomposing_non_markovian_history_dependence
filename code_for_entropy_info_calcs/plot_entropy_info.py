#Load all the packages I might need
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


plt.rcParams['text.usetex'] = True
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


plt.style.use('Leighton_Style_Colors2')


def load_entropy_from_npz(filename):
    """
    Load conditional entropy and dynamical information from a NumPy npz file.
    
    Parameters:
    - filename: Path to the npz file
    
    Returns:
    - h_k: Dictionary of conditional entropy values
    - I_k: Dictionary of dynamical information values
    """
    # Load the compressed NumPy file
    data = np.load(filename, allow_pickle=True)
    
    # Extract the dictionaries
    h_k = data['h_k'].item() if 'h_k' in data else {}
    I_k = data['I_k'].item() if 'I_k' in data else {}
    
    return h_k, I_k


all_data = np.load('entropy_and_info_all_cg_levels.npz', allow_pickle=True)
all_results = all_data['results'].item()


width = 8.6
height = 7
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(1.5*width/2.54,2*height/2.54))#, sharex=True)

# Define coarse-graining levels
cg_levels = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
color_list = ['black','purple','slateblue','blue','green','springgreen','yellow','orange','red','firebrick','brown','grey']

for i in range(len(cg_levels)):
	cg = cg_levels[i]
	h_k = all_results[cg]['h_k']
	I_k = all_results[cg]['I_k']

	# Plot conditional entropy
	ax1.plot(list(h_k.keys()), list(h_k.values()), 'o-',alpha=0.5, linewidth=2,clip_on=False,label=str(cg)+'x',color=color_list[i])

	# Plot dynamical information
	ax2.plot(list(I_k.keys()), list(I_k.values()), 'o-',alpha=0.5, linewidth=2,clip_on=False,label=str(cg)+'x',color=color_list[i])


ax1.set_ylabel(r'$h_k$ (bits)', fontsize=14)

ax2.set_xlabel(r'$k$', fontsize=14)
ax2.set_ylabel(r'$I_k$ (bits)', fontsize=14)

plt.gca().xaxis.set_minor_locator(plt.NullLocator())
plt.gca().yaxis.set_minor_locator(plt.NullLocator())
#ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_ylim(0,1)
ax2.set_ylim(0.00001,1)

#ax2.set_xscale('log')

#k_list = np.arange(1,21)
#ktilde_list = 0.920589*k_list - np.ones(20)*17.6861794
#I_list = np.exp(-2**(-ktilde_list)) - np.exp(-2**(np.ones(20)-ktilde_list))
#plt.plot(k_list,I_list,color='black',alpha=0.9,zorder=1,ls=':',lw=1)


ax1.legend(frameon=False,loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)
#ax2.legend(frameon=False,loc='upper left', bbox_to_anchor=(1, 1))

plt.xticks(range(0, 21))
plt.xlim(0,20)

plt.tight_layout()
plt.show()
