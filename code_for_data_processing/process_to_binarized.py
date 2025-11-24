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


plt.rcParams['text.usetex'] = True
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


plt.style.use('Leighton_Style_Colors2')


### Binarize fly data into 0s and 1s


for i in np.arange(1,48):
	if i!=7 and i!=45:
	    # Load the CSV data (assumes a single column of numbers)
	    data = pd.read_csv('data_fly'+str(i)+'.csv', header=None)
	    
	    # Convert to a 1D array
	    values = data.values.flatten()
	    coarse_grained = np.where(np.logical_or(values == 0, values == 1), 0, 1)
	    
	    # Save as compressed binary instead of text
	    np.savez_compressed('fly'+str(i)+'_binarized.npz', data=coarse_grained)
