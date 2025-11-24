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


##### Converts binary time-series data into distributions of "words" (sequences) of length up to k=21. 


def analyze_binary_words(binary_array, max_k=21):
    """
    Analyze the distribution of binary words up to length max_k
    in the given binary array.
    
    Returns a dictionary of Counters, one for each word length.
    """
    results = {}
    
    # Pre-compute string representation once for efficiency
    str_array = ''.join(map(str, binary_array))
    
    for k in range(1, max_k + 1):
        
        # Create sliding windows of length k
        words = [str_array[i:i+k] for i in range(len(binary_array) - k + 1)]
        
        # Count occurrences of each word
        word_counts = Counter(words)
        
        # Store results
        results[k] = word_counts
    return results


for i in np.arange(1,48):
    if i!=7 and i!= 45:
        print(i)
        # Load your binary data
        binary_data = np.load('fly'+str(i)+'_binarized.npz')['data']  # Adjust filename as needed
        
        # Analyze word distributions
        word_distributions = analyze_binary_words(binary_data, max_k=21)
        
        # Save results efficiently
        output = {}
        for k, counter in word_distributions.items():
            # Convert to more efficient representation
            # Store as {word: count} dictionary
            output[f'k_{k}'] = dict(counter)
        
        # Save as compressed NumPy file
        np.savez_compressed('word_distributions_fly'+str(i)+'.npz', **output)
