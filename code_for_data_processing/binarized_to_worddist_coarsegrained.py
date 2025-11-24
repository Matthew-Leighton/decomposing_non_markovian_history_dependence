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
from numpy import random


plt.rcParams['text.usetex'] = True
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


plt.style.use('Leighton_Style_Colors2')


#Takes in Binarized behaviour time series and returns "word" distributions at various levels of temporal coarse-graining.


def coarse_grain_temporal_random(binary_sequence, factor):
    """
    Temporally coarse-grain a binary sequence by a factor 'factor',
    selecting a random element from each block to preserve the marginal distribution.
    
    Parameters:
    - binary_sequence: NumPy array containing the binary sequence (0s and 1s)
    - factor: Coarse-graining factor (should be a power of 2)
    
    Returns:
    - NumPy array containing the coarse-grained sequence
    """
    # Check if factor is a power of 2
    if not (factor & (factor - 1) == 0) or factor <= 0:
        raise ValueError("Coarse-graining factor must be a positive power of 2")
    
    # Calculate the length of the coarse-grained sequence
    original_length = len(binary_sequence)
    cg_length = original_length // factor
    
    # Create the coarse-grained sequence
    coarse_grained = np.zeros(cg_length, dtype=binary_sequence.dtype)
    
    # For each block, select a random element
    for i in range(cg_length):
        block_start = i * factor
        block_end = block_start + factor
        
        # Get the block and select a random index within it
        block = binary_sequence[block_start:block_end]
        random_index = random.randint(0, len(block) - 1)
        
        # Use the randomly selected element for the coarse-grained sequence
        coarse_grained[i] = block[random_index]
    
    return coarse_grained


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



cg_levels = np.array([2,4,8,16,32,64,128,256,512,1024,2048,4096,8192])

for j in cg_levels:
    print(j)

    for i in np.arange(1,48):
        if i!=7 and i!= 45:
            print(i)
            # Load your binary data
            binary_data = np.load('fly'+str(i)+'_binarized.npz')['data']  # Adjust filename as needed
            
            cg_sequence = coarse_grain_temporal_random(binary_data, j)

            # Analyze word distributions
            word_distributions = analyze_binary_words(cg_sequence, max_k=21)
            
            # Save results efficiently
            output = {}
            for k, counter in word_distributions.items():
                # Convert to more efficient representation
                # Store as {word: count} dictionary
                output[f'k_{k}'] = dict(counter)
            
            # Save as compressed NumPy file
            np.savez_compressed('word_distributions_fly'+str(i)+'_cg'+str(j)+'.npz', **output)
