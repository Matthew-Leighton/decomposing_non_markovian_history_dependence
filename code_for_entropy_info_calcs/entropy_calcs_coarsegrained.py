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

#Compute conditional entropy and dynamical information from the fine-grained data.

def merge_word_distributions(distribution_files):
    """
    Merge multiple word distribution files into a single distribution dictionary.
    
    Parameters:
    - distribution_files: List of filenames for the distribution .npz files
    
    Returns:
    - Dictionary containing the merged distributions
    """
    
    # Initialize a dictionary to hold merged counts for each k
    merged_counts = {}
    
    # Process each file
    for file_idx, file_name in enumerate(distribution_files):
        
        # Load the distributions from this file
        distributions = np.load(file_name, allow_pickle=True)
        
        # Find all k values in this file
        k_values = [key for key in distributions.keys() if key.startswith('k_')]
        
        # Process each k value
        for k_key in k_values:
            # Get the counts for this k
            counts = distributions[k_key].item()
            
            # Initialize this k in merged_counts if it doesn't exist
            if k_key not in merged_counts:
                merged_counts[k_key] = Counter()
            
            # Add these counts to the merged counts
            merged_counts[k_key].update(counts)
            
    
    # Convert merged counters to regular dictionaries for output
    output_dict = {}
    for k_key, counter in merged_counts.items():
        output_dict[k_key] = dict(counter)
    
    return output_dict


def calculate_entropy_measures(distributions, max_k=21):
    """
    Calculate both conditional entropy h_k and dynamical information I_k
    from the word distributions file.
    """
    
    # Initialize results dictionaries
    conditional_entropies = {}
    dynamical_info = {}
    
    # Calculate h_0 (unconditional entropy of a single bit)
    if 'k_1' in distributions:
        counts = distributions['k_1']
        zeros = counts.get('0', 0)
        ones = counts.get('1', 0)
        total = zeros + ones
        
        # Calculate probability of 0s and 1s
        p0 = zeros / total if zeros > 0 else 0
        p1 = ones / total if ones > 0 else 0
        
        # Calculate entropy
        h_0 = 0
        if p0 > 0:
            h_0 -= p0 * log2(p0)
        if p1 > 0:
            h_0 -= p1 * log2(p1)
        
        conditional_entropies[0] = h_0
    
    # Calculate entropy for higher k values
    for k in range(1, max_k): 
        # Check if we have required distributions
        if f'k_{k}' not in distributions or f'k_{k+1}' not in distributions:
            print(f"Skipping k={k}, missing required distributions")
            continue

        # Get word counts for lengths k and k+1
        k_counts = distributions[f'k_{k}']#.item()
        k_total = sum(k_counts.values())
        
        k_plus_1_counts = distributions[f'k_{k+1}']#.item()
        k_plus_1_total = sum(k_plus_1_counts.values())
        
        # Calculate H(X_k)
        h_k = 0
        for count in k_counts.values():
            p = count / k_total
            h_k -= p * log2(p)
        
        # Calculate H(X_{k+1})
        h_k_plus_1 = 0
        for count in k_plus_1_counts.values():
            p = count / k_plus_1_total
            h_k_plus_1 -= p * log2(p)
        
        # Conditional entropy h_k = H(X_{k+1}) - H(X_k)
        conditional_entropy = h_k_plus_1 - h_k
        conditional_entropies[k] = conditional_entropy
            
    # Calculate dynamical information I_k = h_{k-1} - h_k
    for k in range(1, max_k):
        if k-1 in conditional_entropies and k in conditional_entropies:
            info_k = conditional_entropies[k-1] - conditional_entropies[k]
            dynamical_info[k] = info_k
    
    return conditional_entropies, dynamical_info



# Define coarse-graining levels
cg_levels = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])

# Dictionary to store all results
all_results = {}

# Process each coarse-graining level
for cg_level in cg_levels:
    print(f"Processing coarse-graining level: {cg_level}")
    
    # Skip special case for level 1 (no coarse-graining needed)
    if cg_level == 1:
        distribution_files = []
        for i in np.arange(1, 48):
            if i != 7 and i != 45:
                # Load the word distributions for original (non-coarse-grained) data
                distribution_files.append(f'word_distributions_fly{i}.npz')
    else:
        # For coarse-grained levels
        distribution_files = []
        for i in np.arange(1, 48):
            if i != 7 and i != 45:
                # Load the word distributions for this coarse-grained level
                distribution_files.append(f'word_distributions_fly{i}_cg{cg_level}.npz')
    
    try:
        # Get the merged distributions for this level
        merged_distributions = merge_word_distributions(distribution_files)
        
        # Calculate entropy measures
        h_k, I_k = calculate_entropy_measures(merged_distributions, max_k=21)
        
        # Store results for this coarse-graining level
        all_results[cg_level] = {
            'h_k': h_k,
            'I_k': I_k
        }
        
        print(f"Successfully calculated entropy measures for level {cg_level}")
    except Exception as e:
        print(f"Error processing level {cg_level}: {e}")

# Save all results in a single file
np.savez_compressed('entropy_and_info_all_cg_levels.npz', results=all_results)



    