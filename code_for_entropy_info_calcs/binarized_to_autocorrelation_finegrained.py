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

def binary_autocorrelation_logspace(binary_data, k_values):
    """
    Compute normalized autocorrelation for binary (0/1) data at specific lag values.
    
    Parameters:
    -----------
    binary_data : numpy.ndarray
        Binary array of 0s and 1s
    k_values : numpy.ndarray
        Specific lag values to compute (must be integers)
        
    Returns:
    --------
    numpy.ndarray
        Normalized autocorrelation values C(k) for specified k values
        where C(0)=1 and C(∞)→0
    """
    # Convert to numpy array if not already
    x = np.asarray(binary_data, dtype=float)
    
    # Ensure we're working with mean-centered data for proper normalization
    x_mean = np.mean(x)
    x_centered = x - x_mean
    
    # Compute variance (denominator for normalization)
    variance = np.var(x)
    if variance == 0:
        return np.zeros(len(k_values))  # Handle constant sequences
    
    n = len(x)
    
    # Filter k_values to be within valid range and convert to integers
    k_values = np.asarray(k_values, dtype=int)
    valid_k = k_values[k_values < n]
    
    # FFT-based computation for efficiency
    fft_len = 2 ** np.ceil(np.log2(2 * n - 1)).astype(int)
    
    # Compute FFT and power spectrum
    fft_x = np.fft.fft(x_centered, fft_len)
    power_spectrum = np.abs(fft_x) ** 2
    
    # Inverse FFT to get full autocorrelation
    autocorr_full = np.fft.ifft(power_spectrum).real
    
    # Extract autocorrelation at specified k values and normalize
    autocorr = np.zeros(len(k_values))
    for i, k in enumerate(k_values):
        if k < n:
            autocorr[i] = autocorr_full[k] / (variance * n)
        else:
            autocorr[i] = 0  # Set to 0 for k values beyond data length
    
    return autocorr

# Create log-uniform spaced k values from 10^0 to 10^6 with 1000 entries
k_values = np.logspace(0, 6, 1000, dtype=int)
# Remove duplicates that might occur due to integer conversion at small values
k_values = np.unique(k_values)

print(f"Number of unique k values: {len(k_values)}")
print(f"k range: {k_values[0]} to {k_values[-1]}")

# Initialize lists to store results
fly_ids = []
autocorr_results = []

width = 8.6
height = 7
fig = plt.figure(figsize=(1.2*width/2.54, 1.2*height/2.54))

# Process each fly's data
for i in np.arange(1, 48):
    if i != 7 and i != 45:  # Skip flies 7 and 45
        print(f"Processing fly {i}")
        
        try:
            # Load the binary data
            binary_data = np.load('fly'+str(i)+'_binarized.npz')['data']
            
            # Compute autocorrelation at log-spaced k values
            Ck = binary_autocorrelation_logspace(binary_data, k_values)
            
            # Store results
            fly_ids.append(i)
            autocorr_results.append(Ck)
            
            # Plot
            plt.plot(k_values, Ck, label=f'Fly {i}', alpha=0.7)
            
        except Exception as e:
            print(f"Error processing fly {i}: {e}")
            continue

# Create DataFrame with each fly's autocorrelation function
# Each row will be a fly, each column will be a k value
autocorr_df = pd.DataFrame(autocorr_results, 
                          index=[f'fly_{id}' for id in fly_ids],
                          columns=[f'k_{k}' for k in k_values])

# Also create a long-format DataFrame for easier analysis
long_format_data = []
for fly_idx, fly_id in enumerate(fly_ids):
    for k_idx, k in enumerate(k_values):
        long_format_data.append({
            'fly_id': fly_id,
            'k': k,
            'autocorr': autocorr_results[fly_idx][k_idx]
        })

autocorr_long_df = pd.DataFrame(long_format_data)

# Save both DataFrames
autocorr_df.to_csv('fly_autocorrelations_wide.csv')
autocorr_long_df.to_csv('fly_autocorrelations_long.csv', index=False)

print(f"\nSaved autocorrelation data for {len(fly_ids)} flies")
print(f"Wide format: {autocorr_df.shape}")
print(f"Long format: {autocorr_long_df.shape}")

# Finalize plot
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k$')
plt.ylabel(r'$C(k)$')
plt.legend(loc='best', fontsize=6)  # Smaller font for many flies
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fly_autocorrelations_logspace.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Display summary statistics
print("\nDataFrame preview (first 5 flies, first 10 k values):")
print(autocorr_df.iloc[:5, :10])

print(f"\nk values used: {len(k_values)} points from {k_values[0]} to {k_values[-1]}")
print(f"Example k values: {k_values[:10]}...")
print(f"Example k values (end): ...{k_values[-10:]}")