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

def coarse_grain_random(binary_data, M, rng=None):
    """
    Temporal coarse-graining as in the Appendix:
    bin M consecutive timepoints and assign to each bin a
    randomly-selected state from within that bin.
    """
    x = np.asarray(binary_data)
    n = len(x)
    n_bins = n // M
    if n_bins == 0:
        raise ValueError(f"Sequence too short for coarse-graining factor M={M}")

    # Truncate to multiple of M
    x = x[:n_bins * M]

    # RNG for reproducibility
    if rng is None:
        rng = np.random.default_rng()

    # Choose one random index within each bin
    offsets = rng.integers(0, M, size=n_bins)
    idx = np.arange(n_bins) * M + offsets
    return x[idx]


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
k_values = np.unique(k_values)  # Remove duplicates

print(f"Number of unique k values: {len(k_values)}")
print(f"k range: {k_values[0]} to {k_values[-1]}")

# Coarse-graining factors: powers of 2 from 2 to 8192
coarse_graining_factors = 2 ** np.arange(1, 14, dtype=int)  # 2^1=2, ..., 2^13=8192

# Storage for results
fly_ids = []
cg_factors_list = []
autocorr_results = []

# Optional: RNG for reproducibility of coarse-graining
base_rng = np.random.default_rng(12345)

# Loop over coarse-graining factors and flies
for M in coarse_graining_factors:
    print(f"\n=== Coarse-graining factor M = {M} ===")

    for i in np.arange(1, 48):
        if i in (7, 45):  # Skip flies 7 and 45
            continue

        print(f"  Processing fly {i}")

        try:
            # Load the binary data (already 0/1)
            binary_data = np.load(f'fly{i}_binarized.npz')['data']

            # Skip if sequence too short for this M
            if len(binary_data) < M:
                print(f"    Skipping: length {len(binary_data)} < M={M}")
                continue

            # Fly- and M-specific RNG for deterministic coarse-graining
            rng = np.random.default_rng(seed=100000 * i + M)

            # Temporal coarse-graining
            cg_data = coarse_grain_random(binary_data, M, rng=rng)

            # Compute autocorrelation at log-spaced k values
            Ck = binary_autocorrelation_logspace(cg_data, k_values)

            # Store results
            fly_ids.append(i)
            cg_factors_list.append(M)
            autocorr_results.append(Ck)

        except Exception as e:
            print(f"    Error processing fly {i} at M={M}: {e}")
            continue

# Build a wide-format DataFrame:
# rows = (fly_id, coarse-graining factor), columns = k values
index = pd.MultiIndex.from_arrays(
    [fly_ids, cg_factors_list],
    names=['fly_id', 'cg_factor']
)
autocorr_df = pd.DataFrame(
    autocorr_results,
    index=index,
    columns=[f'k_{k}' for k in k_values]
)

# Build a long-format DataFrame: one row per (fly, M, k)
long_records = []
for row_idx, (fly_id, M) in enumerate(zip(fly_ids, cg_factors_list)):
    Ck = autocorr_results[row_idx]
    for k_idx, k in enumerate(k_values):
        long_records.append({
            'fly_id': fly_id,
            'cg_factor': M,
            'k': int(k),
            'autocorr': float(Ck[k_idx]),
        })

autocorr_long_df = pd.DataFrame(long_records)

# Save both DataFrames
autocorr_df.to_csv('fly_autocorrelations_coarse_grained_wide.csv')
autocorr_long_df.to_csv('fly_autocorrelations_coarse_grained_long.csv', index=False)

print(f"\nSaved autocorrelation data for {len(set(fly_ids))} flies")
print(f"Number of coarse-graining factors used: {len(np.unique(cg_factors_list))}")
print(f"Wide format shape: {autocorr_df.shape}")
print(f"Long format shape: {autocorr_long_df.shape}")

print("\nDataFrame preview (first 5 (fly, cg_factor) rows, first 10 k values):")
print(autocorr_df.iloc[:5, :10])

print(f"\nk values used: {len(k_values)} points from {k_values[0]} to {k_values[-1]}")
print(f"Example k values: {k_values[:10]}...")
print(f"Example k values (end): ...{k_values[-10:]}")
