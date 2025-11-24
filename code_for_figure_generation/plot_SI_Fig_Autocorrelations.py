from __future__ import division  # must be first
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import matplotlib
from scipy.io import loadmat
from scipy.integrate import odeint, quad
from mpmath import gammainc
import numbers
from Bio import Entrez, SeqIO
from collections import defaultdict, Counter
import pandas as pd
import time
from scipy import stats

import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------
# Global plotting style (match your figure)
# ---------------------------------------------------------------------
plt.rcParams['text.usetex'] = True
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.style.use('Leighton_Style_Colors2')



def fit_power_law(x, y):
    
    # Remove zeros and negative values before log transform
    mask = (x > 0) & (y > 0)
    x_clean = np.asarray(x)[mask]
    y_clean = np.asarray(y)[mask]
    
    # Log transform the data
    log_x = np.log(x_clean)
    log_y = np.log(y_clean)
    
    n = len(log_x)
    
    # Calculate means
    mean_log_x = np.mean(log_x)
    mean_log_y = np.mean(log_y)
    
    # Calculate slope and intercept
    ss_xx = np.sum((log_x - mean_log_x)**2)
    ss_xy = np.sum((log_x - mean_log_x) * (log_y - mean_log_y))
    alpha = ss_xy / ss_xx
    
    log_C = mean_log_y - alpha * mean_log_x
    C = np.exp(log_C)
    
    # Calculate residuals and standard error
    y_pred = alpha * log_x + log_C
    residuals = log_y - y_pred
    mse = np.sum(residuals**2) / (n - 2)  # degrees of freedom = n - 2
    
    # Standard error of the slope
    se_alpha = np.sqrt(mse / ss_xx)
    
    # 95% confidence interval
    t_val = stats.t.ppf(0.975, n - 2)  # 97.5th percentile for 95% CI
    alpha_ci = alpha + np.array([-1, 1]) * t_val * se_alpha
    
    r, _ = stats.pearsonr(log_x, log_y)
    r_squared = r**2

    return {
        'C': C,
        'alpha': alpha,
        'alpha_se': se_alpha,
        'alpha_ci_95': alpha_ci,
        'r_squared': r_squared
    }


# ---------------------------------------------------------------------
# Coarse-graining levels and color scheme (same as your code)
# ---------------------------------------------------------------------
cg_levels = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])

def truncated_colors(name="viridis", N=14, lo=0.05, hi=0.75):
    cmap = cm.get_cmap(name)
    return [mcolors.to_hex(cmap(lo + (hi - lo) * i / (N - 1))) for i in range(N)]

# tab20b truncated, no yellow region
color_list = truncated_colors("tab20b", N=len(cg_levels), lo=0.0, hi=1.0)
# CG factor 1 in grey
color_list[0] = 'grey'

cg_to_color = {cg_levels[i]: color_list[i] for i in range(len(cg_levels))}

# ---------------------------------------------------------------------
# Load autocorrelation data
#   CG = 1: original
#   CG >= 2: coarse-grained
# ---------------------------------------------------------------------
orig_df = pd.read_csv('fly_autocorrelations_wide.csv', index_col=0)
cg_df   = pd.read_csv('fly_autocorrelations_coarse_grained_wide.csv')

# Identify k columns and extract numeric k values
k_cols = sorted([col for col in orig_df.columns if col.startswith('k_')],
                key=lambda c: int(c.split('_')[1]))
k_values = np.array([int(col.split('_')[1]) for col in k_cols])

# ---------------------------------------------------------------------
# Figure + GridSpec (1 main panel + skinny legend column)
# ---------------------------------------------------------------------
width = 8.6
height = 7
fig = plt.figure(figsize=(1.2 * width / 2.54, height / 2.54))

gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.35)
ax = fig.add_subplot(gs[0, 0])

# ---------------------------------------------------------------------
# CG = 1 (no coarse-graining), plotted vs k
# ---------------------------------------------------------------------
mean_orig = orig_df[k_cols].mean(axis=0).values
std_orig  = orig_df[k_cols].std(axis=0).values

ax.plot(
    k_values,
    mean_orig,
    linewidth=2,ls='-',
    color=cg_to_color[1],
    label='1x'
)
ax.fill_between(
    k_values,
    mean_orig - std_orig,
    mean_orig + std_orig,
    alpha=0.3,
    color=cg_to_color[1]
)

# ---------------------------------------------------------------------
# CG >= 2, same colors and error regions, also vs k
# ---------------------------------------------------------------------
for cg_level in cg_levels:
    if cg_level == 1:
        continue

    group = cg_df[cg_df['cg_factor'] == cg_level]
    if group.empty:
        continue

    mean_autocorr = group[k_cols].mean(axis=0).values
    std_autocorr  = group[k_cols].std(axis=0).values

    color = cg_to_color[cg_level]

    ax.plot(
        k_values,
        mean_autocorr,
        linewidth=1.5,ls='-',
        label=f'{cg_level}x',
        color=color
    )
    ax.fill_between(
        k_values,
        mean_autocorr - std_autocorr,
        mean_autocorr + std_autocorr,
        alpha=0.2,
        color=color
    )

# ---------------------------------------------------------------------
# Power-law reference ~ k^{-0.18}
# ---------------------------------------------------------------------
# Load the autocorrelation data
autocorr_df = pd.read_csv('fly_autocorrelations_wide.csv', index_col=0)

# Extract k values from column names (remove 'k_' prefix)
k_values = np.array([int(col.split('_')[1]) for col in autocorr_df.columns])

# Calculate mean and standard deviation across all flies
mean_autocorr = autocorr_df.mean(axis=0).values
std_autocorr = autocorr_df.std(axis=0).values

k_values = np.array(k_values)

results = fit_power_law(k_values[9:594], mean_autocorr[9:594])
slope = results['alpha']
intercept = results['C']
R2 = results['r_squared']
se = results['alpha_se']
print(R2)
print(se)
#print(slope)
ax.plot(k_values,intercept*(k_values)**(slope),color='black',ls='--',lw=1.5)
ax.text(2000, 0.5, r'$\sim k^{-0.18}$', fontsize=14)

# ---------------------------------------------------------------------
# Formatting to match your style, but with x = k
# ---------------------------------------------------------------------
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(1, 100_000)
ax.set_ylim(0.1, 1)

ax.set_xlabel(r'Lag $k$', fontsize=14)
ax.set_ylabel(r'Autocorrelation $C(k)$', fontsize=14)

ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.6, 1],labels=['0.1', '0.2', '0.3', '0.4', '0.6', '1'])

# Legend to the right in the skinny column
ax.legend(frameon=False,
          loc='lower left',
          bbox_to_anchor=(1, 0),
          fontsize=8)

# Underlined cg factor label, similar to your figure
ax.text(130000, 0.85, r'\underline{CG factor}', fontsize=10)

# Optional panel label if you want it:
# ax.set_title('a)', loc='left')

plt.tight_layout()
plt.savefig('SI_Fig_Autocorr_vs_k_cg.pdf')
plt.show()
