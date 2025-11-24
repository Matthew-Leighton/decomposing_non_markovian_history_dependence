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

from scipy import stats
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


# Set up the figure
width = 8.6
height = 7
fig = plt.figure(figsize=(1.2*width/2.54, height/2.54))


# Create subplot layout (1 row, 3 columns)
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.35)



##### Dynamical information calculations

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
k_vals = np.arange(1,21)
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from cmcrameri import cm as cmc  # pip install cmcrameri

def truncated_colors(name="viridis", N=14, lo=0.05, hi=0.75):
    cmap = cm.get_cmap(name)
    return [mcolors.to_hex(cmap(lo + (hi-lo)*i/(N-1))) for i in range(N)]
color_list = truncated_colors("tab20b", N=14, lo=0.0, hi=1)  # no yellow region


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

        pizza=0
        for j in np.arange(1,20):
            if shuffled_I_k_corrected_mean[j] - shuffled_I_k_corrected_std[j]<0 or pizza==1:
                I_k_corrected_mean[j]=np.nan
                I_k_corrected_std[j]=np.nan
                pizza=1



# ============================================================================
# PANEL B: Dynamical Information
# ============================================================================
ax2 = fig.add_subplot(gs[0, 0])

slope_list = []
slope_se_list = []
r_squared_list = []
for i in range(len(cg_levels)):
    cg_level = cg_levels[i]
    #Corrected:
    I_k_corrected_mean = corrected_results[cg_level]['I_k_mean']
    I_k_corrected_std = corrected_results[cg_level]['I_k_std']
    # Plot dynamical information
    ax2.errorbar(k_vals*cg_level / 100, I_k_corrected_mean,yerr=I_k_corrected_std, fmt='o-',alpha=0.4, linewidth=2,clip_on=True,label=str(cg_level)+'x',color=color_list[i])

    results = fit_power_law(k_vals[2:]*cg_level / 100, I_k_corrected_mean[2:])
    slope = results['alpha']
    intercept = results['C']
    rsquared = results['r_squared']
    slope_se = results['alpha_se']
    slope_list.append(slope)
    slope_se_list.append(2*slope_se)
    r_squared_list.append(rsquared)

    if i >1:
        x_list = k_vals*cg_level / 100

        ax2.plot(x_list,intercept*(x_list**slope),color='black',ls='--',lw=1,zorder=12)


#ax2.plot(x_list,10*x_list**(-2.672),color='black',ls='--',lw=1.5)
#ax2.plot(x_list,100*x_list**(-2.672),color='black',ls='--',lw=1.5)
#ax2.plot(x_list,1000*x_list**(-2.672),color='black',ls='--',lw=1.5)



ax2.set_xlabel(r'Lag $\tau$ (seconds)', fontsize=14)
ax2.set_ylabel(r'Dynamical Information $I_k$', fontsize=14)
ax2.set_yscale('log')
ax2.set_xscale('log')

ax2.set_ylim(0.000001,1)
ax2.set_xlim(0.01,1000)

ax2.legend(frameon=False,loc='lower left', bbox_to_anchor=(1, 0),fontsize=8)

ax2.text(30,0.3,r'$\sim k^{\gamma}$',fontsize=14)
ax2.text(1300,0.45,r'\underline{cg factor}',fontsize=10)



# Create inset plot in top right corner using inset_axes
inset_ax = inset_axes(ax2, width="40%", height="20%", loc='lower right', bbox_to_anchor=(0.01, 0.13, 1, 1), bbox_transform=ax2.transAxes)

for i in range(len(cg_levels)):
    cg_level = cg_levels[i]
    if i>=2:
        inset_ax.errorbar(cg_level,slope_list[i],yerr=slope_se_list[i],fmt='o-',alpha=r_squared_list[i], lw=1,clip_on=False,color='black',markersize=2.5)

inset_ax.hlines(np.mean([ -2.578832440601971, -2.667301999956243, -2.69947319737874, -2.6724207153622257, -2.545817964156472]),1,10000,color='grey',alpha=0.5,ls=':',lw=1.5)
inset_ax.tick_params(labelsize=6,which='both')

inset_ax.set_xlabel(r'cg level', fontsize=10)
inset_ax.set_ylabel(r'$\gamma$', fontsize=10)
inset_ax.set_xscale('log')
inset_ax.set_ylim(-3,-1)
inset_ax.set_xlim(3,8192*1.5)



plt.tight_layout()
plt.savefig('SI_Fig_PowerLaws.pdf')
plt.show()
