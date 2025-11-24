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
fig = plt.figure(figsize=(4*width/2.54, height/2.54))


# Create subplot layout (1 row, 3 columns)
gs = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[0.5, 1, 0.05, 1,1], wspace=0.4)


# ============================================================================
# PANEL A: Autocorrelation function
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])


# Load the autocorrelation data
autocorr_df = pd.read_csv('fly_autocorrelations_wide.csv', index_col=0)

# Extract k values from column names (remove 'k_' prefix)
k_values = np.array([int(col.split('_')[1]) for col in autocorr_df.columns])

# Calculate mean and standard deviation across all flies
mean_autocorr = autocorr_df.mean(axis=0).values
std_autocorr = autocorr_df.std(axis=0).values

k_values = np.array(k_values)/100


ax1.plot(k_values, mean_autocorr, linewidth=2, color='grey', label='Mean')
ax1.fill_between(k_values, mean_autocorr - std_autocorr, mean_autocorr + std_autocorr, alpha=0.3, color='grey')


results = fit_power_law(k_values[9:594], mean_autocorr[9:594])
slope = results['alpha']
intercept = results['C']
R2 = results['r_squared']
se = results['alpha_se']
print(R2)
print(se)
#print(slope)
ax1.plot(k_values,intercept*(k_values)**(slope),color='black',ls='--',lw=1.5)

ax1.text(2,0.7,r'$\sim k^{-0.18}$',fontsize=14)



# Set log-log scale and labels
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(0.1,1)
ax1.set_xlim(1/100,100000/100)
ax1.set_xlabel(r'Lag $\tau$ (s)',fontsize=14)
ax1.set_ylabel(r'Autocorrelation $C(\tau)$',fontsize=14)
ax1.set_yticks([0.1,0.2,0.3,0.4,0.6,1],labels=['0.1','0.2','0.3','0.4','0.6','1'])


##### Dynamical information calculations

# No Finite Data Corrections:
uncorrected_data = np.load('entropy_and_info_all_cg_levels.npz', allow_pickle=True)
uncorrected_results = uncorrected_data['results'].item()

corrected_data = np.load('info_cg_finitedata.npz', allow_pickle=True)
corrected_results = corrected_data['results'].item()

shuffled_corrected_data = np.load('info_cg_finitedata_shuffled.npz', allow_pickle=True)
shuffled_corrected_results = shuffled_corrected_data['results'].item()

# Define coarse-graining level
cg_levels = np.array([1,2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,4096,8192])
k_vals = np.arange(1,21)
color_list = ['black','darkviolet','purple','slateblue','blue','cyan','green','springgreen','yellow','orange','red','firebrick','brown','grey']

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
ax2 = fig.add_subplot(gs[0, 1])

slope_list = []
slope_se_list = []
r_squared_list = []
for i in range(len(cg_levels)):
    cg_level = cg_levels[i]
    #Corrected:
    I_k_corrected_mean = corrected_results[cg_level]['I_k_mean']
    I_k_corrected_std = corrected_results[cg_level]['I_k_std']
    # Plot dynamical information
    ax2.errorbar(k_vals*cg_level / 100, I_k_corrected_mean,yerr=I_k_corrected_std, fmt='o-',alpha=0.5, linewidth=2,clip_on=True,label=str(cg_level)+'x',color=color_list[i])



ax2.set_xlabel(r'Lag $\tau$ (s)', fontsize=14)
ax2.set_ylabel(r'Dynamical Information $I_k$ (bits)', fontsize=14)
ax2.set_yscale('log')
ax2.set_xscale('log')

ax2.set_ylim(0.00001,1)
ax2.set_xlim(0.01,1000)

ax2.legend(frameon=False,loc='lower left', bbox_to_anchor=(1, 0),fontsize=8)
ax2.text(1300,0.45,r'\underline{CG factor}',fontsize=10)





# ============================================================================
# PANEL C: Dynamical Information collapsed
# ============================================================================
ax3 = fig.add_subplot(gs[0, 3])

for i in range(len(cg_levels)):
    cg_level = cg_levels[i]
    #Corrected:
    I_k_corrected_mean = corrected_results[cg_level]['I_k_mean']
    I_k_corrected_std = corrected_results[cg_level]['I_k_std']
    # Plot dynamical information
    ax3.errorbar(k_vals, I_k_corrected_mean,yerr=I_k_corrected_std, fmt='o-',alpha=0.5, linewidth=2,clip_on=True,label=str(cg_level)+'x',color=color_list[i])

ax3.set_xlabel(r'Lag $k$', fontsize=14)
ax3.set_ylabel(r'Dynamical Information $I_k$ (bits)', fontsize=14)
ax3.set_yscale('log')
ax3.set_xscale('log')

ax3.set_ylim(0.00001,1)
ax3.set_xlim(1,20)

x_list = np.logspace(-2,3,num=1000)


######################
# --- pooled power-law fit for panel (c): cg >= 16, k >= 3 ---
x_chunks, y_chunks = [], []
for cg in cg_levels[cg_levels >= 32]:
    y = corrected_results[cg]['I_k_mean']
    m = (k_vals >= 3) & np.isfinite(y) & (y > 0)
    x_chunks.append(k_vals[m])
    y_chunks.append(y[m])

x_pool = np.concatenate(x_chunks).astype(float)
y_pool = np.concatenate(y_chunks).astype(float)

res = fit_power_law(x_pool, y_pool)
kfit = np.logspace(np.log10(3), np.log10(k_vals.max()), 256)
ax3.plot(x_list, res['C'] * x_list**res['alpha'], color='black', ls='--', lw=1, zorder=20)
print(f"Panel c pooled fit: alpha={res['alpha']:.3f} ± {res['alpha_se']:.3f}, R^2={res['r_squared']:.3f}")
ax3.text(6,0.01,r'$\sim k^{-2.3}$',fontsize=14)

##########

# ============================================================================
# PANEL D: Dynamical Information vs cg
# ============================================================================
ax4 = fig.add_subplot(gs[0, 4])


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
k_vals = np.arange(1,11)
#color_list = ['black','purple','slateblue','blue','green','springgreen','yellow','orange','red','firebrick','brown','grey']

I_1_mean_list = []
I_k_sum_mean_list = []
I_k_sum_std_list = []
I_k_mean_list = np.zeros((len(cg_levels),20))
I_k_std_list = np.zeros((len(cg_levels),20))

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


        for j in np.arange(1,20):
            if shuffled_I_k_corrected_mean[j] - shuffled_I_k_corrected_std[j]<0:
                I_k_corrected_mean[j]=np.nan
                I_k_corrected_std[j]=np.nan
    
    I_k_sum_mean = 0
    I_k_sum_std = 0
    
    for j in np.arange(1,10):
        I_k_sum_mean += I_k_corrected_mean[j]
        I_k_sum_std += I_k_corrected_std[j]
    

    I_k_sum_mean_list.append(I_k_sum_mean)
    I_k_sum_std_list.append(I_k_sum_std)
    I_1_mean_list.append(I_k_corrected_mean[0])

    

    I_k_mean_list[i,:] = I_k_corrected_mean
    I_k_std_list[i,:] = I_k_corrected_std


ax4.errorbar(cg_levels, I_1_mean_list,yerr=np.zeros(len(I_1_mean_list)), fmt='o-',alpha=0.5, linewidth=2,clip_on=False,label='Markovian',color=color_list[0])

ax4.errorbar(cg_levels, I_k_sum_mean_list,yerr=I_k_sum_std_list, fmt='o-',alpha=0.7, linewidth=2,clip_on=False,label='Non-Markovian',zorder=10,color=color_list[6])


ax4.set_xlabel(r'Coarse-Graining Factor', fontsize=14)
ax4.set_ylabel(r'Dynamical Information (bits)', fontsize=14)
#plt.ylabel(r'$I_k$ (bits)', fontsize=14)
ax4.set_xscale('log')
#plt.yscale('log')
#plt.ylim(0.0008,1)
ax4.set_ylim(0,)
ax4.set_xlim(min(cg_levels),max(cg_levels))

ax4.legend(frameon=False,loc='lower left', bbox_to_anchor=(0, 0.1))


# Create inset plot in top right corner using inset_axes
inset_ax = inset_axes(ax4, width="45%", height="40%", loc='upper right', bbox_to_anchor=(0.01, -0.005, 1, 1), bbox_transform=ax4.transAxes)


timescale_list = []
upper_bound_list =[]
lower_bound_list = []
for k in range(20):
    cg_max_val = cg_levels[np.nanargmax(I_k_mean_list[:,k])]

    if k>0:
        timescale_list.append(cg_max_val*k /100)
        upper_bound_list.append(cg_levels[np.nanargmax(I_k_mean_list[:,k])+1]*k/100)
        lower_bound_list.append(cg_levels[np.nanargmax(I_k_mean_list[:,k])-1]*k/100)


inset_ax.plot(np.arange(2,14),timescale_list[:12],'o-',alpha=1, lw=1,clip_on=False,color=color_list[6],markersize=2.5)

inset_ax.hlines(np.mean(timescale_list[:12]),2,21,color='black',ls='--',lw=1)

inset_ax.fill_between(np.arange(2,14),lower_bound_list[:12],upper_bound_list[:12],alpha=0.3,color=color_list[6])
inset_ax.tick_params(labelsize=8)

inset_ax.set_xlabel(r'Lag $k$', fontsize=10)
inset_ax.set_ylabel(r'$\tau^*$ (s)', fontsize=10)
inset_ax.set_yscale('log')
inset_ax.set_ylim(0.1,100)
inset_ax.set_xlim(2,13)


ax1.set_title('(a)',loc='left')
ax2.set_title('(b)',loc='left')
ax3.set_title('(c)',loc='left')
ax4.set_title('(d)',loc='left')


plt.tight_layout()
plt.savefig('Figure_45.pdf')
plt.show()
