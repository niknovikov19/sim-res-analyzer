import os
from pathlib import Path
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy import signal as sig
import scipy.optimize as optimize
from scipy.signal import correlate, correlation_lags
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))

import common as cmn
from sim_res_parser import SimResultParser, RateParams
from data_keeper import DataKeeper
from data_proc import DataProcessor, PSDParams
from plot_utils import plot_xarray_2d, polar_to_rgb


dirpath_storage_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\data')
dirpath_figs_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\results')

fpath_sim_res = (r'D:\WORK\Salvador\repo\A1_model_old\data\exp_10s_LFPpop\exp_10s_LFPpop_data.pkl')
exp_name = 'exp_10s_LFPpop_1'

fname_metadata = 'metadata.json'


rate_par = RateParams(dt=0.002)

psd_par = PSDParams(
    #inp_limits=(0.5, None), win_len=1.5, win_overlap=0.75, fmax=150)
    inp_limits=(1.5, 8), win_len=1.5, win_overlap=0.75, fmax=150)


# Folders for the data and figures
t1, t2 = psd_par.inp_limits[0], psd_par.inp_limits[1]
dirpath_storage = dirpath_storage_root / exp_name
dirpath_figs = dirpath_figs_root / exp_name / f'tlim={t1}-{t2}'
os.makedirs(dirpath_storage, exist_ok=True)
os.makedirs(dirpath_figs, exist_ok=True)

# Initialize data keeper
dk = DataKeeper(str(dirpath_storage), fname_metadata)

data_type = 'CSD'

X = dk.get_data('CSDpop', [('csd', {})])
tt = X.time.values
fs = 1 / (tt[1] - tt[0])

pop_yrange_descs = [
    {'pop': 'IT2', 'yrange': (0, 250)},
    {'pop': 'IT3', 'yrange': (0, 250)},
    {'pop': 'IT3', 'yrange': (750, 1000)},
    {'pop': 'ITP4', 'yrange': (750, 1000)},
    {'pop': 'ITP4', 'yrange': (1050, 1300)},
    {'pop': 'IT5A', 'yrange': (1050, 1300)},
    {'pop': 'IT5B', 'yrange': (1350, 1600)},
    {'pop': 'IT6', 'yrange': (1400, 1650)},
    {'pop': 'CT6', 'yrange': (1400, 1650)}
    ]

nperseg = 1024
fmax = 150

x = {}
for n, desc in enumerate(pop_yrange_descs):
    x[n] = {}
    x[n]['orig'] = X.sel(pop=desc['pop'], y=slice(*desc['yrange'])).mean(dim='y')
    
def pad_mask(b):
     b_ = np.concatenate(([False], b, [False]))
     b_ = b_[:-1] * b_[1:]
     return b_
 
def angle_mod(x):
     return (x + pi) % (2 * pi) - pi
 
dphi2_thresh = 0.7
sigma_thresh = 1
num_med_iter = 3
flim_delay_estim = (10, 100)

nsig = len(pop_yrange_descs)
S = np.full((nsig, nsig), np.nan)
T = np.full((nsig, nsig), np.nan)

for n1, desc1 in enumerate(pop_yrange_descs):
    for n2, desc2 in enumerate(pop_yrange_descs):
        
        if n1 >= n2:
            continue
        
        title_str = (f'{data_type}  '
             f'{desc1["pop"]} (y={desc1["yrange"][0]}-{desc1["yrange"][1]}) - '
             f'{desc2["pop"]} (y={desc2["yrange"][0]}-{desc2["yrange"][1]})')
        print(f'{n1}-{n2} {title_str}')
        
        # Coherence and CPSD
        x1 = x[n1]['orig']
        x2 = x[n2]['orig']
        ff, c = sig.coherence(x1, x2, fs=fs,
                        nperseg=nperseg, noverlap=int(nperseg * 0.5))
        ff, w = sig.csd(x1, x2, fs=fs,
                        nperseg=nperseg, noverlap=int(nperseg * 0.5))
        mask = ff < fmax
        ff = ff[mask]
        c = c[mask]
        w = w[mask]
        
        phi = np.angle(w)
        
        dphi = phi[1:] - phi[:-1]
        dphi = np.where(np.abs(dphi + 2 * pi) > np.abs(dphi), dphi, dphi + 2 * pi)
        dphi = np.where(np.abs(dphi - 2 * pi) > np.abs(dphi), dphi, dphi - 2 * pi)
        dphi2 = dphi[1:] - dphi[:-1]
        mask0 = np.abs(dphi2) < dphi2_thresh
        
        # Find a, such that a * f = phi
        mask_f = ((ff[:-1] > flim_delay_estim[0]) &
                 (ff[:-1] < flim_delay_estim[1]))
        mask0 = (pad_mask(mask0) * mask_f)
        mask = mask0
        a, s = None, None
        for n in range(num_med_iter):
            if n == 0:
                mask = mask0
            else:
                mask = mask0 * (np.abs(dphi - a) < (sigma_thresh * s))    
            a = np.median(dphi[mask])
            s = np.std(dphi[mask_f] - a) 
        a0 = a
        a /= (ff[1] - ff[0])
        tau = a / (2 * pi)
        
        # Linear regression for phi
        phi_hat = angle_mod(a * ff)
        
        S[n2, n1] = s
        T[n2, n1] = tau
            
        # Plot
        plt.figure(111)
        plt.clf()
        col = 1 - c
        plt.subplot(2, 2, 1)
        plt.plot(ff, c)
        plt.ylabel('Coherence')
        plt.ylim(0, 1)
        plt.title(title_str)
        plt.subplot(2, 2, 3)
        for k in (-1, 0, 1):
            d = 2 * pi * k
            plt.scatter(ff, phi + d, c=col, cmap='gray')
            plt.scatter(ff, phi_hat + d, c=col, cmap='gray', marker='x')
            plt.plot([0, fmax], [d, d], 'k--')
        plt.xlabel('Frequency')
        plt.ylabel('Phase diff.')
        plt.subplot(2, 2, 4)
        for k in (-1, 0, 1):
            d = 2 * pi * k
            plt.scatter(ff, phi - phi_hat + d, c=col, cmap='gray')
            plt.plot([0, fmax], [d, d], 'k--')
        plt.xlabel('Frequency')
        plt.ylabel('Phase diff.')
        plt.title('Phase diff. without time delay')
        
        # Save the figure
        dirname_out_1 = f'{data_type}_coher_yrange_nperseg={nperseg}_delay_fix'
        dirname_out_2 = (
            f'd2thresh={dphi2_thresh}_'
            f'flim={flim_delay_estim[0]}-{flim_delay_estim[1]}')
        dirpath_out = dirpath_figs / dirname_out_1 / dirname_out_2
        os.makedirs(dirpath_out, exist_ok=True)
        fpath_out = dirpath_out / f'{title_str}.png'
        plt.savefig(fpath_out) 

X.close()
