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
    inp_limits=(0.5, None), win_len=1.5, win_overlap=0.75, fmax=150)


# Folders for the data and figures
dirpath_storage = dirpath_storage_root / exp_name
dirpath_figs = dirpath_figs_root / exp_name
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

n1 = 2
n2 = 5
        
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

#plt.figure()
#plt.plot(ff, np.unwrap(phi), '.')   

# =============================================================================
# def phase_difference(x, y):
#     return np.angle(np.exp(1j * (x - y)))
# 
# def cost_function(params, f, phi):
#     a, b = params
#     predicted_phase = a * f + b
#     phase_error = phase_difference(predicted_phase, phi)
#     return np.sum(phase_error**2)
# 
# # Perform optimization
# mask_opt = (ff < 50)
# params_start = [1, 1.5]
# result = optimize.minimize(
#     cost_function, params_start, args=(ff[mask_opt], phi[mask_opt]), method='L-BFGS-B',
#     tol=1e-8, options={'disp': False, 'maxiter': 1e4, 'ftol': 1e-6})
# a, b = result.x
# print(result)
# 
# phi_hat = a * ff + b
# phi_hat = np.mod(phi_hat + pi, 2 * pi) - pi
# 
# plt.figure()
# for n in (-1, 0, 1):
#     d = 2 * pi * n
#     plt.plot(ff, phi + d, 'k.')
#     plt.plot(ff, phi_hat + d, 'r.')
#     
# plt.figure()
# for n in (-1, 0, 1):
#     d = 2 * pi * n
#     plt.plot(ff, phi - phi_hat + d, 'k.')
# 
# =============================================================================

dphi = phi[1:] - phi[:-1]
dphi = np.where(np.abs(dphi + 2 * pi) > np.abs(dphi), dphi, dphi + 2 * pi)
dphi = np.where(np.abs(dphi - 2 * pi) > np.abs(dphi), dphi, dphi - 2 * pi)

mask = dphi < 100  # True

for n in range(1):
    med = np.median(dphi[mask])
    s = np.std(dphi - med)
    mask = np.abs(dphi - med) < (1.0 * s)
    
plt.figure()
plt.plot(ff[:-1], dphi, '.')
plt.plot(ff[:-1][mask], dphi[mask], '.')
plt.plot([ff[0], ff[-2]], [med, med], 'k--')

# =============================================================================
# corr = correlate(x1, x2, mode='full')
# lags = correlation_lags(len(x1), len(x2), mode='full')
# plt.figure()
# plt.plot(lags, corr)
# =============================================================================

X.close()
