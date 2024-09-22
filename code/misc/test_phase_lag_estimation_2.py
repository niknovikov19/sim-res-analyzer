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
    inp_limits=(1.5, 8), win_len=1.5, win_overlap=0.75, fmax=150)


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

n1 = 0
n2 = 4
        
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
mask0 = np.abs(dphi2) < 1

#plt.figure()
#plt.plot(ff[1:-1], dphi2, '.')
#plt.plot(ff[1:-1][mask0], dphi2[mask0], '.')

def pad_mask(b):
    b_ = np.concatenate(([False], b, [False]))
    b_ = b_[:-1] * b_[1:]
    return b_

# Find a, such that a * f + b = phi
mask0 = pad_mask(mask0) * (ff[:-1] > 15) & (ff[:-1] < 100)
mask = mask0
a, s = None, None
for n in range(3):
    if n == 0:
        mask = mask0
    else:
        mask = mask0 * (np.abs(dphi - a) < (1 * s))    
    a = np.median(dphi[mask])
    s = np.std(dphi - a) 
a0 = a
df = (ff[1] - ff[0])
a /= df
tau = a / (2 * pi)
    
# Find b
mask_ = pad_mask(mask)
theta = phi[mask_] - a * ff[mask_]
sin_sum = np.sum(np.sin(theta))
cos_sum = np.sum(np.cos(theta))
b = np.arctan2(sin_sum, cos_sum)
b = 0

# Linear regression for phi
def angle_mod(x):
    return (x + pi) % (2 * pi) - pi
phi_hat = angle_mod(a * ff + b)
    
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(ff[:-1], dphi, '.')
plt.plot(ff[:-1][mask], dphi[mask], '.')
plt.plot([ff[0], ff[-2]], [a0, a0], 'k--')
plt.title(f'{n1} - {n2}, tau={tau:.05f}')
plt.subplot(3, 1, 2)
plt.plot(ff, phi, '.')
plt.plot(ff, phi_hat, '.')
plt.subplot(3, 1, 3)
plt.plot(ff, angle_mod(phi - phi_hat), '.')

# =============================================================================
# corr = correlate(x1, x2, mode='full')
# lags = correlation_lags(len(x1), len(x2), mode='full')
# plt.figure()
# plt.plot(lags, corr)
# =============================================================================

X.close()
