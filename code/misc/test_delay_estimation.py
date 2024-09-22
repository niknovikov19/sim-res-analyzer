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

n1 = 0
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

# Phase
phi = np.angle(w)

# 1-st derivative
dphi = phi[1:] - phi[:-1]
dphi = np.where(np.abs(dphi + 2 * pi) > np.abs(dphi), dphi, dphi + 2 * pi)
dphi = np.where(np.abs(dphi - 2 * pi) > np.abs(dphi), dphi, dphi - 2 * pi)
dphi = np.concatenate((dphi, [0]))

dphi_sm = 3
dphi2_thresh = 0.4
dphi2_sm_thresh = 0.2

# Smooth 1-st derivative
from scipy.ndimage import uniform_filter1d
dphi_smooth = uniform_filter1d(dphi, size=dphi_sm)

# 2-nd derivative
#dphi2 = dphi_smooth[1:] - dphi_smooth[:-1]
dphi2 = dphi[1:] - dphi[:-1]
dphi2 = np.concatenate(([0], dphi2))

# Smooth 2-nd derivative
dphi2_smooth = uniform_filter1d(dphi2, size=dphi_sm)

# Find point with small 2-nd derivative (i.e. on linear segments)
mask = np.abs(dphi2) < dphi2_thresh
mask1 = np.abs(dphi2_smooth) < dphi2_sm_thresh

# Expand the obtained segmennts by 1 bin to the left and right
#mask1 = mask | np.roll(mask, -1) | np.roll(mask, 1)

a0 = np.median(dphi[mask])
a = a0 / (ff[1] - ff[0])
a01 = np.median(dphi[mask1])
a1 = a01 / (ff[1] - ff[0])

s = np.mean((dphi[mask] - a0) ** 2)
s1 = np.mean((dphi[mask1] - a01) ** 2)
#s = np.mean(mask)
#s1 = np.mean(mask1)

phi_hat = (a * ff + pi) % (2 * pi) - pi
phi_hat1 = (a1 * ff + pi) % (2 * pi) - pi

tau_vec = np.zeros(len(ff))
tau_vec[0] = np.nan
tau_vec[1] = phi[1] / (2 * pi * ff[1])
dd = 2 * pi * np.arange(100)
for n, f in enumerate(ff[2:]):
    k = int(np.round(f * (tau_vec[n - 1] - (phi[n] / (2 * pi * f)))))
    tau_vec[n] = (phi[n] + 2 * pi * k) / (2 * pi * f)

plt.figure()

plt.subplot(2, 1, 1)
for k in (-1, 0, 1):
    d = 2 * pi * k
    plt.plot([0, fmax], [d, d], 'b--')
    plt.plot(ff, phi + d, 'k.')
plt.title('Phase diff.')

plt.subplot(2, 1, 2)
# =============================================================================
# for k in range(10):
#     tau = (phi + 2 * pi * k) / (2 *pi * ff)
#     plt.plot(ff, tau, '.')
# plt.ylim(-0.1, 0.7)
# =============================================================================
plt.plot(ff, tau_vec, '.-')
#plt.plot(ff, tau_vec + 1 / ff, '.')
#plt.plot(ff, tau_vec - 1 / ff, '.')
plt.xlabel('Frequency')
plt.title('Time delay')

X.close()
