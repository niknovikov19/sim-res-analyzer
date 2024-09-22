import os
from pathlib import Path
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy import signal as sig
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))

import common as cmn
from sim_res_parser import SimResultParser, RateParams
from data_keeper import DataKeeper
from data_proc import DataProcessor, PSDParams
from plot_utils import plot_xarray_2d, polar_to_rgb, hsv_colorbar


dirpath_storage_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\data')
dirpath_figs_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\results')

fpath_sim_res = (r'D:\WORK\Salvador\repo\A1_model_old\data\exp_10s_LFPpop\exp_10s_LFPpop_data.pkl')
exp_name = 'exp_10s_LFPpop_1'

fname_metadata = 'metadata.json'


def corr(t, x, y, T):
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)
    dt = t[1] - t[0]
    lag_bins = int(T / dt)
    if len(x) != len(y):
        raise ValueError('Signals should have the same length')
    N = len(x)
    
    # Calculate the autocorrelation
    raw_ac = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')
    
    # Take lags from -lag_bins to lag_bins (centered around zero)
    mid_point = len(raw_ac) // 2
    autocorr = raw_ac[mid_point - lag_bins: mid_point + lag_bins + 1]
    
    # Create the corresponding lags in time
    lags = np.arange(-lag_bins, lag_bins + 1) * dt

    # Compensate for the "triangular" effect
    overlap = N - np.abs(np.arange(-lag_bins, lag_bins + 1))
    autocorr_corrected = autocorr / overlap
    return lags, autocorr_corrected


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

nsig = len(pop_yrange_descs)

# OK: 0, 5, 7

n1 = 0
n2 = 5

x1 = x[n1]['orig'].values
x2 = x[n2]['orig'].values

#mask = (tt >= 1.5) & (tt <= 8)
mask = (tt >= 0.5) & (tt <= 10)
tt = tt[mask]
x1 = x1[mask]
x2 = x2[mask]

fband = (40, 60)
#x1 = cmn.filter_signal(x1, tt, fband)
#x2 = cmn.filter_signal(x2, tt, fband)

lags, c = corr(tt, x1, x2, T=0.2)

plt.figure()
plt.plot(tt, x1)
plt.plot(tt, x2)
plt.xlabel('Time')

plt.figure()
plt.plot(lags, c)
plt.plot([0, 0], [min(c), max(c)], 'k--')
plt.xlabel('Lag')
plt.title(f'Correlation: {n1} - {n2}')

