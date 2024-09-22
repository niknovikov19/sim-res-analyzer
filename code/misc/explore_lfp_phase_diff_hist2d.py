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
from plot_utils import plot_xarray_2d, polar_to_rgb


dirpath_storage_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\data')
dirpath_figs_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\results')

fpath_sim_res = (r'D:\WORK\Salvador\repo\A1_model_old\data\exp_3s_LFPpop\exp_3s_LFPpop_data.pkl')
exp_name = 'exp_3s_LFPpop_1'

fname_metadata = 'metadata.json'


rate_par = RateParams(dt=0.002)

psd_par = PSDParams(
    inp_limits=(0.5, None), win_len=1.5, win_overlap=0.75, fmax=150)

fbands = {'low': (5, 25), 'high': (40, 150)}

# none | log | self | total
norm_type = 'log'


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

#sig_descs = [{'pop': 'IT3', 'yrange': (0, 250)},
#             {'pop': 'IT3', 'yrange': (750, 1000)}]
#sig_descs = [{'pop': 'IT3', 'yrange': (750, 1000)},
#             {'pop': 'ITP4', 'yrange': (750, 1000)}]
#sig_descs = [{'pop': 'IT3', 'yrange': (750, 1000)},
#             {'pop': 'IT5A', 'yrange': (1100, 1300)}]
sig_descs = [{'pop': 'ITP4', 'yrange': (750, 1000)},
             {'pop': 'IT5A', 'yrange': (1100, 1300)}]

#fband = (7, 14)
#fband = (70, 80)

nperseg = 512
noverlap=int(nperseg * 0.95)

x = {}
for n, desc in enumerate(sig_descs):
    x[n] = {}
    x[n]['orig'] = X.sel(pop=desc['pop'], y=slice(*desc['yrange'])).mean(dim='y')
    ff, tt1, x[n]['stft'] = sig.stft(
        x[n]['orig'], fs=fs, nperseg=nperseg, noverlap=noverlap)

# CPSD and cross-coherence for each time window
Z0 = x[0]['stft']
Z1 = x[1]['stft']
P00 = np.abs(Z0) ** 2
P11 = np.abs(Z1) ** 2
P01 = Z0 * np.conj(Z1)
C = np.abs(P01) ** 2 / (P00 * P11)
P01 = xr.DataArray(P01, dims=['freq', 'time'], 
                 coords={'freq': ff, 'time': tt1})
C = xr.DataArray(C, dims=P01.dims, coords=P01.coords)

fmax = 150
P01 = P01.sel(freq=slice(0, fmax))
C = C.sel(freq=slice(0, fmax))
ff = C.freq.values

# Histogram of the phase diff. at each frequency
nbins = 15
bins = np.linspace(-pi, pi, nbins + 1)
H = np.full((len(ff), nbins), np.nan)
for n, f in enumerate(ff):
    H[n, :], _ = np.histogram(
        np.angle(P01.sel(freq=f)), bins=bins, density=True)
H = xr.DataArray(
    H, dims=['freq', 'phase'], coords={'freq': ff, 'phase': bins[:-1]})

title_str = (f'{data_type}  '
     f'{sig_descs[0]["pop"]} (y={sig_descs[0]["yrange"][0]}-{sig_descs[0]["yrange"][1]}) - '
     f'{sig_descs[1]["pop"]} (y={sig_descs[1]["yrange"][0]}-{sig_descs[1]["yrange"][1]})')

plt.figure()
plot_xarray_2d(H, clim=(0, 1))
plt.colorbar()

X.close()
