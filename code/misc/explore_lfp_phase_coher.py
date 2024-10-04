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

X = X.sel(time=slice(1.5, 8))

tt = X.time.values
fs = 1 / (tt[1] - tt[0])

#sig_descs = [{'pop': 'IT3', 'yrange': (0, 250)},
#             {'pop': 'IT3', 'yrange': (750, 1000)}]
#sig_descs = [{'pop': 'IT3', 'yrange': (750, 1000)},
#             {'pop': 'ITP4', 'yrange': (750, 1000)}]
#sig_descs = [{'pop': 'IT3', 'yrange': (750, 1000)},
#             {'pop': 'IT5A', 'yrange': (1100, 1300)}]
#sig_descs = [{'pop': 'ITP4', 'yrange': (750, 1000)},
#             {'pop': 'IT5A', 'yrange': (1100, 1300)}]
sig_descs = [{'pop': 'IT2', 'yrange': (0, 250)},
             {'pop': 'IT6', 'yrange': (1400, 1650)}]

#fband = (7, 14)
#fband = (70, 80)

nperseg = 1024

x = {}
for n, desc in enumerate(sig_descs):
    x[n] = {}
    x[n]['orig'] = X.sel(pop=desc['pop'], y=slice(*desc['yrange'])).mean(dim='y')

ff, c = sig.coherence(x[0]['orig'], x[1]['orig'], fs=2000,
                nperseg=nperseg, noverlap=int(nperseg * 0.5))
ff, w = sig.csd(x[0]['orig'], x[1]['orig'], fs=2000,
                nperseg=nperseg, noverlap=int(nperseg * 0.5))

fmax = 150
mask = ff < fmax
ff_ = ff[mask]

title_str = (f'{data_type}  '
     f'{sig_descs[0]["pop"]} (y={sig_descs[0]["yrange"][0]}-{sig_descs[0]["yrange"][1]}) - '
     f'{sig_descs[1]["pop"]} (y={sig_descs[1]["yrange"][0]}-{sig_descs[1]["yrange"][1]})')

plt.figure()
plt.subplot(2, 1, 1)
c_ = np.abs(c[mask])
plt.plot(ff_, c_)
plt.ylabel('Coherence')
plt.title(title_str)
plt.subplot(2, 1, 2)
for k in (-1, 0, 1):
    d = 2 * pi * k
    col = 1 - c_
    plt.scatter(ff_, np.angle(w[mask]) + d, c=col, cmap='gray')
    plt.plot([0, fmax], [d, d], 'k--')
plt.xlabel('Frequency')
plt.ylabel('Phase diff.')

X.close()
