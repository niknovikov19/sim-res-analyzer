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


t_limits=(1.5, 8)


# Folders for the data and figures
t1, t2 = t_limits[0], t_limits[1]
dirpath_storage = dirpath_storage_root / exp_name
dirpath_figs = dirpath_figs_root / exp_name / f'tlim={t1}-{t2}'
os.makedirs(dirpath_storage, exist_ok=True)
os.makedirs(dirpath_figs, exist_ok=True)

# Initialize data keeper
dk = DataKeeper(str(dirpath_storage), fname_metadata)

data_type = 'CSD'

X = dk.get_data('CSDpop', [('csd', {})])
X = X.sel(time=slice(*t_limits))
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

fband = (4, 14)
#fband = (50, 100)

# =============================================================================
# title_str = (f'{data_type}  '
#      f'{desc1["pop"]} (y={desc1["yrange"][0]}-{desc1["yrange"][1]}) - '
#      f'{desc2["pop"]} (y={desc2["yrange"][0]}-{desc2["yrange"][1]})')
# print(f'{n1}-{n2} {title_str}')
# =============================================================================

x = {}
for n, desc in enumerate(pop_yrange_descs):
    x[n] = {}
    x[n]['orig'] = X.sel(pop=desc['pop'], y=slice(*desc['yrange'])).mean(dim='y')

nsig = len(pop_yrange_descs)
C = np.zeros((nsig, nsig), dtype=np.complex128)

for n1, desc1 in enumerate(pop_yrange_descs):
    for n2, desc2 in enumerate(pop_yrange_descs):
        # Coherence and CPSD
        ff, c = sig.coherence(x[n1]['orig'], x[n2]['orig'], fs=fs,
                        nperseg=nperseg, noverlap=int(nperseg * 0.5))
        ff, w = sig.csd(x[n1]['orig'], x[n2]['orig'], fs=fs,
                        nperseg=nperseg, noverlap=int(nperseg * 0.5))
        mask = (ff >= fband[0]) & (ff <= fband[1])
        c_ = c[mask].mean()
        w_ = w[mask].mean()
        C[n1, n2] = c_ * w_ / np.abs(w_)

n_base = 0
#n_base = 8

need_sqrt = 1

title_str = f'{data_type}, {fband[0]}-{fband[1]} Hz'
labels = [f'{desc["pop"]} {desc["yrange"][0]}-{desc["yrange"][1]}'
          for desc in pop_yrange_descs]
plt.figure()
for n in range(nsig):
    c = C[n, n_base]
    if need_sqrt:
        c = c / np.abs(c) * np.sqrt(np.abs(c))
    plt.plot([0, np.real(c)], [0, np.imag(c)], '.-', label=labels[n],
             linewidth=3, markersize=13)
#plt.xlabel('Real')
#plt.ylabel('Imag')
plt.xticks([])
plt.yticks([])
#plt.legend()
l = 0.9
plt.xlim(-l, l)
plt.ylim(-l, l)
plt.gca().set_aspect('equal', 'box')
plt.title(title_str)

X.close()
