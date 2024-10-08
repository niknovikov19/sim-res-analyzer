import os
from pathlib import Path
from pprint import pprint
import sys

import cmasher
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
from plot_utils import plot_circle_matrix


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
fmax = 150

df = 10
f0_vals = np.arange(0, 125, 1)

need_sqrt = 1

# =============================================================================
# title_str = (f'{data_type}  '
#      f'{desc1["pop"]} (y={desc1["yrange"][0]}-{desc1["yrange"][1]}) - '
#      f'{desc2["pop"]} (y={desc2["yrange"][0]}-{desc2["yrange"][1]})')
# print(f'{n1}-{n2} {title_str}')
# =============================================================================

# Output folder
dirname_out = f'{data_type}_coher_mat_nperseg={nperseg}_df={df}'
if need_sqrt:
    dirname_out += '_sqrt'
dirpath_out = dirpath_figs / dirname_out
os.makedirs(dirpath_out, exist_ok=True)

x = {}
for n, desc in enumerate(pop_yrange_descs):
    x[n] = {}
    x[n]['orig'] = X.sel(pop=desc['pop'], y=slice(*desc['yrange'])).mean(dim='y')

nsig = len(pop_yrange_descs)

# Coherence and CPSD (all freqs)
C, W = {}, {}
for n1, desc1 in enumerate(pop_yrange_descs):
    for n2, desc2 in enumerate(pop_yrange_descs):        
        ff, C[(n1, n2)] = sig.coherence(x[n1]['orig'], x[n2]['orig'], fs=fs,
                        nperseg=nperseg, noverlap=int(nperseg * 0.5))
        ff, W[(n1, n2)] = sig.csd(x[n1]['orig'], x[n2]['orig'], fs=fs,
                        nperseg=nperseg, noverlap=int(nperseg * 0.5))

for n, f0 in enumerate(f0_vals):
    print(f'{n} / {len(f0_vals)}')
    
    # Averge over freq band
    fband = (f0, f0 + df)        
    mask = (ff >= fband[0]) & (ff <= fband[1])
    P = np.zeros((nsig, nsig), dtype=np.complex128)
    for n1, desc1 in enumerate(pop_yrange_descs):
        for n2, desc2 in enumerate(pop_yrange_descs):    
            c_ = C[(n1, n2)][mask].mean()
            w_ = W[(n1, n2)][mask].mean()
            P[n1, n2] = c_ * w_ / np.abs(w_)
    
    # Plot
    plt.figure(113)
    plt.clf()
# =============================================================================
#     Prgb = polar_to_rgb(P, s_mult=1.5)  
#     plt.imshow(Prgb, aspect='equal', origin='lower')
#     labels = [f'{desc["pop"]} {desc["yrange"][0]}-{desc["yrange"][1]}'
#               for desc in pop_yrange_descs]
#     plt.xticks(ticks=np.arange(nsig), labels=labels, rotation=45, ha='right')
#     plt.yticks(ticks=np.arange(nsig), labels=labels, rotation=0)
#     hsv_colorbar()
# =============================================================================
    Pabs = np.abs(P)
    if need_sqrt:
        Pabs = np.sqrt(Pabs)
    plot_circle_matrix(np.angle(P), Pabs, clim=(-pi, pi),
                       cmap=cmasher.infinity, fig_num=113)
    plt.gca().set_aspect('equal', 'box')
    plt.xticks([])
    plt.yticks([])
    title_str = f'{data_type}, {fband[0]}-{fband[1]} Hz'
    plt.title(title_str)
    
    # Save
    fname_out = f'fband={fband[0]}-{fband[1]}.png'
    plt.savefig(dirpath_out / fname_out)

X.close()
