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

desc_labels = [f'{desc["pop"]}_{desc["yrange"][0]}-{desc["yrange"][1]}'
          for desc in pop_yrange_descs]

#desc_num_base = 0
desc_num_base = 8

nperseg = 1024
fmax = 150

df = 10
f0_vals = np.arange(0, 125, 1)

need_sqrt = 1

# Output folder
dirname_out = f'{data_type}_coher_polar_nperseg={nperseg}_df={df}'
if need_sqrt:
    dirname_out += '_sqrt'
dirpath_out = dirpath_figs / dirname_out / f'base={desc_labels[desc_num_base]}'
os.makedirs(dirpath_out, exist_ok=True)

x = {}
for n, desc in enumerate(pop_yrange_descs):
    x[n] = {}
    x[n]['orig'] = X.sel(pop=desc['pop'], y=slice(*desc['yrange'])).mean(dim='y')

nsig = len(pop_yrange_descs)

# Coherence and CPSD (all freqs)
C, W = {}, {}
for n1 in range(nsig):
    n2 = desc_num_base
    ff, C[(n1, n2)] = sig.coherence(x[n1]['orig'], x[n2]['orig'], fs=fs,
                    nperseg=nperseg, noverlap=int(nperseg * 0.5))
    ff, W[(n1, n2)] = sig.csd(x[n1]['orig'], x[n2]['orig'], fs=fs,
                    nperseg=nperseg, noverlap=int(nperseg * 0.5))

for n, f0 in enumerate(f0_vals):
    print(f'{n} / {len(f0_vals)}')
    
    # Averge over freq band
    fband = (f0, f0 + df)        
    mask = (ff >= fband[0]) & (ff <= fband[1])
    P = np.zeros(nsig, dtype=np.complex128)
    for n1 in range(nsig):
        n2 = desc_num_base
        c_ = C[(n1, n2)][mask].mean()
        w_ = W[(n1, n2)][mask].mean()
        if need_sqrt:
            P[n1] = np.sqrt(c_) * w_ / np.abs(w_)
        else:
            P[n1] = c_ * w_ / np.abs(w_)
    
    # Plot
    plt.figure(113)
    plt.clf()
    for m in range(nsig):
        plt.plot([0, np.real(P[m])], [0, np.imag(P[m])], '.-', label=desc_labels[m],
             linewidth=3, markersize=13)
    #plt.xlabel('Real')
    #plt.ylabel('Imag')
    plt.xticks([])
    plt.yticks([])
    #plt.legend()
    l = 1
    plt.xlim(-l, l)
    plt.ylim(-l, l)
    plt.gca().set_aspect('equal', 'box')
    title_str = f'{data_type}, {fband[0]}-{fband[1]} Hz'
    plt.title(title_str)
    plt.show()
    plt.draw()
    
    # Save
    fname_out = f'fband={fband[0]}-{fband[1]}.png'
    plt.savefig(dirpath_out / fname_out)
    #plt.close()

X.close()
