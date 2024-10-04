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

x = {}
for n, desc in enumerate(pop_yrange_descs):
    x[n] = {}
    x[n]['orig'] = X.sel(pop=desc['pop'], y=slice(*desc['yrange'])).mean(dim='y')

for n1, desc1 in enumerate(pop_yrange_descs):
    for n2, desc2 in enumerate(pop_yrange_descs):
        
        if n1 >= n2:
            continue
        
        title_str = (f'{data_type}  '
             f'{desc1["pop"]} (y={desc1["yrange"][0]}-{desc1["yrange"][1]}) - '
             f'{desc2["pop"]} (y={desc2["yrange"][0]}-{desc2["yrange"][1]})')
        print(f'{n1}-{n2} {title_str}')
        
        # Coherence and CPSD
        ff, c = sig.coherence(x[n1]['orig'], x[n2]['orig'], fs=fs,
                        nperseg=nperseg, noverlap=int(nperseg * 0.5))
        ff, w = sig.csd(x[n1]['orig'], x[n2]['orig'], fs=fs,
                        nperseg=nperseg, noverlap=int(nperseg * 0.5))
        mask = ff < fmax
        ff_ = ff[mask]

        # Plot
        #plt.figure(113)
        plt.figure(113, figsize=(12, 7))
        plt.clf()
        plt.subplot(2, 1, 1)
        c_ = np.abs(c[mask])
        plt.plot(ff_, c_, linewidth=3)
        #plt.ylabel('Coherence')
        plt.ylim(0, 1)
        plt.title(title_str)
        plt.xticks([])
        plt.subplot(2, 1, 2)
        #for k in (-1, 0, 1):
        for k in (0, 1):
            d = 2 * pi * k
            col = 1 - c_
            plt.scatter(ff_, np.angle(w[mask]) + d, c=col, cmap='gray',
                        s=60)
            plt.plot([0, fmax], [d, d], 'k--')
        plt.ylim(-pi, 3 * pi)
        #plt.xlabel('Frequency')
        #plt.ylabel('Phase diff.')
        plt.subplots_adjust(hspace=0.1) 
        
        # Save the figure
        dirname_out = f'{data_type}_coher_yrange_nperseg={nperseg}'
        dirpath_out = dirpath_figs / dirname_out
        os.makedirs(dirpath_out, exist_ok=True)
        fpath_out = dirpath_out / f'{title_str}.png'
        plt.savefig(fpath_out)
        
        #break

X.close()
