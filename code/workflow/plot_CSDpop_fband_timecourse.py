import os
from pathlib import Path
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))

import common as cmn
from data_keeper import DataKeeper
from data_proc import DataProcessor, PSDParams
from sim_res_parser import SimResultParser, RateParams


dirpath_storage_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\data')
dirpath_figs_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\results')

fpath_sim_res = (r'D:\WORK\Salvador\repo\A1_model_old\data\exp_10s_LFPpop\exp_10s_LFPpop_data.pkl')
exp_name = 'exp_10s_LFPpop_1'

fname_metadata = 'metadata.json'

#need_save = 1

rate_par = RateParams(dt=0.002)

fbands = {'low': (5, 25), 'high': (40, 150)}


# Folders for the data and figures
#t1, t2 = psd_par.inp_limits[0], psd_par.inp_limits[1]
dirpath_storage = dirpath_storage_root / exp_name
dirpath_figs = dirpath_figs_root / exp_name #/ f'tlim={t1}-{t2}'
os.makedirs(dirpath_storage, exist_ok=True)
os.makedirs(dirpath_figs, exist_ok=True)

# Initialize data keeper
dk = DataKeeper(str(dirpath_storage), fname_metadata)

data_type = 'CSD'
X = dk.get_data('CSDpop', [('csd', {})])
tt = X.time.values
#fs = 1 / (tt[1] - tt[0])

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

#fband = (5, 10)
#fband = (15, 20)
#fband = (25, 30)
#fband = (35, 40)
fband = (12, 50)

tlim = (1.5, 8)

n1 = 0
n2 = 7

x = {}
for n, desc in enumerate(pop_yrange_descs):
    x[n] = {}
    x_ = X.sel(pop=desc['pop'], y=slice(*desc['yrange'])).mean(dim='y')
    #x_ = x_.sel(time=slice(*tlim))
    x[n]['orig'] = x_
    #x[n]['filt'] = cmn.filter_signal(x_, x_.time.values, fband)
    x[n]['filt'] = x_
    x[n]['filt'] /= x[n]['filt'].std()
    
tt = x[0]['orig'].time.values
    
labels = [f'{desc["pop"]} {desc["yrange"][0]}-{desc["yrange"][1]}'
          for desc in pop_yrange_descs]

tau = 0.067
#tau = 0

plt.figure(figsize=(12, 5))
lw = 2
plt.plot(tt, x[n1]['filt'], label=labels[n1], linewidth=lw)
plt.plot(tt + tau, x[n2]['filt'], label=labels[n2], linewidth=lw)
plt.xlabel('Time')
plt.legend()
plt.title(f'CSD, {fband[0]}-{fband[1]} Hz')
plt.xlim((0, 1))


