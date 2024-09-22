import os
from pathlib import Path
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))

import common as cmn
from sim_res_parser import SimResultParser, RateParams
from data_keeper import DataKeeper
from data_proc import DataProcessor, PSDParams
from plot_utils import plot_xarray_2d


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

sig_descs = [{'pop': 'IT3', 'yrange': (0, 250)},
             {'pop': 'IT3', 'yrange': (750, 1000)}]
#sig_descs = [{'pop': 'IT3', 'yrange': (750, 1000)},
#             {'pop': 'ITP4', 'yrange': (750, 1000)}]
#sig_descs = [{'pop': 'IT3', 'yrange': (750, 1000)},
#             {'pop': 'IT5A', 'yrange': (1100, 1300)}]

#fband = (7, 14)
fband = (100, 125)

x = {}
for n, desc in enumerate(sig_descs):
    x[n] = {}
    x[n]['orig'] = X.sel(pop=desc['pop'], y=slice(*desc['yrange'])).mean(dim='y')
    x[n]['filt'] = cmn.filter_signal(x[n]['orig'], tt, fband)
    x[n]['hilb'] = sig.hilbert(x[n]['filt'])

dphi = np.angle(x[1]['hilb'] / x[0]['hilb'])
h, b = np.histogram(dphi, bins=30, density=True)
h = np.concatenate((h, h))
b = np.concatenate((b[:-1], 2 * np.pi + b[:-1]))

title_str = (f'{data_type}  '
    f'{sig_descs[0]["pop"]} (y={sig_descs[0]["yrange"][0]}-{sig_descs[0]["yrange"][1]}) - '
    f'{sig_descs[1]["pop"]} (y={sig_descs[1]["yrange"][0]}-{sig_descs[1]["yrange"][1]})  '
    f'[{fband[0]}-{fband[1]} Hz]')

plt.figure()
plt.clf()
plt.plot(b, h)
plt.title(title_str)
plt.xlabel('Phase difference')
plt.ylabel('Probability')

dirpath_figs_ = dirpath_figs / 'fband_phase_diff_high'
os.makedirs(dirpath_figs_, exist_ok=True)
fpath_fig = dirpath_figs_ / f'{title_str}.png'
plt.savefig(fpath_fig)

# =============================================================================
# plt.figure()
# #plt.subplot(2, 1, 1)
# plt.plot(tt, x[0])
# #plt.subplot(2, 1, 2)
# plt.plot(tt, x[1])
# plt.xlabel('Time')
# =============================================================================



X.close()
