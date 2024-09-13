from pathlib import Path
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy import stats
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sim_res_parser import SimResultParser, RateParams
from data_keeper import DataKeeper
from data_proc import DataProcessor, PSDParams
from plot_utils import plot_xarray_2d


fpath_sim_res = (r'D:\WORK\Salvador\repo\A1_model_old\data\exp_3s_LFPpop\exp_3s_LFPpop_data.pkl')
storage_dir = r'D:\WORK\Salvador\repo\sim_res_analyzer\data\exp_3s_LFPpop_1'

#fpath_sim_res = r'D:\WORK\Salvador\repo\A1_model_old\data\A1_paper\v34_batch56_10s_data.pkl'
#storage_dir = r'D:\WORK\Salvador\repo\sim_res_analyzer\data\exp_10s_1'

metadata_file = 'metadata.json'

need_parse = 0
need_recalc = 0

rate_par = RateParams(dt=0.002)
psd_par = PSDParams(
    inp_limits=(0.5, None), win_len=1.5, win_overlap=0.75, fmax=150)


# Initialize data keeper
dk = DataKeeper(storage_dir, metadata_file)

#r = dk.get_data('rpop_dyn', rate_par)

# Parse sim result
if need_parse:
    print('Open pkl...')
    parser = SimResultParser(dk, fpath_sim_res)
    print('Extract LFP and LFPpop...')
    parser.extract_lfp()
    parser.extract_pop_lfps()
    print('Calculate rate dynamics...')
    parser.extract_pop_rates_dyn(rate_par)
    # TODO: free pkl

# Initializa data processor
dproc = DataProcessor(dk)

# Calculate bipolar and CSD
inp_name = 'LFPpop'
print('Calculate BIP and CSD...')
bip_name_par = dproc.calc_bipolar(inp_name, out_name='BIP', recalc=need_recalc)
csd_name_par = dproc.calc_csd(inp_name, out_name='CSD', recalc=need_recalc)

data_info = {
    'LFP': {'orig': (inp_name, [])},
    'BIP': {'orig': bip_name_par},
    'CSD': {'orig': csd_name_par},
    }

# Calculate PSD
for name, info in data_info.items():
    print(f'Calculate PSD: {name}...')
    info['psd'] = dproc.calc_psd(*info['orig'], psd_params=psd_par,
                                 recalc=need_recalc)
    
#print(list(dk.list_data().keys()))

# Visualize
nx = 2
ny = len(data_info)
plt.figure()
for n, (name, info) in enumerate(data_info.items()):
    # Depth x time
    plt.subplot(ny, nx, nx * n + 1)
    with dk.get_data(*info['orig']) as X:
        plot_xarray_2d(X[1 : -2, :])
    plt.title(name)
    plt.colorbar()
    # Depth x freq.
    plt.subplot(ny, nx, nx * n + 2)
    with dk.get_data(*info['psd']) as W:
        W_ = W / W.max(dim='y')
        plot_xarray_2d(W_, need_log=False)
    plt.title(f'{name}, PSD')
    plt.colorbar()
