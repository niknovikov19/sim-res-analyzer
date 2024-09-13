import os
from pathlib import Path
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sim_res_parser import SimResultParser, RateParams
from data_keeper import DataKeeper
from data_proc import DataProcessor, PSDParams
from plot_utils import plot_xarray_2d


dirpath_storage_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\data')
dirpath_figs_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\results')

fpath_sim_res = (r'D:\WORK\Salvador\repo\A1_model_old\data\exp_3s_LFPpop\exp_3s_LFPpop_data.pkl')
exp_name = 'exp_3s_LFPpop_1'

#fpath_sim_res = r'D:\WORK\Salvador\repo\A1_model_old\data\A1_paper\v34_batch56_10s_data.pkl'
#exp_name = 'exp_10s_1'

fname_metadata = 'metadata.json'

need_parse = 0
need_recalc = 0

rate_par = RateParams(dt=0.002)

psd_par = PSDParams(
    inp_limits=(0.5, None), win_len=1.5, win_overlap=0.75, fmax=150)

# Normalize each freq. bin by max. over depth
freq_norm = 1


# Folders for the data and figures
dirpath_storage = dirpath_storage_root / exp_name
dirpath_figs = dirpath_figs_root / exp_name
os.makedirs(dirpath_storage, exist_ok=True)
os.makedirs(dirpath_figs, exist_ok=True)

def gen_fig_fpath(group_name, fig_name):
    dirname_fig = (
        f'{group_name}_wlen={psd_par.win_len}_wover={psd_par.win_overlap}')
    fname_fig = f'{fig_name}_norm={freq_norm}.png'
    os.makedirs(dirpath_figs / dirname_fig, exist_ok=True)
    return dirpath_figs / dirname_fig / fname_fig

# Initialize data keeper
dk = DataKeeper(str(dirpath_storage), fname_metadata)

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

# Here we store data descriptors (name + params), 'sig' - original signals.
# We will derive 'psd' from 'sig' and store them at the same level
data_desc = {
    'LFP': {'all': {'sig': 'LFP'},      # total signal
            'pop': {'sig': 'LFPpop'}},  # contributions of individual pops
    'BIP': {'all': {'sig': 'BIP'},
            'pop': {'sig': 'BIPpop'}},
    'CSD': {'all': {'sig': 'CSD'},
            'pop': {'sig': 'CSDpop'}}
    }

# Calculate bipolar and CSD
inp_name = 'LFPpop'
print('Calculate BIP and CSD...')
for s in ['all', 'pop']:
    # Input: LFP signal
    lfp_desc = (data_desc['LFP'][s]['sig'], [])  # empty params
    # Calculate bipolar
    bip_desc = dproc.calc_bipolar(
        *lfp_desc, out_name=data_desc['BIP'][s]['sig'], recalc=need_recalc)
    # Calculate CSD
    csd_desc = dproc.calc_csd(
        *lfp_desc, out_name=data_desc['CSD'][s]['sig'], recalc=need_recalc)
    # Store descriptors of LFP/BIP/CSD
    data_desc['LFP'][s]['sig'] = lfp_desc
    data_desc['BIP'][s]['sig'] = bip_desc
    data_desc['CSD'][s]['sig'] = csd_desc

# Calculate PSD
for data_type in ['LFP', 'BIP', 'CSD']:
    for s in ['all', 'pop']:
        print(f'Calculate PSD: {data_type}, {s}...')
        desc = data_desc[data_type][s]
        desc['psd'] = dproc.calc_psd(*desc['sig'], psd_params=psd_par,
                                     recalc=need_recalc)

# Get names of populations for which LFPpop exists
with dk.get_data(*data_desc['LFP']['pop']['sig']) as X:
    pop_names = X.coords['pop'].values
    
# PSD (y x freq), pops/total as subplots, LFP/BIP/CSD as separate figures
nx = 4
ny = 2
need_log = not freq_norm
clim = (0, 1) if freq_norm else (-10, 0)
clim_pop = (0, 0.5) if freq_norm else (-30, -5)

for data_type in ['LFP', 'BIP', 'CSD']:    
    plt.figure(111)
    plt.clf()
    print(f'Plot {data_type}...')
    
    # Total LFP
    W0 = dk.get_data(*data_desc[data_type]['all']['psd'])
    plt.subplot(ny, nx, nx * ny)
    if freq_norm:
        W_ = W0 / W0.max(dim='y')
    else:
        W_ = W0 / W0.max()
    plot_xarray_2d(W_, need_log=need_log, clim=clim)
    plt.colorbar()
    plt.title(f'PSD: {data_type}, total')
    
    # LFP by population
    with dk.get_data(*data_desc[data_type]['pop']['psd']) as W:
        for n, pop_name in enumerate(pop_names):
            plt.subplot(ny, nx, n + 1)
            W_ = W.sel(pop=pop_name)
            if freq_norm:
                if freq_norm:
                    W_ = W_ / W0.max(dim='y')
                else:
                    W_ = W_ / W0.max()
            plot_xarray_2d(W_, need_log=need_log, clim=clim_pop)
            plt.colorbar()
            plt.title(f'PSD: {data_type}, {pop_name}')
    W0.close()
            
    # Save the result
    fpath_fig = gen_fig_fpath('LFP_CSD_pop_PSD_yf', data_type)
    plt.savefig(fpath_fig, dpi=300)

