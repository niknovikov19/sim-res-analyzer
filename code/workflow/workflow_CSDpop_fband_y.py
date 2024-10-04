import os
from pathlib import Path
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sim_res_parser import SimResultParser, RateParams
from data_keeper import DataKeeper
from data_proc import DataProcessor, PSDParams
from plot_utils import plot_xarray_2d


dirpath_storage_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\data')
dirpath_figs_root = Path(r'D:\WORK\Salvador\repo\sim_res_analyzer\results')

fpath_sim_res = (r'D:\WORK\Salvador\repo\A1_model_old\data\exp_10s_LFPpop\exp_10s_LFPpop_data.pkl')
exp_name = 'exp_10s_LFPpop_1'

#fpath_sim_res = r'D:\WORK\Salvador\repo\A1_model_old\data\A1_paper\v34_batch56_10s_data.pkl'
#exp_name = 'exp_10s_1'

fname_metadata = 'metadata.json'

need_parse = 0
need_recalc = 0
need_save = 1

rate_par = RateParams(dt=0.002)

psd_par = PSDParams(
    #inp_limits=(0.5, None), win_len=1.5, win_overlap=0.75, fmax=150)
    inp_limits=(1.5, 8), win_len=1.5, win_overlap=0.75, fmax=150)

fbands = {'low': (5, 25), 'high': (40, 150)}

# none | log | self | total | totalbin
norm_type = 'self'


# Folders for the data and figures
t1, t2 = psd_par.inp_limits[0], psd_par.inp_limits[1]
dirpath_storage = dirpath_storage_root / exp_name
dirpath_figs = dirpath_figs_root / exp_name / f'tlim={t1}-{t2}'
os.makedirs(dirpath_storage, exist_ok=True)
os.makedirs(dirpath_figs, exist_ok=True)

def gen_fig_fpath(group_name, fig_name):
    dirname_fig = (
        f'{group_name}_wlen={psd_par.win_len}_wover={psd_par.win_overlap}_'
        f'low=({fbands["low"][0]}-{fbands["low"][1]})_'
        f'high=({fbands["high"][0]}-{fbands["high"][1]})') 
    fname_fig = f'{fig_name}_norm={norm_type}.png'
    os.makedirs(dirpath_figs / dirname_fig, exist_ok=True)
    return dirpath_figs / dirname_fig / fname_fig

# Initialize data keeper
dk = DataKeeper(str(dirpath_storage), fname_metadata)

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

def plot_fband_psd(W, W0, norm_type):
    for fband in fbands.values():
        w = W.sel(freq=slice(fband[0], fband[1])).mean(dim='freq')
        w0 = W0.sel(freq=slice(fband[0], fband[1])).mean(dim='freq')
        if norm_type == 'total':
            w /= w0.max(dim='y')
        elif norm_type == 'self':
            w /= w.max(dim='y')
        elif norm_type == 'log':
            w = np.log(w)
        elif norm_type == 'totalbin':
            w /= w0
        plt.plot(w.values, w.y.values, label=f'{fband[0]}-{fband[1]}',
                 linewidth=3)
    plt.ylim(w.y[0], w.y[-1])
    plt.gca().invert_yaxis()
    #plt.ylabel('y')
    #plt.legend()

for data_type in ['LFP', 'BIP', 'CSD']:    
    plt.figure(113)
    plt.clf()
    print(f'Plot {data_type}...')
    
    # Total LFP
    W0 = dk.get_data(*data_desc[data_type]['all']['psd'])
    plt.subplot(ny, nx, nx * ny)
    plot_fband_psd(W0, W0, norm_type='self')
    #plt.xlim((0, 1))
    #plt.title(f'PSD: {data_type}, total')
    
    # LFP by population
    with dk.get_data(*data_desc[data_type]['pop']['psd']) as W:
        for n, pop_name in enumerate(pop_names):
            plt.subplot(ny, nx, n + 1)
            W_ = W.sel(pop=pop_name)
            plot_fband_psd(W_, W0, norm_type=norm_type)
            #plt.xlim((0, 0.4))
            plt.title(f'PSD: {data_type}, {pop_name}')
    W0.close()
            
    # Save the result
    if need_save:
        fpath_fig = gen_fig_fpath('LFP_CSD_pop_PSD_fband_y', data_type)
        plt.savefig(fpath_fig, dpi=300)

