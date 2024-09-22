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

fpath_sim_res = (r'D:\WORK\Salvador\repo\A1_model_old\data\exp_10s_LFPpop\exp_10s_LFPpop_data.pkl')
exp_name = 'exp_10s_LFPpop_1'

fname_metadata = 'metadata.json'

need_parse = 0
need_recalc = 0

#rate_par = RateParams(dt=0.002)
rate_par = RateParams(dt=0.002, time_limits=(7, 9))


# Folders for the data and figures
t1 = rate_par.time_limits[0]
t2 = rate_par.time_limits[1]
dt = rate_par.dt
dirpath_storage = dirpath_storage_root / exp_name
dirpath_figs = dirpath_figs_root / exp_name / f'pop_rates_dt={dt}_t={t1}-{t2}'
os.makedirs(dirpath_storage, exist_ok=True)
os.makedirs(dirpath_figs, exist_ok=True)

# Initialize data keeper
dk = DataKeeper(str(dirpath_storage), fname_metadata)

with dk.get_data('rpop_dyn', [('rate_par', rate_par)]) as R:
    for pop in R.pop.values:
        print(pop)
        plt.figure(112)
        plt.clf()
        plt.plot(R.time, R.sel(pop=pop))
        plt.xlabel('Time')
        plt.title(f'Firing rate, {pop}')
        fpath_out = dirpath_figs / f'{pop}.png'
        plt.savefig(fpath_out)
    

