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

fpath_sim_res = (r'D:\WORK\Salvador\repo\A1_model_old\data\exp_10s_LFPpop\exp_10s_LFPpop_data.pkl')
exp_name = 'exp_10s_LFPpop_1'

fname_metadata = 'metadata.json'


#rate_par = RateParams(dt=0.002)
rate_par = RateParams(dt=0.002, time_limits=(7, 9))


# Folders for the data and figures
dirpath_storage = dirpath_storage_root / exp_name
os.makedirs(dirpath_storage, exist_ok=True)

# Initialize data keeper
dk = DataKeeper(str(dirpath_storage), fname_metadata)

#r = dk.get_data('rpop_dyn', rate_par)

# Parse sim result
print('Open pkl...')
parser = SimResultParser(dk, fpath_sim_res)
print('Extract LFP and LFPpop...')
parser.extract_lfp()
parser.extract_pop_lfps()
print('Calculate rate dynamics...')
parser.extract_pop_rates_dyn(rate_par)
# TODO: free pkl