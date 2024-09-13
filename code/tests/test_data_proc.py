#import os
from pathlib import Path
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy import stats
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_keeper import DataKeeper
from data_proc import DataProcessor, PSDParams
from plot_utils import plot_xarray_2d


storage_dir = r'D:\WORK\Salvador\repo\sim_res_analyzer\data\test_data_proc'
metadata_file = 'metadata3.json'


dt = 0.001
T = 3

ny = 200

osc = [{'f': 10, 'amp': 1, 'phi': 0},
       {'f': 25, 'amp': 0.5, 'phi': pi/3}]

depth_peaks = [{'y': 0.2, 'amp': 1, 'width': 0.1}, 
               {'y': 0.8, 'amp': 3, 'width': 0.2}]

psd_par = PSDParams(win_len=0.75, win_overlap=0.5, fmax=50)


# Time bins
tt = np.arange(0, T, dt)
nt = len(tt)

# Depth bins
yy = np.linspace(0, 1, ny)

# Generate 1-d signal with several sinusoidal components
x = np.zeros((1, nt))
for osc_ in osc:
    x += osc_['amp'] * np.cos(2 * np.pi * osc_['f'] * tt + osc_['phi'])
    
# Generate depth profile
z = np.zeros((ny, 1))
for dp in depth_peaks:
    z_ = dp['amp'] * stats.norm.pdf(yy, loc=dp['y'], scale=dp['width'])
    z += z_.reshape(z.shape)
    
# Create depth x time signal matrix
X = x * z
dims = ['y', 'time']
coords = {'y': yy, 'time': tt}
X = xr.DataArray(X, dims=dims, coords=coords)

#plt.figure()
#plot_xarray_2d(X)

# Initialize data keeper
dk = DataKeeper(storage_dir, metadata_file)

# Store the input signal
inp_name = 'X'
inp_name_par = (inp_name, None)
dk.store_data(X, inp_name)

# Initializa data processor
dproc = DataProcessor(dk)

# Calculate bipolar and CSD
bip_name_par = dproc.calc_bipolar(inp_name)
csd_name_par = dproc.calc_csd(inp_name)

data_info = {
    'lfp': {'orig': inp_name_par},
    'bip': {'orig': bip_name_par},
    'csd': {'orig': csd_name_par},
    }

# Calculate PSD
for info in data_info.values():
    info['psd'] = dproc.calc_psd(*info['orig'], psd_params=psd_par)
    
pprint(dk.list_data())

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
    # Depth x freq.
    plt.subplot(ny, nx, nx * n + 2)
    with dk.get_data(*info['psd']) as W:
        plot_xarray_2d(W[1 : -2, :])
    plt.title(f'{name}, PSD')
