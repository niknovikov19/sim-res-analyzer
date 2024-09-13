import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from data_keeper import DataKeeper


class DataPlotter:
    
    def __init__(self):
        self.dk = DataKeeper()
        
    def plot_psd_yf(self, data_name: str, data_params: dict, slice_coords=None):
        """Plot 2-d PSD (depth x freq) on the current axes. """
        data_in = self.dk.get_data(data_name, data_params)
        # Take slice, e.g. a pop for LFPpop or a batch param combination
        # Plot...
    