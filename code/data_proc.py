from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from xr_proc import xr_proc
from data_keeper import DataKeeper


@dataclass(frozen=True)
class PSDParams:
    """Parameters of DataProcessor.calc_psd() """
    time_dim: str = 'time'
    inp_limits: list = (None, None)
    method: str = 'welch'
    win_len: float = 1
    win_overlap: float = 0.5
    fmax: float = 100

  
class DataProcessor:
    def __init__(self, dk: DataKeeper = None):
        self.dk = dk
        
    def _check_state(self):
        if self.dk is None:
            raise ValueError('DataKeeper was not assigned')
    
    def _gen_out_par(self, inp_par, step_name, step_par=None):
        if inp_par is None: inp_par = []
        if step_par is None: step_par = {}
        return inp_par + [(step_name, step_par)]
    
    def _gen_out_name_par(self, inp_name, inp_par, out_name=None, 
                          step_name='step', step_par=None):
        if out_name is None:
            out_name = inp_name + '_' + step_name
        out_par = self._gen_out_par(inp_par, step_name, step_par)
        return out_name, out_par
        
    def set_data_keeper(self, dk: DataKeeper):
        """Set DataKeeper object responsible for storing parsed data. """
        self.dk = dk
        
    def calc_bipolar(self, inp_name: str, inp_params: dict = None,
                     out_name: str = None, recalc: bool = False):
        """Calculate bipolar reference (1-st derivative along "y" dimension). """
        self._check_state()
        out_name, out_params = self._gen_out_name_par(
            inp_name, inp_params, out_name, step_name='bip')
        if recalc or not self.dk.exists(out_name, out_params):       
            with self.dk.get_data(inp_name, inp_params) as X:
                Y = xr_proc.calc_xr_diff(X, n=1)            
                self.dk.store_data(Y, out_name, out_params)
        return out_name, out_params
    
    def calc_csd(self, inp_name: str, inp_params: dict = None,
                 out_name: str = None, recalc: bool = False):
        """Calculate CSD (2-nd derivative along "y" dimension). """
        self._check_state()
        out_name, out_params = self._gen_out_name_par(
            inp_name, inp_params, out_name, step_name='csd')
        if recalc or not self.dk.exists(out_name, out_params):
            with self.dk.get_data(inp_name, inp_params) as X:
                Y = xr_proc.calc_xr_diff(X, n=2)            
                self.dk.store_data(Y, out_name, out_params)
        return out_name, out_params
    
    def calc_psd(self, inp_name: str, inp_params: dict = None,
                 psd_params: PSDParams = None,
                 out_name: str = None, recalc: bool = False):
        """Calculate power spectral density. """
        self._check_state()
        if psd_params is None: psd_params = PSDParams()
        if psd_params.method != 'welch':
            raise ValueError('Only "welch" method is supported')
        out_name, out_params = self._gen_out_name_par(
            inp_name, inp_params, out_name,
            step_name='psd', step_par=psd_params)
        if recalc or not self.dk.exists(out_name, out_params):
            # Open the input
            with self.dk.get_data(inp_name, inp_params) as X:
                # Select time interval of interest
                Y = X.sel({psd_params.time_dim: slice(*psd_params.inp_limits)})
                # Calculate power using welch method
                W = xr_proc.calc_xr_welch(
                    Y, win_len=psd_params.win_len,
                    win_overlap=psd_params.win_overlap,
                    fmax=psd_params.fmax, time_dim=psd_params.time_dim
                )
                # Store the result            
                self.dk.store_data(W, out_name, out_params)
        return out_name, out_params
    
    # TODO: need_recalc, return both new name and new params
    
    # Smoothing
    # Re-reference
    # Arbitrary function

#from scipy import signal as sig
#sig.welch()