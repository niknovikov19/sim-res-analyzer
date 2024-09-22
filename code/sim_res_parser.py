from dataclasses import dataclass, field
import pickle as pkl

import numpy as np
import xarray as xr

from data_keeper import DataKeeper
import sim_res_parse_utils as utils


@dataclass(frozen=True)
class RateParams:
    """Parameters of SimResultParser.extract_pop_rates_dyn() """
    dt: float = 0.001
    time_limits: list = (0, None)


def _sim_res_to_xr_LFP(sim_res):
    lfp, tt, lfp_coords = utils.get_lfp(sim_res)
    dims = ['y', 'time']
    coords = {'y': lfp_coords[:, 1], 'time': tt / 1000}
    return xr.DataArray(lfp, dims=dims, coords=coords)

def _sim_res_to_xr_pop_LFPs(sim_res):
    lfp, tt, lfp_coords = utils.get_pop_lfps(sim_res)
    dims = ['pop', 'y', 'time']
    pop_names = list(lfp.keys())
    coords = {'pop': pop_names, 'y': lfp_coords[:, 1], 'time': tt / 1000}
    sz = [len(coord) for coord in coords.values()]
    X = xr.DataArray(np.full(sz, np.nan), dims=dims, coords=coords)
    for pop in pop_names:
        X.loc[{'pop': pop}] = lfp[pop]
    return X

def _sim_res_to_xr_pop_rates_dyn(sim_res, dt, time_limits=(0, None)):
    pop_names = utils.get_pop_names(sim_res)
    # If upper time limit not specified - use the whole simulation
    time_limits = list(time_limits)
    if time_limits[1] is None:
        time_limits[1] = utils.get_sim_duration(sim_res) / 1000
    # Allocate the output
    tt, _ = utils.calc_rate_dynamics([], time_limits, dt)
    coords = {'pop': pop_names, 'time': tt}
    R = xr.DataArray(np.full((len(pop_names), len(tt)), np.nan),
                     dims=['pop', 'time'], coords=coords)
    # Extract spikes and compute rate dynamics
    for pop_name in pop_names:
        S = utils.get_pop_spikes(sim_res, pop_name)
        pop_sz = utils.get_pop_size(sim_res, pop_name)
        _, r = utils.calc_rate_dynamics(S, time_limits, dt, pop_sz=pop_sz)
        R.loc[{'pop': pop_name}] = r
    return R


class SimResultParser:    
    def __init__(self, dk: DataKeeper = None, fpath_sim_res=None):
        self.dk = dk
        self.sim_res = None
        if fpath_sim_res is not None:
            self.read_sim_result(fpath_sim_res)
        
    def _check_state(self):
        if self.dk is None:
            raise ValueError('DataKeeper was not assigned')
        if self.sim_res is None:
            raise ValueError('Simulation result was not loaded')            
        
    def set_data_keeper(self, dk: DataKeeper):
        """Set DataKeeper object responsible for storing parsed data. """
        self.dk = dk
        
    def read_sim_result(self, fpath_sim_res):
        """Load simulation result from pkl file. """
        with open(fpath_sim_res, 'rb') as fid:
            self.sim_res = pkl.load(fid)
        
    def extract_lfp(self, output_name='LFP'):
        self._check_state()
        X = _sim_res_to_xr_LFP(self.sim_res)
        X -= X.isel(y=0)
        self.dk.store_data(X, output_name)
    
    def extract_pop_lfps(self, output_name='LFPpop'):
        self._check_state()
        X = _sim_res_to_xr_pop_LFPs(self.sim_res)
        self.dk.store_data(X, output_name)
        
    def extract_pop_rates_dyn(self, rate_par: RateParams,
                              output_name='rpop_dyn'):
        self._check_state()
        R = _sim_res_to_xr_pop_rates_dyn(
            self.sim_res, rate_par.dt, rate_par.time_limits)
        self.dk.store_data(R, output_name, [('rate_par', rate_par)])


#@dataclass(frozen=True)
#class PopRateParams:
#    dt: float

#class BatchResultParser:
#    def __init__(self):
#        pass