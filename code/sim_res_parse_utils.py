import numpy as np


def get_pop_names(sim_result):
    return list(sim_result['net']['pops'].keys())

def get_lfp_coords(sim_result):
    cfg = sim_result['simConfig']
    lfp_coords = np.array(cfg['recordLFP'])
    return lfp_coords

def get_record_times(sim_result):
    cfg = sim_result['simConfig']
    dt = cfg['recordStep']
    T = cfg['duration']
    tt = np.arange(0, T, dt)
    return tt

def get_lfp(sim_result):
    lfp = np.array(sim_result['simData']['LFP']).T
    lfp_coords = get_lfp_coords(sim_result)
    tt = get_record_times(sim_result)
    return lfp, tt, lfp_coords

def get_pop_lfps(sim_result):
    lfp = {}
    for pop_name, pop_lfp in sim_result['simData']['LFPPops'].items():
        lfp[pop_name] = np.array(pop_lfp).T
    lfp_coords = get_lfp_coords(sim_result)
    tt = get_record_times(sim_result)
    return lfp, tt, lfp_coords

def get_pop_ylim(sim_res, pop_name):
    h = sim_res['simConfig']['sizeY']
    yy = sim_res['net']['pops'][pop_name]['tags']['ynormRange']
    return (yy[0] * h, yy[1] * h)

def get_layer_borders(sim_result):
    layer_yrange = {}
    for pop_name in sim_result['net']['pops']:
        layer_yrange[pop_name] = get_pop_ylim(sim_result, pop_name)
    return layer_yrange

def get_net_params(sim_result):
    return sim_result['net']['params']

def get_pop_params(sim_result, pop_name=None):
    if pop_name is None:
        return get_net_params(sim_result)['popParams']
    else:
        return get_net_params(sim_result)['popParams'][pop_name]

def get_pop_cell_gids(sim_result, pop_name):
    return sim_result['net']['pops'][pop_name]['cellGids']

def get_sim_data(sim_result):
    return sim_result['simData']

def get_pop_size(sim_result, pop_name):
    return len(get_pop_cell_gids(sim_result, pop_name))

def get_sim_duration(sim_result):
    return sim_result['simConfig']['duration'] / 1000

def get_pop_spikes(sim_result, pop_name, combine_cells=True,
                   t0=0, tmax=None, subtract_t0=True, ms=False,
                   ndigits=6):
    """Get times of spikes generated by a given population. """
    sim_data = get_sim_data(sim_result)
    pop_cell_idx = get_pop_cell_gids(sim_result, pop_name)
    spike_cell_idx = np.array(sim_data['spkid'])
    spkt = np.array(sim_data['spkt']) / 1000
    if tmax is None:
        tmax = get_sim_duration(sim_result)
    t_mask = (spkt >= t0) & (spkt <= tmax)
    tsub = t0 if subtract_t0 else 0
    mult = 1000 if ms else 1
    if combine_cells:
        pop_mask = np.isin(sim_data['spkid'], pop_cell_idx)
        spike_times = (spkt[pop_mask & t_mask] - tsub) * mult
        spike_times = np.round(spike_times, ndigits)
    else:
        spike_times = []
        for cell_id in pop_cell_idx:
            cell_mask = (spike_cell_idx == cell_id)
            s = (spkt[cell_mask & t_mask] - tsub) * mult
            s = np.round(s, ndigits)
            spike_times.append(s)
    return spike_times

def get_pop_cell_rates(sim_result, pop_name, t0=0, tmax=None):
    S = get_pop_spikes(sim_result, pop_name, combine_cells=False)
    T = get_sim_duration(sim_result)
    if tmax is None:
        tmax = T
    r = np.array([len(s[(s >= t0) & (s <= tmax)]) / T for s in S])
    return r

def calc_rate_dynamics(spike_times, time_range, dt, pop_sz=1,
                              epoch_len=None):
    """Calculate firing rate dynamics from combined spiketrains. """
    t1 = time_range[0]
    t2 = time_range[1]
    # Decrease the time range so it is a multiple of the epoch
    if epoch_len is not None:
        num_epochs = np.floor((time_range[1] - time_range[0]) / epoch_len)
        t2 = t1 + epoch_len * num_epochs
    else:
        num_epochs = 1
    # Get spike times within the given time range
    spike_times = np.array(spike_times)
    mask = (spike_times >= t1) & (spike_times <= t2)
    spike_times = spike_times[mask]
    # Put all the spikes into a single epoch
    if epoch_len is not None:
        spike_times = ((spike_times - t1) % epoch_len) + t1
        t2 = t1 + epoch_len
    # Transform: spike time -> sample number
    Nbins = int((t2 - t1) / dt)
    #spike_times = np.sort(spike_times)
    bin_idx = np.floor((spike_times - t1) / dt)
    bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < Nbins)]
    bin_idx = bin_idx.astype(np.int64)
    # Calculate firing rate dynamics
    rvec = np.bincount(bin_idx, minlength=Nbins)
    rvec = rvec / (dt * pop_sz * num_epochs)
    # Time samples
    tvec = np.arange(Nbins, dtype=np.float64) * dt + t1
    # Return the result
    return tvec, rvec