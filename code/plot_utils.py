import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_xarray_2d(X: xr.DataArray, need_log=False, show_ax_names=True,
                   clim=(None, None)):
    coord_x = X.dims[1]
    coord_y = X.dims[0]
    cx = X.coords[coord_x].values
    cy = X.coords[coord_y].values
    ext = [cx[0], cx[-1], cy[-1], cy[0]]
    X_ = X.values
    if need_log:
        X_ = np.log(X_)
    plt.imshow(X_, aspect='auto', extent=ext, origin='upper',
               vmin=clim[0], vmax=clim[1])
    plt.xlim(cx[0], cx[-1])
    plt.ylim(cy[-1], cy[0])
    if show_ax_names:
        plt.xlabel(coord_x)
        plt.ylabel(coord_y)

def plot_xarray_2d_irreg(X: xr.DataArray, need_log=False, cmap=None):
    coord_x = X.dims[1]
    coord_y = X.dims[0]
    cx = X.coords[coord_x].values
    cy = X.coords[coord_y].values
    Cx, Cy = np.meshgrid(cx, cy)
    ax = plt.gca()
    X_ = X.values
    if need_log:
        X_ = np.log(X_)
    if cmap is not None:
        cmap = plt.get_cmap(cmap)
    m = ax.pcolormesh(Cx, Cy, X_, shading='auto', cmap=cmap)
    #ax.set_xscale('log')
    ax.invert_yaxis()
    plt.xlabel(coord_x)
    plt.ylabel(coord_y)
    plt.colorbar(m)
    plt.title(X.name)