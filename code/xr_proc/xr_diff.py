import numpy as np
import xarray as xr

def _diff_keepdims(X, n, axis):
    """Calculate derivative with initial padding to preserve dimensions. """
    npad1 = int(n / 2)
    npad2 = n - npad1
    X = np.concatenate(
        [np.take(X, np.arange(npad1), axis),
         X,
         np.take(X, np.arange(-npad2, 0), axis)],
        axis=axis
        )
    return np.diff(X, n, axis)

def calc_xr_diff(X: xr.DataArray, n: int = 1, ydim='y', compute=False):
    """Calculate diff of an xarray along the y-dimension, supports dask. """
    #if X.dims[-1] != ydim:
    #    raise ValueError('"y" should be the last dimension')
    Y = xr.apply_ufunc(
        _diff_keepdims, X,
        input_core_dims=[[ydim]], output_core_dims=[[ydim]],
        kwargs={'n': n, 'axis': -1},  # because the core dim becomes last
        dask='parallelized', output_dtypes=[X.dtype]
    )
    Y = Y.assign_coords({ydim: X.coords[ydim]})
    Y = Y.transpose(*X.dims)
    if compute:
        Y = Y.compute()
    return Y