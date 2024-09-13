from dataclasses import dataclass, field
import os
from pathlib import Path
from pprint import pprint
import sys

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_keeper import DataKeeper


storage_dir = r'D:\WORK\Salvador\repo\sim_res_analyzer\data\test_data_keeper'
metadata_file = 'metadata.json'

dk = DataKeeper(storage_dir, metadata_file)

X1 = xr.DataArray(np.ones((10, 10)))
par1 = {'par11': 1.1, 'par12': '2'}
dk.store_data(X1, 'X1', par1)

X2 = xr.DataArray(2 * np.ones((10, 10)))
par2 = {'par21': 2.1, 'par22': 'ggg'}
dk.store_data(X1, 'X2', par2)

Y1 = dk.get_data('X1', par1)
print(np.all(X1 == Y1).values)
Y1.close()

@dataclass(frozen=True)
class ParamsV1:
    par1: float = 1
    par2: int = 2

@dataclass(frozen=True)
class ParamsV2:
    par1: float = 1
    par2: int = 2
    par3: str = 'ggg'
    
dk.store_data(X1, 'X3', ParamsV1(10, 20))
dk.store_data(X1, 'X3', ParamsV2(10, 20))  # should replace the previous one
dk.store_data(X1, 'X3', ParamsV2(10, 20, 'aaa'))

pprint(dk.list_data())