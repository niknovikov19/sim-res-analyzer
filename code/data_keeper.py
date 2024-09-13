from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
#import pickle as pkl
#from typing import Any, Dict, Optional

import numpy as np
import xarray as xr


# =============================================================================
# def _get_non_default_values(obj):
#     """Filter non-default values of a dataclass object. """
#     if not is_dataclass(obj):
#         return obj    # Not a dataclass - return the object as-is
#     obj1 = {field.name: getattr(obj, field.name) for field in fields(obj)
#             if getattr(obj, field.name) != field.default}
#     return obj1
# 
# def _params_to_dict(par):
#     par_dict = {}
#     for key, val in par.items():
#         if hasattr(val, '__dict__'):
#             par_dict[key] = val.__dict__
#         else:
#             par_dict[key] = val
#     return par_dict
# =============================================================================

class CustomEncoder(json.JSONEncoder):
    """JSON encoder that treats ndarrays and dataclasses. """
    def treat_dataclass(self, obj):
        return obj.__dict__    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            obj = int(obj)
        if is_dataclass(obj):
            obj = self.treat_dataclass(obj)
        return json.JSONEncoder.encode(self, obj)
    
class NonDefEncoder(CustomEncoder):
    """JSON encoder: remove dataclass fields with default values. """
    def treat_dataclass(self, obj):
        return {field.name: getattr(obj, field.name) for field in fields(obj)
                if getattr(obj, field.name) != field.default}
    

class DataKeeper:
    def __init__(self, storage_dir: str, metadata_file: str):
        # Directory where data will be stored
        self.storage_dir = Path(storage_dir)
        self.metadata_file = self.storage_dir / metadata_file
        # Metadata to track stored data
        self.data_index = self._load_metadata()
        
    def _hash_params(self, params: list) -> str:
        """Create a hash from params (dict) to generate a unique key for each data. """
        # Don't hash dataclass fields with default values (for backward compatibility)
        params_str = json.dumps(params, sort_keys=True, cls=NonDefEncoder)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _generate_data_key(self, name: str, params: list):
        """Generate a unique key for data based on its name and params. """
        return f'{name}_{self._hash_params(params)}'

    def _load_metadata(self) -> dict:
        """Load the metadata file that contains information about stored data. """
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata about stored data in a human-readable format (JSON). """
        with open(self.metadata_file, 'w') as f:
            json.dump(self.data_index, f, indent=4, cls=CustomEncoder)
            
    def exists(self, name, params):
        """Check if data exists. """
        key = self._generate_data_key(name, params)
        return (key in self.data_index)

    def get_data(self, name: str, params: list = None, **kwargs) -> xr.DataArray:
        """Lazy-load the data from disk if it exists. """
        if params is None: params = []
        if self.exists(name, params):
            key = self._generate_data_key(name, params)
            data_path = self.storage_dir / self.data_index[key]['file_name']
            return xr.open_dataarray(data_path, **kwargs)
        else:
            raise ValueError(f'Data "{name}" not found')

    def store_data(self, data: xr.DataArray, name: str, params: list = None,
                   allow_rewrite=True, **kwargs):
        """Store data to disk. """
        if params is None: params = []
        if self.exists(name, params) and not allow_rewrite:
            raise ValueError('Data rewriting is prohibited')
        key = self._generate_data_key(name, params)
        file_name = f'{key}.nc'
        data_path = self.storage_dir / file_name
        # Save metadata
        self.data_index[key] = {
            'name': name, 'params': params, 'file_name': file_name}
        self._save_metadata()
        # Save the data
        data.to_netcdf(data_path, engine='h5netcdf', **kwargs)

    def list_data(self):
        """Return a human-readable format of stored data information."""
        return self.data_index
    
    # TODO: implement chunking
    # Params list insteead of dict?
    # A custom decorator for json serialization of dataclass?
