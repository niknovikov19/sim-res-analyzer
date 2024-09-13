from dataclasses import dataclass, field, fields, is_dataclass
import json

import numpy as np


class CustomEncoder(json.JSONEncoder):
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
    def treat_dataclass(self, obj):
        return {field.name: getattr(obj, field.name) for field in fields(obj)
                if getattr(obj, field.name) != field.default}

@dataclass(frozen=True)
class DataObj:
    val: np.int64 = 1
    #val: int = 1
    lst: list = field(default_factory=list)
    tpl: tuple = (2, 3)

x = DataObj(1, ['a', 'b', DataObj(20)])
x = {'x': x, 'y': ('aa', 6)}

#s = json.dumps(x, sort_keys=True, cls=CustomEncoder)
s = json.dumps(x, sort_keys=True, cls=NonDefEncoder)
print(s.replace('\\', ''))