# General
import json, inspect
from pathlib import Path
from typing import Any, Mapping
from numpy import ndarray

# Torch

# Wordle



def filtered_kwargs(cls, source: Mapping[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls).parameters
    return {k: v for k, v in source.items() if k in params}


class Config:
    def __init__(self):
        pass

    def _to_serializable(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple, ndarray)):
            return [self._to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._to_serializable(val) for k, val in obj.items()}
        return obj

    def to_dict(self):
        out = self.__dict__.copy()
        return self._to_serializable(out)
    
    def save(self, path: Path):
        cfg = self.to_dict()
        with path.open("w") as f:
            json.dump(cfg, f, indent=4)
