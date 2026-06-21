import numpy as np
from Statistics.StatHelper import *
from DriverUtils.Zarr import load_slice

def get_onehot_ids(probe_index: dict) -> np.ndarray:
    ids = probe_index.get("source=onehot", None)
    if ids is None:
        raise KeyError("probe_index missing key 'source=onehot'")
    return np.asarray(ids, dtype=np.int64)

def get_ratio_ids(probe_index: dict, ratio: str) -> np.ndarray:
    ids = probe_index.get(f"ratio={ratio}", None)
    if ids is None:
        return np.asarray([], dtype=np.int64)
    return np.asarray(ids, dtype=np.int64)

def assert_data_shape(eval: list[int], cfg: list[int], names: list[str]=None):
    for (e, c, n) in zip(eval, cfg, names if names is not None else ["param"]*len(eval)):
        assert e == c, f"Expected {n}={c} from config but got {e} in eval"