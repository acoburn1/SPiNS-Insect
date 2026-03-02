import numpy as np
from Statistics.StatHelper import *
from DriverUtils.Zarr import load_slice

def get_epoch_stats(data_array: np.ndarray, ci=0.95):
    data_array = np.asarray(data_array, dtype=np.float64)
    E = int(data_array.shape[1])
    return AggregateStatsObject([StatsObject(data_array[:, e], ci=ci) for e in range(E)])

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
