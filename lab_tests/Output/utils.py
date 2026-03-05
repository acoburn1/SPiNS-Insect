import numpy as np
import os


def load_mean_ci(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    mean = np.asarray(data["mean"], dtype=np.float64)
    ci_lo = np.asarray(data["ci_lo"], dtype=np.float64)
    ci_hi = np.asarray(data["ci_hi"], dtype=np.float64)
    return mean, ci_lo, ci_hi


def _load_significant_epochs(analysis_dir: str) -> list[int]:
    path = os.path.join(analysis_dir, "sige.npz")
    data = np.load(path)
    results = np.asarray(data["results"])
    sig_zero_based = np.flatnonzero(np.any(results.astype(bool), axis=0))
    return [int(e) for e in sig_zero_based]