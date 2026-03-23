import numpy as np
import os
import re


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

def get_hyperparameter_runs_with_data(analysis_root: str, data_names: list[str]):
    runs = _discover_hyperparameter_runs(analysis_root)

    for run in runs:
        full = run["analysis_dir"]

        for name in data_names:
            filename = name + ".npz" if not name.endswith(".npz") else name

            path = os.path.join(full, filename)
            if not os.path.exists(path):
                continue
            
            data = np.load(path)
            run[name] = data

    return runs

def _discover_hyperparameter_runs(analysis_root: str) -> list[dict]:
    parent = os.path.dirname(analysis_root.rstrip("/\\"))
    base = os.path.basename(analysis_root.rstrip("/\\"))
    if not parent:
        parent = "."

    pattern = re.compile(rf"^{re.escape(base)}_hls(\d+)_lr([A-Za-z0-9p\-]+)$")

    runs = []
    for name in os.listdir(parent):
        full = os.path.join(parent, name)
        if not os.path.isdir(full):
            continue

        m = pattern.match(name)
        if not m:
            continue

        runs.append(
            {
                "analysis_dir": full,
                "hls": int(m.group(1)),
                "lr": _parse_lr_token(m.group(2)),
            }
        )

    return runs

def _parse_lr_token(token: str) -> float:
    return float(token.replace("p", "."))