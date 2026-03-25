import os
import re
import numpy as np


def load_mean_ci(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    mean = np.asarray(data["mean"], dtype=np.float64)
    ci_lo = np.asarray(data["ci_lo"], dtype=np.float64)
    ci_hi = np.asarray(data["ci_hi"], dtype=np.float64)
    return mean, ci_lo, ci_hi


def load_ratio_test_bundle(analysis_dir: str) -> dict[str, np.ndarray | list[str]]:
    path = os.path.join(analysis_dir, "RatioTest.npz")
    data = np.load(path, allow_pickle=True)

    raw = np.asarray(data["raw"], dtype=np.float64)

    metadata = None
    if "metadata" in data:
        metadata = data["metadata"].item()

    ratio_labels = None
    set_labels = None
    trial_counts = None

    if metadata is not None:
        ratio_labels = list(metadata.get("ratio_labels", []))
        set_labels = list(metadata.get("set_labels", []))

        if "trial_counts" in metadata:
            trial_counts = np.asarray(metadata["trial_counts"], dtype=np.float64)

    if (not ratio_labels) and "ratio_labels" in data:
        ratio_labels = [str(v) for v in data["ratio_labels"].tolist()]

    if (not set_labels) and "set_labels" in data:
        set_labels = [str(v) for v in data["set_labels"].tolist()]

    if trial_counts is None and "trial_counts" in data:
        trial_counts = np.asarray(data["trial_counts"], dtype=np.float64)

    return {
        "raw": raw,
        "ratio_labels": ratio_labels or [],
        "set_labels": set_labels or [],
        "trial_counts": trial_counts,
    }


def mod_count_from_ratio(ratio_label: str) -> int:
    left = str(ratio_label).split(":")[0].strip()
    return int(left)


def weighted_ratio_average(x: np.ndarray, trial_counts: np.ndarray) -> np.ndarray:
    """
    x: (M, R, S)
    trial_counts: (R, S)

    Returns:
        (M, R) weighted average across sets for each ratio,
        using trial counts as fixed weights and ignoring NaNs.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(trial_counts, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError(f"Expected x shape (M, R, S), got {x.shape}")
    if w.shape != x.shape[1:]:
        raise ValueError(f"Weight shape {w.shape} does not match ratio/set shape {x.shape[1:]}")

    valid = np.isfinite(x)
    w_b = np.broadcast_to(w[None, :, :], x.shape)

    weighted_sum = np.nansum(np.where(valid, x * w_b, np.nan), axis=2)
    weight_sum = np.sum(np.where(valid, w_b, 0.0), axis=2)

    out = np.full((x.shape[0], x.shape[1]), np.nan, dtype=np.float64)
    good = weight_sum > 0
    out[good] = weighted_sum[good] / weight_sum[good]
    return out


def weighted_single_ratio(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    x: (M, E, S)
    weights: (S,)

    Returns:
        (M, E) weighted average across sets for the selected ratio,
        using trial counts as fixed weights and ignoring NaNs.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError(f"Expected x shape (M, E, S), got {x.shape}")
    if w.shape != (x.shape[2],):
        raise ValueError(f"Weight shape {w.shape} does not match set axis {(x.shape[2],)}")

    valid = np.isfinite(x)
    w_b = np.broadcast_to(w[None, None, :], x.shape)

    weighted_sum = np.nansum(np.where(valid, x * w_b, np.nan), axis=2)
    weight_sum = np.sum(np.where(valid, w_b, 0.0), axis=2)

    out = np.full((x.shape[0], x.shape[1]), np.nan, dtype=np.float64)
    good = weight_sum > 0
    out[good] = weighted_sum[good] / weight_sum[good]
    return out


def first_sig_epochs(analysis_dir: str, n_models: int, n_epochs: int) -> np.ndarray:
    path = os.path.join(analysis_dir, "sige.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing sige.npz: {path}")

    data = np.load(path, allow_pickle=True)
    if "results" not in data:
        raise ValueError("sige.npz is missing 'results'.")

    sig = np.asarray(data["results"]).astype(bool)
    if sig.shape != (n_models, n_epochs):
        raise ValueError(f"Expected sige results shape {(n_models, n_epochs)}, got {sig.shape}")

    out = np.full((n_models,), np.nan, dtype=np.float64)
    for m in range(n_models):
        idx = np.flatnonzero(sig[m])
        if idx.size > 0:
            out[m] = float(idx[0])

    return out


def spread_x(center: float, n: int, width: float = 0.18) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    if n == 1:
        return np.asarray([center], dtype=np.float64)
    return np.linspace(center - width / 2.0, center + width / 2.0, n, dtype=np.float64)


def normalize_k95_raw(raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float64)

    if x.ndim == 4:
        if x.shape[2] != 2 or x.shape[3] != 1:
            raise ValueError(f"Expected K95 raw shape (M, E, 2, 1), got {x.shape}")
        x = x[..., 0]
    elif x.ndim == 3:
        if x.shape[2] != 2:
            raise ValueError(f"Expected K95 raw shape (M, E, 2), got {x.shape}")
    else:
        raise ValueError(f"Expected K95 raw ndim 3 or 4, got shape {x.shape}")

    return x


def avg_over_epochs_per_model(raw: np.ndarray) -> np.ndarray:
    x = normalize_k95_raw(raw)
    return np.nanmean(x, axis=1)


def epoch_k95_summaries(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = normalize_k95_raw(raw)
    mod = x[:, :, 0]
    lat = x[:, :, 1]

    mean_avg = np.nanmean((mod + lat) / 2.0, axis=0)
    diff = np.nanmean(mod - lat, axis=0)

    return mean_avg, diff


def sample_epochs(raw: np.ndarray, step: int = 3) -> np.ndarray:
    x = np.asarray(raw)
    if x.ndim < 2:
        raise ValueError(f"Expected raw with epoch axis, got shape {x.shape}")

    n_epochs = int(x.shape[1])
    last = n_epochs // 2
    return np.arange(0, last + 1, step, dtype=np.int64)


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
