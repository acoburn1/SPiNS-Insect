import os
import re
import numpy as np
from scipy.stats import linregress


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
        try:
            ratio_labels = list(metadata.get("ratio_labels", []))
            set_labels = list(metadata.get("set_labels", []))
            trial_counts = np.asarray(metadata["trial_counts"], dtype=np.float64)
        except:
            raise ValueError("Invalid RatioTest metadata format. Expected 'ratio_labels', 'set_labels', and 'trial_counts'.")
    else:
        raise ValueError("RatioTest.npz is missing metadata")

    return {
        "raw": raw,
        "ratio_labels": ratio_labels,
        "set_labels": set_labels,
        "trial_counts": trial_counts,
    }


def mod_count_from_ratio(ratio_label: str) -> int:
    left = str(ratio_label).split(":")[0].strip()
    return int(left)


def _weighted_average_over_sets(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    if x.ndim < 2:
        raise ValueError(f"Expected x to include a set axis, got shape {x.shape}")

    try:
        w_core = np.broadcast_to(w, x.shape[1:])
    except ValueError as exc:
        raise ValueError(
            f"Weight shape {w.shape} is not broadcastable to {x.shape[1:]}"
        ) from exc

    valid = np.isfinite(x)
    w_b = np.broadcast_to(w_core[None, ...], x.shape)

    weighted_sum = np.nansum(np.where(valid, x * w_b, np.nan), axis=-1)
    weight_sum = np.sum(np.where(valid, w_b, 0.0), axis=-1)

    out = np.full(x.shape[:-1], np.nan, dtype=np.float64)
    good = weight_sum > 0
    out[good] = weighted_sum[good] / weight_sum[good]
    return out


def weighted_ratio_average(x: np.ndarray, trial_counts: np.ndarray) -> np.ndarray:
    """
    x: (M, R, S)
    trial_counts: (R, S)

    Returns:
        (M, R) weighted average across sets for each ratio,
        using trial counts as fixed weights and ignoring NaNs.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x shape (M, R, S), got {x.shape}")
    return _weighted_average_over_sets(x, trial_counts)


def weighted_single_ratio(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    x: (M, E, S)
    weights: (S,)

    Returns:
        (M, E) weighted average across sets for the selected ratio,
        using trial counts as fixed weights and ignoring NaNs.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x shape (M, E, S), got {x.shape}")
    return _weighted_average_over_sets(x, weights)


def first_sig_epochs(analysis_dir: str, n_models: int, n_epochs: int, mode: str = "sig") -> np.ndarray:
    mode = str(mode).lower()
    filename = "sige.npz" if mode == "sig" else "wb-sige.npz" if mode == "wb-sig" else None
    if filename is None:
        raise ValueError(f"Unsupported significance mode: {mode}. Expected 'sig' or 'wb-sig'.")

    path = os.path.join(analysis_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {filename}: {path}")

    data = np.load(path, allow_pickle=True)
    if "results" not in data:
        raise ValueError(f"{filename} is missing 'results'.")

    sig = np.asarray(data["results"]).astype(bool)
    if sig.shape != (n_models, n_epochs):
        raise ValueError(f"Expected {mode} results shape {(n_models, n_epochs)}, got {sig.shape}")

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


def resolve_epoch_range(
    sub_cfg: dict,
    total_epochs: int,
    *,
    default_start: int = 0,
) -> list[int]:
    range_cfg = sub_cfg.get("range", {}) or {}
    start = int(range_cfg.get("start", default_start))
    stop_raw = range_cfg.get("stop", total_epochs - 1)
    stop = (total_epochs - 1) if stop_raw is None else int(stop_raw)
    step = int(range_cfg.get("step", 1))

    if step <= 0:
        raise ValueError(f"Epoch range step must be positive, got {step}.")
    if not 0 <= start < total_epochs:
        raise ValueError(f"Epoch range start {start} is out of bounds for 0..{total_epochs - 1}.")
    if not 0 <= stop < total_epochs:
        raise ValueError(f"Epoch range stop {stop} is out of bounds for 0..{total_epochs - 1}.")
    if start > stop:
        raise ValueError(f"Epoch range start {start} must be <= stop {stop}.")

    return list(range(start, stop + 1, step))


def finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    good = np.isfinite(xv) & np.isfinite(yv)
    return xv[good], yv[good]


def points_at_epochs(x_arr: np.ndarray, y_arr: np.ndarray, epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.full((x_arr.shape[0],), np.nan, dtype=np.float64)
    y = np.full((y_arr.shape[0],), np.nan, dtype=np.float64)

    for m in range(x_arr.shape[0]):
        e = epochs[m]
        if not np.isfinite(e):
            continue
        ei = int(e)
        x[m] = x_arr[m, ei]
        y[m] = y_arr[m, ei]

    return finite_xy(x, y)


def points_for_epoch_list(x_arr: np.ndarray, y_arr: np.ndarray, epoch_indices: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_list = []
    y_list = []
    for e in epoch_indices:
        x, y = finite_xy(x_arr[:, e], y_arr[:, e])
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


def shared_limits(
    arrays: list[np.ndarray],
    *,
    fallback: list[float] | None = None,
    clamp_01: bool = False,
    padding: float = 0.05,
) -> list[float] | None:
    if clamp_01:
        return [0.0, 1.0]

    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size > 0]) if arrays else np.asarray([], dtype=np.float64)
    if vals.size == 0:
        return fallback

    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if lo == hi:
        return [lo - 0.5, hi + 0.5]

    span = hi - lo
    return [lo - padding * span, hi + padding * span]


def fit_line_with_stats(x: np.ndarray, y: np.ndarray, *, label_prefix: str = "fit") -> tuple[np.ndarray, np.ndarray, str]:
    xv, yv = finite_xy(x, y)
    if xv.size < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), f"{label_prefix} unavailable"

    reg = linregress(xv, yv)
    x0 = float(np.nanmin(xv))
    x1 = float(np.nanmax(xv))
    if x0 == x1:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), f"{label_prefix} unavailable"

    fx = np.asarray([x0, x1], dtype=np.float64)
    fy = reg.slope * fx + reg.intercept
    return fx, fy, f"{label_prefix} (r={reg.rvalue:.3f}, p={reg.pvalue:.3g})"


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
