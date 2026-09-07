import csv
import os
import re
import numpy as np
from scipy.stats import linregress
from Statistics.StatHelper import stats_over_models


def load_mean_ci(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    mean = np.asarray(data["mean"], dtype=np.float64)
    ci_lo = np.asarray(data["ci_lo"], dtype=np.float64)
    ci_hi = np.asarray(data["ci_hi"], dtype=np.float64)
    return mean, ci_lo, ci_hi


def export_epoch_series_csv(
    figure_id: str,
    series_data: dict[str, np.ndarray],
    output_dir: str,
    *,
    metadata: dict[str, object] | None = None,
    model_id_prefix: str = "",
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    metadata = metadata or {}
    reserved = {"model_id", "epoch"}
    for name in series_data:
        reserved.add(name)
        reserved.update(
            f"{name}_{suffix}"
            for suffix in ("n", "mean", "std", "se", "ci_lower", "ci_upper")
        )
    overlap = reserved.intersection(metadata)
    if overlap:
        raise ValueError(f"CSV metadata uses reserved field(s): {sorted(overlap)}")
    if not series_data:
        raise ValueError(f"No series data provided for '{figure_id}'.")

    shape = None
    arrays = {}
    for label, values in series_data.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"Series '{label}' must have shape (models, epochs), got {array.shape}.")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"Series '{label}' has shape {array.shape}; expected {shape}.")
        arrays[label] = array

    models_path = os.path.join(output_dir, f"{figure_id}_models.csv")
    model_fields = [
        *metadata.keys(),
        "model_id",
        "epoch",
        *arrays.keys(),
    ]
    with open(models_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=model_fields)
        writer.writeheader()
        n_models, n_epochs = shape
        for model_i in range(n_models):
            for epoch in range(n_epochs):
                writer.writerow(
                    {
                        **metadata,
                        "model_id": f"{model_id_prefix}m{model_i + 1:03d}",
                        "epoch": epoch,
                        **{
                            name: _csv_number(array[model_i, epoch])
                            for name, array in arrays.items()
                        },
                    }
                )

    summary_path = os.path.join(output_dir, f"{figure_id}_summary.csv")
    summary_fields = [
        *metadata.keys(),
        "epoch",
        *[
            f"{name}_{suffix}"
            for name in arrays
            for suffix in ("n", "mean", "std", "se", "ci_lower", "ci_upper")
        ],
    ]
    series_stats = {name: stats_over_models(array) for name, array in arrays.items()}
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for epoch in range(n_epochs):
            row = {**metadata, "epoch": epoch}
            for name, stats in series_stats.items():
                row.update(
                    {
                        f"{name}_n": int(stats["n"][epoch]),
                        f"{name}_mean": _csv_number(stats["mean"][epoch]),
                        f"{name}_std": _csv_number(stats["std"][epoch]),
                        f"{name}_se": _csv_number(stats["se"][epoch]),
                        f"{name}_ci_lower": _csv_number(stats["ci_lo"][epoch]),
                        f"{name}_ci_upper": _csv_number(stats["ci_hi"][epoch]),
                    }
                )
            writer.writerow(row)

    return [models_path, summary_path]


def _csv_number(value) -> str:
    value = float(value)
    return "NaN" if not np.isfinite(value) else f"{value:.10g}"


def hyperparameters_from_analysis_dir(analysis_dir: str) -> tuple[int, float, str]:
    name = os.path.basename(analysis_dir.rstrip("/\\"))
    match = re.search(r"_hls(\d+)_lr([A-Za-z0-9p\-]+)$", name)
    if match is None:
        raise ValueError(f"Could not parse HLS and learning rate from analysis directory: {analysis_dir}")

    hls = int(match.group(1))
    learning_rate = _parse_lr_token(match.group(2))
    model_id_prefix = f"hls{hls}_lr{match.group(2)}_"
    return hls, learning_rate, model_id_prefix


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


def valid_set_indices_for_ratio(raw: np.ndarray, ratio_index: int) -> list[int]:
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError(f"Expected ratio raw shape (M, E, R, S), got {x.shape}")
    if not (0 <= ratio_index < x.shape[2]):
        raise ValueError(f"ratio_index {ratio_index} out of bounds for ratio axis size {x.shape[2]}")

    set_mask = np.any(np.isfinite(x[:, :, ratio_index, :]), axis=(0, 1))
    return [i for i, ok in enumerate(set_mask) if bool(ok)]


def resolve_requested_set_indices(sub_cfg: dict, set_labels: list[str]) -> list[int]:
    requested = sub_cfg.get("sets")
    if requested is None:
        return list(range(len(set_labels)))

    requested_labels = [str(v) for v in requested]
    seen = set()
    missing = []
    indices = []

    for label in requested_labels:
        if label in seen:
            continue
        seen.add(label)
        if label not in set_labels:
            missing.append(label)
            continue
        indices.append(set_labels.index(label))

    if missing:
        raise ValueError(f"Requested set label(s) not found: {missing}. Available: {set_labels}")
    if not indices:
        raise ValueError("No set labels were selected. Provide at least one entry in sub_cfg['sets'].")

    return indices


def filename_safe_token(text: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(text).strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "na"


def selected_sets_suffix(selected_labels: list[str], all_labels: list[str]) -> str:
    if len(selected_labels) == len(all_labels) and set(selected_labels) == set(all_labels):
        return ""
    tokens = [filename_safe_token(label) for label in selected_labels]
    return "_sets_" + "__".join(tokens)


def load_hidden_correlation_raw(analysis_dir: str, *, mode: str = "standard") -> dict[str, np.ndarray]:
    raw = load_correlation_raw(analysis_dir, corr_type=mode)
    mode = str(mode).lower()
    if mode == "standard":
        return {
            "mod": raw[:, :, 0, 0, 0],
            "lat": raw[:, :, 0, 1, 0],
        }
    if mode == "wb":
        return {
            "mod": raw[:, :, 0, 0],
            "lat": raw[:, :, 0, 1],
        }
    raise ValueError(f"Unsupported hidden correlation mode: {mode}. Expected 'standard' or 'wb'.")


def correlation_filename(corr_type: str) -> str:
    c = str(corr_type).lower()
    if c == "standard":
        return "Correlation.npz"
    if c == "wb":
        return "WithinVsBetweenCorrelation.npz"
    raise ValueError(f"Unsupported corr_type: {corr_type}. Expected 'standard' or 'wb'.")


def corr_type_from_cfg(sub_cfg: dict, *, default: str = "standard") -> str:
    corr_type = str(sub_cfg.get("corr_type", default)).lower()
    if corr_type not in ("standard", "wb"):
        raise ValueError(f"Unsupported corr_type: {corr_type}. Expected 'standard' or 'wb'.")
    return corr_type


def load_correlation_raw(analysis_dir: str, *, corr_type: str = "standard") -> np.ndarray:
    filename = correlation_filename(corr_type)
    data = np.load(os.path.join(analysis_dir, filename))
    raw = np.asarray(data["raw"], dtype=np.float64)

    corr_type = str(corr_type).lower()
    if corr_type == "standard":
        if raw.ndim != 5 or raw.shape[2:] != (2, 2, 2):
            raise ValueError(f"Expected Correlation raw shape (M,E,2,2,2), got {raw.shape}")
        return raw

    if corr_type == "wb":
        if raw.ndim != 4 or raw.shape[2:] != (2, 2):
            raise ValueError(f"Expected WithinVsBetweenCorrelation raw shape (M, E, 2, 2), got {raw.shape}")
        return raw

    raise ValueError(f"Unsupported corr_type: {corr_type}. Expected 'standard' or 'wb'.")


def load_k95_hidden_raw(analysis_dir: str) -> np.ndarray:
    data = np.load(os.path.join(analysis_dir, "K95.npz"))
    return normalize_k95_raw(np.asarray(data["raw"], dtype=np.float64))


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
