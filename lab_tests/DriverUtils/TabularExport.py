import csv
import os
import numpy as np

from Output.utils import (
    load_ratio_test_bundle,
    weighted_single_ratio,
    load_hidden_correlation_raw,
    load_k95_hidden_raw,
    valid_set_indices_for_ratio,
)
from Statistics.StatHelper import stats_over_models


def export_regression_tables(analysis_dir: str, output_dir: str) -> None:
    sig_dir = os.path.join(output_dir, "tabular_sig")
    range_dir = os.path.join(output_dir, "tabular_range")

    os.makedirs(sig_dir, exist_ok=True)
    os.makedirs(range_dir, exist_ok=True)

    _export_sig_mode(analysis_dir, os.path.join(sig_dir, "sige.csv"), corr_mode="standard")
    _export_sig_mode(analysis_dir, os.path.join(sig_dir, "wb-sige.csv"), corr_mode="wb")
    _export_range_mode(analysis_dir, os.path.join(range_dir, "standard"), corr_mode="standard", step=5)
    _export_range_mode(analysis_dir, os.path.join(range_dir, "wb"), corr_mode="standard", step=5)
    _export_sig_masks(analysis_dir, os.path.join(sig_dir, "tabular_masks"))


def _export_sig_masks(analysis_dir: str, masks_dir: str) -> None:
    os.makedirs(masks_dir, exist_ok=True)
    _write_mask_csv(os.path.join(analysis_dir, "sige.npz"), os.path.join(masks_dir, "sige.csv"))
    _write_mask_csv(os.path.join(analysis_dir, "wb-sige.npz"), os.path.join(masks_dir, "wb-sige.csv"))


def _write_mask_csv(npz_path: str, csv_path: str) -> None:
    data = np.load(npz_path, allow_pickle=True)
    if "results" not in data:
        raise ValueError(f"{npz_path} is missing 'results'.")

    mask = np.asarray(data["results"], dtype=np.float64)
    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape (M, E), got {mask.shape} from {npz_path}")

    n_epochs = int(mask.shape[1])
    fieldnames = ["model"] + [f"e{e}" for e in range(n_epochs)]
    rows = []
    for m in range(mask.shape[0]):
        row = {"model": f"m{m + 1}"}
        for e in range(n_epochs):
            v = mask[m, e]
            row[f"e{e}"] = "NaN" if not np.isfinite(v) else str(int(v != 0))
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _export_sig_mode(analysis_dir: str, csv_path: str, *, corr_mode: str) -> None:
    corr = load_hidden_correlation_raw(analysis_dir, mode="standard" if corr_mode == "standard" else "wb")
    k95 = load_k95_hidden_raw(analysis_dir)
    ratio_bundle = load_ratio_test_bundle(analysis_dir)

    ratio_raw = np.asarray(ratio_bundle["raw"], dtype=np.float64)
    ratio_labels = list(ratio_bundle["ratio_labels"])
    set_labels = list(ratio_bundle["set_labels"])
    trial_counts = np.asarray(ratio_bundle["trial_counts"], dtype=np.float64)

    if "3:3" not in ratio_labels:
        raise ValueError(f"Ratio label '3:3' not found in RatioTest metadata: {ratio_labels}")

    r_idx = ratio_labels.index("3:3")
    valid_set_indices = valid_set_indices_for_ratio(ratio_raw, r_idx)
    valid_set_labels = [set_labels[i] for i in valid_set_indices]

    n_models, n_epochs = corr["mod"].shape
    sig_mode = "sig" if corr_mode == "standard" else "wb-sig"
    sig_epochs = _first_sig_epochs_strict(analysis_dir, n_models, n_epochs, mode=sig_mode)

    rows = []
    for model_i in range(n_models):
        e = sig_epochs[model_i]
        if not np.isfinite(e):
            continue

        epoch = int(e)
        rows.append(_metric_row_for_epoch(
            model_i=model_i,
            epoch=epoch,
            corr=corr,
            k95=k95,
            ratio_raw=ratio_raw,
            r_idx=r_idx,
            valid_set_indices=valid_set_indices,
            valid_set_labels=valid_set_labels,
            trial_counts=trial_counts,
        ))

    metric_cols = _metric_cols(valid_set_labels)
    summary_rows = _build_summary_rows(rows, metric_cols)
    all_rows = summary_rows + rows

    fieldnames = ["model"] + _metric_cols(valid_set_labels)
    _write_csv(csv_path, fieldnames, all_rows)


def _export_range_mode(analysis_dir: str, output_dir: str, corr_mode: str, *, step: int = 5) -> None:
    os.makedirs(output_dir, exist_ok=True)
    corr = load_hidden_correlation_raw(analysis_dir, mode=corr_mode)
    k95 = load_k95_hidden_raw(analysis_dir)
    ratio_bundle = load_ratio_test_bundle(analysis_dir)

    ratio_raw = np.asarray(ratio_bundle["raw"], dtype=np.float64)
    ratio_labels = list(ratio_bundle["ratio_labels"])
    set_labels = list(ratio_bundle["set_labels"])
    trial_counts = np.asarray(ratio_bundle["trial_counts"], dtype=np.float64)

    if "3:3" not in ratio_labels:
        raise ValueError(f"Ratio label '3:3' not found in RatioTest metadata: {ratio_labels}")

    r_idx = ratio_labels.index("3:3")
    valid_set_indices = valid_set_indices_for_ratio(ratio_raw, r_idx)
    valid_set_labels = [set_labels[i] for i in valid_set_indices]
    metric_cols = _metric_cols(valid_set_labels)
    fieldnames = ["model"] + metric_cols

    n_models, n_epochs = corr["mod"].shape
    for epoch in range(0, n_epochs, step):
        rows = []
        for model_i in range(n_models):
            rows.append(_metric_row_for_epoch(
                model_i=model_i,
                epoch=epoch,
                corr=corr,
                k95=k95,
                ratio_raw=ratio_raw,
                r_idx=r_idx,
                valid_set_indices=valid_set_indices,
                valid_set_labels=valid_set_labels,
                trial_counts=trial_counts,
            ))

        summary_rows = _build_summary_rows(rows, metric_cols)
        all_rows = summary_rows + rows
        _write_csv(os.path.join(output_dir, f"tabular-e{epoch}.csv"), fieldnames, all_rows)


def _metric_row_for_epoch(
    *,
    model_i: int,
    epoch: int,
    corr: dict[str, np.ndarray],
    k95: np.ndarray,
    ratio_raw: np.ndarray,
    r_idx: int,
    valid_set_indices: list[int],
    valid_set_labels: list[str],
    trial_counts: np.ndarray,
) -> dict:
    mod_corr = corr["mod"][model_i, epoch]
    lat_corr = corr["lat"][model_i, epoch]
    mod_k95 = k95[model_i, epoch, 0]
    lat_k95 = k95[model_i, epoch, 1]

    ratio_33_by_set = ratio_raw[model_i, epoch, r_idx, :]
    weighted_pref = weighted_single_ratio(
        ratio_raw[model_i : model_i + 1, epoch : epoch + 1, r_idx, :],
        trial_counts[r_idx],
    )[0, 0]

    row = {
        "model": f"m{model_i + 1}",
        "mod_hidden_corr": mod_corr,
        "lat_hidden_corr": lat_corr,
        "mod_lat_hidden_corr": mod_corr - lat_corr,
        "mod_k95": mod_k95,
        "lat_k95": lat_k95,
        "mod_lat_k95": mod_k95 - lat_k95,
        "avg_mod_pref_3_3_weighted": weighted_pref,
    }

    for set_i, set_label in zip(valid_set_indices, valid_set_labels):
        row[f"avg_mod_pref_3_3_set_{set_label}"] = ratio_33_by_set[set_i]

    return row


def _metric_cols(valid_set_labels: list[str]) -> list[str]:
    return [
        "mod_hidden_corr",
        "lat_hidden_corr",
        "mod_lat_hidden_corr",
        "mod_k95",
        "lat_k95",
        "mod_lat_k95",
        "avg_mod_pref_3_3_weighted",
    ] + [f"avg_mod_pref_3_3_set_{s}" for s in valid_set_labels]


def _first_sig_epochs_strict(analysis_dir: str, n_models: int, n_epochs: int, *, mode: str) -> np.ndarray:
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

    raw = np.asarray(data["results"])
    if raw.shape != (n_models, n_epochs):
        raise ValueError(f"Expected {mode} results shape {(n_models, n_epochs)}, got {raw.shape}")

    if raw.dtype == np.bool_:
        sig_mask = raw
    else:
        sig_mask = np.isfinite(raw) & (raw != 0)

    first = np.full((n_models,), np.nan, dtype=np.float64)
    for model_i in range(n_models):
        hits = np.flatnonzero(sig_mask[model_i])
        if hits.size > 0:
            first[model_i] = float(hits[0])

    return first


def _build_summary_rows(model_rows: list[dict], metric_cols: list[str]) -> list[dict]:
    if not model_rows:
        return []

    matrix = np.full((len(model_rows), len(metric_cols)), np.nan, dtype=np.float64)
    for i, row in enumerate(model_rows):
        for j, col in enumerate(metric_cols):
            val = row.get(col, np.nan)
            matrix[i, j] = float(val) if np.isfinite(val) else np.nan

    st = stats_over_models(matrix)
    summary_specs = [
        ("mean", np.asarray(st["mean"], dtype=np.float64)),
        ("std", np.asarray(st["std"], dtype=np.float64)),
        ("se", np.asarray(st["se"], dtype=np.float64)),
        ("cilow", np.asarray(st["ci_lo"], dtype=np.float64)),
        ("cihigh", np.asarray(st["ci_hi"], dtype=np.float64)),
    ]

    summary_rows = []
    for name, vals in summary_specs:
        row = {"model": name}
        for j, col in enumerate(metric_cols):
            row[col] = vals[j]
        summary_rows.append(row)

    return summary_rows


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row.get(k, np.nan)
                if isinstance(v, (np.floating, float)):
                    out[k] = "NaN" if not np.isfinite(v) else f"{float(v):.10g}"
                else:
                    out[k] = v
            writer.writerow(out)
