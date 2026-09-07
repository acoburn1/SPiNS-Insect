import csv
import os
from dataclasses import dataclass

import numpy as np
from scipy.stats import ttest_1samp

from Output.schema.OutputSpec import OutputSpec
from Output.utils import (
    get_hyperparameter_runs_with_data,
    load_ratio_test_bundle,
    resolve_requested_set_indices,
    weighted_ratio_average,
    weighted_single_ratio,
)


@dataclass
class GeneralizationBiasCell:
    hls: int
    learning_rate: float
    selected_epoch: int
    model_preferences: np.ndarray
    ratio_labels: list[str]
    model_ratio_preferences: np.ndarray
    mean_modular_preference: float
    signed_bias: float
    t_statistic: float
    p_value: float


@dataclass
class GeneralizationBiasData:
    start: int
    stop: int
    cells: list[GeneralizationBiasCell]


class GeneralizationBiasHeatmapOutput:
    name = "GeneralizationBiasHeatmap"
    hyperd = True

    def generate_output(self, sub_cfg: dict, analysis_root: str) -> list[OutputSpec]:
        data = _collect_generalization_bias_data(sub_cfg, analysis_root)
        hls_values = sorted({cell.hls for cell in data.cells}, reverse=True)
        lr_values = sorted({cell.learning_rate for cell in data.cells})
        hls_indices = {hls: i for i, hls in enumerate(hls_values)}
        lr_indices = {lr: i for i, lr in enumerate(lr_values)}

        bias_matrix = np.full((len(hls_values), len(lr_values)), np.nan, dtype=np.float64)
        epoch_matrix = np.full_like(bias_matrix, np.nan)
        significance_annotations = np.full(bias_matrix.shape, "", dtype=object)

        for cell in data.cells:
            row = hls_indices[cell.hls]
            column = lr_indices[cell.learning_rate]
            bias_matrix[row, column] = cell.signed_bias
            epoch_matrix[row, column] = cell.selected_epoch
            if np.isfinite(cell.p_value) and cell.p_value < 0.05:
                significance_annotations[row, column] = "•"

        range_token = f"e{data.start:03d}_e{data.stop:03d}"
        common = {
            "x_label": "Learning Rate",
            "y_label": "Hidden Layer Size",
            "grid": False,
            "figsize": tuple(sub_cfg.get("figsize", (12, 8))),
            "dpi": int(sub_cfg.get("dpi", 300)),
            "x_ticks": [float(i) for i in range(len(lr_values))],
            "x_ticklabels": [f"{lr:g}" for lr in lr_values],
            "y_ticks": [float(i) for i in range(len(hls_values))],
            "y_ticklabels": [str(hls) for hls in hls_values],
        }

        specs = [
            OutputSpec(
                figure_id=f"generalization_bias_heatmap_{range_token}",
                title=f"Maximum Generalization Bias (Epochs {data.start}-{data.stop}; • uncorrected p < .05)",
                matrix=bias_matrix.tolist(),
                matrix_annotations=significance_annotations.tolist(),
                matrix_vmin=-0.5,
                matrix_vmax=0.5,
                matrix_split=0.0,
                matrix_cmap_neg="Reds",
                matrix_cmap_pos="Blues",
                matrix_center_color="white",
                **common,
            )
        ]

        if bool(sub_cfg.get("epoch_map", False)):
            specs.append(
                OutputSpec(
                    figure_id=f"generalization_bias_epoch_heatmap_{range_token}",
                    title=f"Epoch of Maximum Generalization Bias (Epochs {data.start}-{data.stop})",
                    matrix=epoch_matrix.tolist(),
                    matrix_vmin=float(data.start),
                    matrix_vmax=float(data.stop),
                    matrix_cmap="gray_r",
                    **common,
                )
            )

        return specs

    def export_data(self, sub_cfg: dict, analysis_root: str, output_dir: str) -> list[str]:
        data = _collect_generalization_bias_data(sub_cfg, analysis_root)
        os.makedirs(output_dir, exist_ok=True)

        range_token = f"e{data.start:03d}_e{data.stop:03d}"
        models_path = os.path.join(output_dir, f"generalization_bias_models_{range_token}.csv")
        summary_path = os.path.join(output_dir, f"generalization_bias_summary_{range_token}.csv")
        scurve_path = os.path.join(output_dir, f"generalization_bias_scurve_models_{range_token}.csv")

        _write_model_csv(models_path, data.cells)
        _write_summary_csv(summary_path, data.cells)
        _write_scurve_csv(scurve_path, data.cells)
        return [models_path, summary_path, scurve_path]


def _epoch_range_from_cfg(sub_cfg: dict) -> tuple[int, int]:
    epoch_range = sub_cfg.get("range")
    if not isinstance(epoch_range, (list, tuple)) or len(epoch_range) != 2:
        raise ValueError("GeneralizationBiasHeatmap range must be [start, stop], inclusive.")

    start, stop = (int(value) for value in epoch_range)
    if start < 0:
        raise ValueError(f"Epoch range start must be non-negative, got {start}.")
    if start > stop:
        raise ValueError(f"Epoch range start {start} must be <= stop {stop}.")
    return start, stop


def _collect_generalization_bias_data(sub_cfg: dict, analysis_root: str) -> GeneralizationBiasData:
    start, stop = _epoch_range_from_cfg(sub_cfg)
    runs = get_hyperparameter_runs_with_data(analysis_root, ["RatioTest"])
    runs = [run for run in runs if "RatioTest" in run]

    if not runs:
        raise FileNotFoundError(f"No RatioTest.npz files found under {analysis_root}")

    cells = [_generalization_bias_cell(run, sub_cfg, start, stop) for run in runs]
    return GeneralizationBiasData(start=start, stop=stop, cells=cells)


def _generalization_bias_cell(
    run: dict,
    sub_cfg: dict,
    start: int,
    stop: int,
) -> GeneralizationBiasCell:
    analysis_dir = run["analysis_dir"]
    bundle = load_ratio_test_bundle(analysis_dir)
    raw = np.asarray(bundle["raw"], dtype=np.float64)
    ratio_labels = list(bundle["ratio_labels"])
    set_labels = list(bundle["set_labels"])
    trial_counts = np.asarray(bundle["trial_counts"], dtype=np.float64)

    ratio_name = "3:3"
    if ratio_name not in ratio_labels:
        raise ValueError(f"Requested ratio '{ratio_name}' not found. Available: {ratio_labels}")
    if stop >= raw.shape[1]:
        raise ValueError(
            f"Epoch range stop {stop} is out of bounds for {analysis_dir}; "
            f"available epochs are 0..{raw.shape[1] - 1}."
        )

    ratio_index = ratio_labels.index(ratio_name)
    set_indices = resolve_requested_set_indices(sub_cfg, set_labels)
    scurve_set_indices = resolve_requested_set_indices(
        {"sets": sub_cfg.get("scurve_sets")},
        set_labels,
    )
    per_model_preference = weighted_single_ratio(
        raw[:, :, ratio_index, set_indices],
        trial_counts[ratio_index, set_indices],
    )
    with np.errstate(invalid="ignore"):
        mean_preference = np.nanmean(per_model_preference, axis=0)

    selected = mean_preference[start : stop + 1]
    if not np.any(np.isfinite(selected)):
        raise ValueError(f"No finite modular-preference values in epoch range {start}..{stop} for {analysis_dir}.")

    local_epoch = int(np.nanargmax(np.abs(selected - 0.5)))
    epoch = start + local_epoch
    values = per_model_preference[:, epoch]
    model_ratio_preferences = weighted_ratio_average(
        raw[:, epoch, :, :],
        trial_counts,
    )
    model_ratio_preferences[:, ratio_index] = weighted_single_ratio(
        raw[:, epoch : epoch + 1, ratio_index, scurve_set_indices],
        trial_counts[ratio_index, scurve_set_indices],
    )[:, 0]
    finite_values = values[np.isfinite(values)]
    t_statistic = np.nan
    p_value = np.nan
    if finite_values.size >= 2:
        result = ttest_1samp(finite_values, popmean=0.5)
        if np.isfinite(result.statistic):
            t_statistic = float(result.statistic)
        if np.isfinite(result.pvalue):
            p_value = float(result.pvalue)

    return GeneralizationBiasCell(
        hls=int(run["hls"]),
        learning_rate=float(run["lr"]),
        selected_epoch=epoch,
        model_preferences=np.asarray(values, dtype=np.float64),
        ratio_labels=ratio_labels,
        model_ratio_preferences=np.asarray(model_ratio_preferences, dtype=np.float64),
        mean_modular_preference=float(mean_preference[epoch]),
        signed_bias=float(mean_preference[epoch] - 0.5),
        t_statistic=t_statistic,
        p_value=p_value,
    )


def _write_model_csv(path: str, cells: list[GeneralizationBiasCell]) -> None:
    fieldnames = [
        "hls",
        "learning_rate",
        "model_id",
        "selected_epoch",
        "modular_preference",
        "signed_bias",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for cell in sorted(cells, key=lambda item: (item.hls, item.learning_rate)):
            lr_token = f"{cell.learning_rate:g}".replace(".", "p")
            for model_index, preference in enumerate(cell.model_preferences, start=1):
                writer.writerow(
                    {
                        "hls": cell.hls,
                        "learning_rate": cell.learning_rate,
                        "model_id": f"hls{cell.hls}_lr{lr_token}_m{model_index:03d}",
                        "selected_epoch": cell.selected_epoch,
                        "modular_preference": float(preference),
                        "signed_bias": float(preference - 0.5),
                    }
                )


def _write_summary_csv(path: str, cells: list[GeneralizationBiasCell]) -> None:
    fieldnames = [
        "hls",
        "learning_rate",
        "n_models",
        "selected_epoch",
        "mean_modular_preference",
        "signed_bias",
        "t_statistic",
        "p_value",
        "significant",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for cell in sorted(cells, key=lambda item: (item.hls, item.learning_rate)):
            writer.writerow(
                {
                    "hls": cell.hls,
                    "learning_rate": cell.learning_rate,
                    "n_models": int(cell.model_preferences.size),
                    "selected_epoch": cell.selected_epoch,
                    "mean_modular_preference": cell.mean_modular_preference,
                    "signed_bias": cell.signed_bias,
                    "t_statistic": cell.t_statistic,
                    "p_value": cell.p_value,
                    "significant": bool(np.isfinite(cell.p_value) and cell.p_value < 0.05),
                }
            )


def _write_scurve_csv(path: str, cells: list[GeneralizationBiasCell]) -> None:
    if not cells:
        raise ValueError("Cannot export S-curve data without any hyperparameter cells.")

    ratio_labels = cells[0].ratio_labels
    if not ratio_labels:
        raise ValueError("Cannot export S-curve data without ratio labels.")

    ratio_fields = [f"modular_preference_{label.replace(':', '_')}" for label in ratio_labels]
    if len(set(ratio_fields)) != len(ratio_fields):
        raise ValueError(f"Ratio labels do not produce unique CSV columns: {ratio_labels}")

    fieldnames = [
        "hls",
        "learning_rate",
        "model_id",
        "selected_epoch",
        *ratio_fields,
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for cell in sorted(cells, key=lambda item: (item.hls, item.learning_rate)):
            if cell.ratio_labels != ratio_labels:
                raise ValueError(
                    f"Ratio labels differ across hyperparameter cells: {cell.ratio_labels} != {ratio_labels}"
                )
            expected_shape = (cell.model_preferences.size, len(ratio_labels))
            if cell.model_ratio_preferences.shape != expected_shape:
                raise ValueError(
                    f"Expected model ratio preferences shape {expected_shape}, "
                    f"got {cell.model_ratio_preferences.shape}."
                )

            lr_token = f"{cell.learning_rate:g}".replace(".", "p")
            for model_index, preferences in enumerate(cell.model_ratio_preferences, start=1):
                row = {
                    "hls": cell.hls,
                    "learning_rate": cell.learning_rate,
                    "model_id": f"hls{cell.hls}_lr{lr_token}_m{model_index:03d}",
                    "selected_epoch": cell.selected_epoch,
                }
                row.update(
                    {
                        field: float(preference)
                        for field, preference in zip(ratio_fields, preferences)
                    }
                )
                writer.writerow(row)
