import os
import numpy as np
from Output.schema.OutputSpec import *


class AllModelsSCurveOutput:
    name = "AllModelsSCurve"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        path = os.path.join(analysis_dir, "RatioTest.npz")
        data = np.load(path, allow_pickle=True)

        raw = np.asarray(data["raw"], dtype=np.float64)  # (M, E, R, S)

        metadata = None
        if "metadata" in data:
            metadata = data["metadata"].item()

        ratio_labels = None
        trial_counts = None

        if metadata is not None:
            ratio_labels = list(metadata.get("ratio_labels", []))
            if "trial_counts" in metadata:
                trial_counts = np.asarray(metadata["trial_counts"], dtype=np.float64)

        if (not ratio_labels) and "ratio_labels" in data:
            ratio_labels = [str(v) for v in data["ratio_labels"].tolist()]

        if trial_counts is None and "trial_counts" in data:
            trial_counts = np.asarray(data["trial_counts"], dtype=np.float64)

        if not ratio_labels:
            raise ValueError("RatioTest.npz is missing ratio_labels metadata.")
        if trial_counts is None:
            raise ValueError("RatioTest.npz is missing trial_counts metadata.")

        if trial_counts.shape != raw.shape[2:4]:
            raise ValueError(
                f"trial_counts shape {trial_counts.shape} does not match raw ratio/set shape {raw.shape[2:4]}"
            )

        x_vals = np.asarray([_mod_count_from_ratio(r) for r in ratio_labels], dtype=np.float64)

        mode = str(sub_cfg.get("epochs", "all")).lower()
        if mode == "all":
            return _build_all_epoch_specs(sub_cfg, raw, x_vals, trial_counts)
        elif mode == "sig":
            return [_build_sig_spec(sub_cfg, analysis_dir, raw, x_vals, trial_counts)]
        else:
            raise ValueError(f"Unsupported epochs mode: {mode}. Expected 'all' or 'sig'.")


def _build_all_epoch_specs(sub_cfg: dict, raw: np.ndarray, x_vals: np.ndarray, trial_counts: np.ndarray) -> list[OutputSpec]:
    M, E, R, S = raw.shape
    specs = []

    for e in range(E):
        per_model_curve = _weighted_ratio_average(raw[:, e, :, :], trial_counts)

        series_list = []
        for m in range(M):
            y = per_model_curve[m]
            if m == 0:
                label = "models"
            else:
                label = "_nolabel_"
            series_list.append(
                Series(
                        kind=PlotKind.LINE,
                        label=label,
                        x=[float(v) for v in x_vals],
                        y=[float(v) for v in y],
                        color=Color.BLUE,
                        y_axis=YAxis.LEFT,
                        marker="o",
                        markersize=3.0,
                        linewidth=.8,
                        alpha=.4
                    )
                )

        specs.append(
            OutputSpec(
                figure_id=f"all_models_{'s_curve'}_e{e:03d}",
                title="Modular Response by Feature Composition - Hidden",
                x_label="# mod feats",
                y_label="% mod resp",
                x_lim=[-0.3, 6.3],
                y_lim=[0.0, 1.0],
                legend_loc="legend_loc",
                legend_fontsize="legend_fontsize",
                figsize=(12, 8),
                dpi=300,
                x_ref=[
                    RLine(
                        val=float(sub_cfg.get("mid_x", 3.0)),
                        color=Color.GRAY,
                        linestyle=LineStyle.DASHED,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                ],
                y_ref=[
                    RLine(
                        val=0.5,
                        color=Color.GRAY,
                        linestyle=LineStyle.DASHED,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                ],
                series_list=series_list
            )
        )

    return specs


def _build_sig_spec(sub_cfg: dict, analysis_dir: str, raw: np.ndarray, x_vals: np.ndarray, trial_counts: np.ndarray) -> OutputSpec:
    sig_epochs = _first_sig_epochs(analysis_dir, raw.shape[0], raw.shape[1])

    per_model_curves = np.full((raw.shape[0], raw.shape[2]), np.nan, dtype=np.float64)

    for m in range(raw.shape[0]):
        e = sig_epochs[m]
        if not np.isfinite(e):
            continue
        e = int(e)
        per_model_curves[m, :] = _weighted_ratio_average(raw[m, e, :, :][None, ...], trial_counts)[0]

    series_list = []
    for m in range(per_model_curves.shape[0]):
        y = per_model_curves[m]
        if m == 0:
            label = "models"
        else:
            label = "_nolabel_"
        series_list.append(
            Series(
                    kind=PlotKind.LINE,
                    label=label,
                    x=[float(v) for v in x_vals],
                    y=[float(v) for v in y],
                    color=Color.BLUE,
                    y_axis=YAxis.LEFT,
                    marker="o",
                    markersize=3.0,
                    linewidth=.8,
                    alpha=.4
                )
            )

    return OutputSpec(
        figure_id="all_models_s_curve_sig",
        title=sub_cfg.get("title", "Modular Response by Feature Composition - Hidden"),
        x_label="# mod feats",
        y_label="% mod resp",
        x_lim=sub_cfg.get("x_lim", [-0.3, 6.3]),
        y_lim=sub_cfg.get("y_lim", [0.0, 1.0]),
        legend_loc=sub_cfg.get("legend_loc"),
        legend_fontsize=sub_cfg.get("legend_fontsize"),
        figsize=tuple(sub_cfg.get("figsize", (12, 8))),
        dpi=int(sub_cfg.get("dpi", 300)),
        x_ref=[
            RLine(
                val=float(sub_cfg.get("mid_x", 3.0)),
                color=Color.GRAY,
                linestyle=LineStyle.DASHED,
                linewidth=1.5,
                alpha=0.8,
            )
        ],
        y_ref=[
            RLine(
                val=float(sub_cfg.get("mid_y", 0.5)),
                color=Color.GRAY,
                linestyle=LineStyle.DASHED,
                linewidth=1.5,
                alpha=0.8,
            )
        ],
        series_list=series_list
    )


def _weighted_ratio_average(x: np.ndarray, trial_counts: np.ndarray) -> np.ndarray:
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


def _first_sig_epochs(analysis_dir: str, M: int, E: int) -> np.ndarray:
    path = os.path.join(analysis_dir, "sige.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing sige.npz for SCurve sig mode: {path}")

    data = np.load(path, allow_pickle=True)
    if "results" not in data:
        raise ValueError("sige.npz is missing 'results'.")

    sig = np.asarray(data["results"]).astype(bool)
    if sig.shape != (M, E):
        raise ValueError(f"Expected sige results shape {(M, E)}, got {sig.shape}")

    out = np.full((M,), np.nan, dtype=np.float64)
    for m in range(M):
        idx = np.flatnonzero(sig[m])
        if idx.size > 0:
            out[m] = float(idx[0])

    return out


def _mod_count_from_ratio(ratio_label: str) -> int:
    left = str(ratio_label).split(":")[0].strip()
    return int(left)