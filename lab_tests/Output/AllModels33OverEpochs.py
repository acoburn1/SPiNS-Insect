import os
import numpy as np

from Output.schema.OutputSpec import *


class AllModels33OverEpochsOutput:
    name = "AllModels33OverEpochs"
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

        ratio_name = str(sub_cfg.get("ratio", "3:3"))
        if ratio_name not in ratio_labels:
            raise ValueError(f"Requested ratio '{ratio_name}' not found. Available: {ratio_labels}")

        r_idx = ratio_labels.index(ratio_name)

        model_curves = _weighted_single_ratio(raw[:, :, r_idx, :], trial_counts[r_idx, :])

        n_models, n_epochs = model_curves.shape
        epochs = np.arange(n_epochs, dtype=np.float64)

        series_list = []

        for m in range(n_models):
            y = _smooth_1d(model_curves[m], window=5)
            if not np.any(np.isfinite(y)):
                continue

            if m == 0:
                label = "models"
            else:
                label = "_nolabel_"

            series_list.append(
                Series(
                    kind=PlotKind.LINE,
                    label=label,
                    x=[float(v) for v in epochs],
                    y=[float(v) for v in y],
                    color=Color.BLUE,
                    y_axis=YAxis.LEFT,
                    linewidth=1,
                    alpha=.4,
                    marker=None,
                )
            )

        r_lines = [
                RLine(
                    val=float(sub_cfg.get("mid_y", 0.5)),
                    color=Color.GRAY,
                    linestyle=LineStyle.DASHED,
                    linewidth=1.5,
                    alpha=0.8,
                )]

        return [
            OutputSpec(
                figure_id=sub_cfg.get("name", "all_models_33_over_epochs"),
                title=sub_cfg.get("title", "All Models 3:3 Modular Response Across Epochs - Hidden"),
                x_label="Epoch",
                y_label="% Modular Response",
                y_lim=[0.0, 1.0],
                x_lim=[0.0, float(n_epochs - 1)],
                legend_loc=None,
                legend_fontsize=sub_cfg.get("legend_fontsize"),
                figsize=(12, 8),
                dpi=300,
                y_ref=r_lines,
                series_list=series_list,
            )
        ]


def _weighted_single_ratio(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
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

def _smooth_1d(y: np.ndarray, window: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        raise ValueError("window must be odd")

    half = window // 2
    out = np.full_like(y, np.nan, dtype=np.float64)

    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        out[i] = np.nanmean(y[lo:hi])

    return out