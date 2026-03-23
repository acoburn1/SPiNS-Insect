import numpy as np

from Output.schema.OutputSpec import OutputSpec
from Output.utils import get_hyperparameter_runs_with_data


class EpochsHLSwK95HeatmapOutput:
    name = "Epochs-HLSwK95Heatmap"
    hyperd = True

    def generate_output(self, sub_cfg: dict, analysis_root: str) -> list[OutputSpec]:
        runs = get_hyperparameter_runs_with_data(analysis_root, ["K95"])
        runs = [run for run in runs if "K95" in run]

        if not runs:
            raise FileNotFoundError(f"No K95.npz files found under {analysis_root}")

        by_lr = {}
        for run in runs:
            by_lr.setdefault(run["lr"], []).append(run)

        step = 3

        specs = []
        for lr in sorted(by_lr.keys()):
            lr_runs = sorted(by_lr[lr], key=lambda d: d["hls"])
            if not lr_runs:
                continue

            sample_epochs = _sample_epochs(np.asarray(lr_runs[0]["K95"]["raw"], dtype=np.float64), step=step)

            hls_values = [run["hls"] for run in lr_runs]
            mean_avg_mat = np.full((len(sample_epochs), len(lr_runs)), np.nan, dtype=np.float64)
            diff_mat = np.full((len(sample_epochs), len(lr_runs)), np.nan, dtype=np.float64)

            for j, run in enumerate(lr_runs):
                raw = np.asarray(run["K95"]["raw"], dtype=np.float64)
                per_epoch_mean_avg, per_epoch_diff = _epoch_k95_summaries(raw)

                mean_avg_mat[:, j] = per_epoch_mean_avg[sample_epochs]
                diff_mat[:, j] = per_epoch_diff[sample_epochs]

            lr_str = f"{lr:g}"

            specs.append(
                OutputSpec(
                    figure_id=f"epochs_hls_k95_mean_lr{str(lr).replace('.', 'p')}_meanavg",
                    title=f"Mean K95 Across Categories by Epoch and HLS (LR={lr_str})",
                    x_label="Hidden Layer Size",
                    y_label="Epoch",
                    matrix=mean_avg_mat.tolist(),
                    grid=False,
                    figsize=(10, 8),
                    dpi=300,
                    x_ticks=[float(i) for i in range(len(hls_values))],
                    x_ticklabels=[str(h) for h in hls_values],
                    y_ticks=[float(i) for i in range(len(sample_epochs))],
                    y_ticklabels=[str(e) for e in sample_epochs],
                )
            )

            specs.append(
                OutputSpec(
                    figure_id=f"epochs_hls_k95_diff_lr{str(lr).replace('.', 'p')}_mod_minus_lat",
                    title=f"K95 Mod - Lat by Epoch and HLS (LR={lr_str})",
                    x_label="Hidden Layer Size",
                    y_label="Epoch",
                    matrix=diff_mat.tolist(),
                    grid=False,
                    figsize=(10, 8),
                    dpi=300,
                    x_ticks=[float(i) for i in range(len(hls_values))],
                    x_ticklabels=[str(h) for h in hls_values],
                    y_ticks=[float(i) for i in range(len(sample_epochs))],
                    y_ticklabels=[str(e) for e in sample_epochs],
                    matrix_split=0.0,
                )
            )

        return specs


def _epoch_k95_summaries(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    mod = x[:, :, 0]
    lat = x[:, :, 1]

    mean_avg = np.nanmean((mod + lat) / 2.0, axis=0)
    diff = np.nanmean(mod - lat, axis=0)

    return mean_avg, diff


def _sample_epochs(raw: np.ndarray, step: int = 3) -> np.ndarray:
    x = np.asarray(raw)
    if x.ndim < 2:
        raise ValueError(f"Expected raw with epoch axis, got shape {x.shape}")

    n_epochs = int(x.shape[1])
    last = n_epochs // 2
    return np.arange(0, last + 1, step, dtype=np.int64)