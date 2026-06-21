import numpy as np

from Output.schema.OutputSpec import OutputSpec
from Output.utils import epoch_k95_summaries, get_hyperparameter_runs_with_data, sample_epochs


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

            epoch_samples = sample_epochs(np.asarray(lr_runs[0]["K95"]["raw"], dtype=np.float64), step=step)

            hls_values = [run["hls"] for run in lr_runs]
            mean_avg_mat = np.full((len(epoch_samples), len(lr_runs)), np.nan, dtype=np.float64)
            diff_mat = np.full((len(epoch_samples), len(lr_runs)), np.nan, dtype=np.float64)

            for j, run in enumerate(lr_runs):
                raw = np.asarray(run["K95"]["raw"], dtype=np.float64)
                per_epoch_mean_avg, per_epoch_diff = epoch_k95_summaries(raw)

                mean_avg_mat[:, j] = per_epoch_mean_avg[epoch_samples]
                diff_mat[:, j] = per_epoch_diff[epoch_samples]

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
                    y_ticks=[float(i) for i in range(len(epoch_samples))],
                    y_ticklabels=[str(e) for e in epoch_samples],
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
                    y_ticks=[float(i) for i in range(len(epoch_samples))],
                    y_ticklabels=[str(e) for e in epoch_samples],
                    matrix_split=0.0,
                )
            )

        return specs

