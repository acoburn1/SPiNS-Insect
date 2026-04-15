import numpy as np
from Output.schema.OutputSpec import *
from Output.utils import first_sig_epochs, get_hyperparameter_runs_with_data, spread_x


class SigEHLSOutput:
    name = "SigE-HLS"
    hyperd = True

    def generate_output(self, sub_cfg: dict, analysis_root: str) -> list[OutputSpec]:
        sig_mode = str(sub_cfg.get("sige_type", "sig")).lower()
        if sig_mode not in ("sig", "wb-sig"):
            raise ValueError(f"Unsupported sige_type: {sig_mode}. Expected 'sig' or 'wb-sig'.")

        sig_name = "sige" if sig_mode == "sig" else "wb-sige"
        runs = get_hyperparameter_runs_with_data(analysis_root, [sig_name])
        runs = [run for run in runs if sig_name in run]

        if not runs:
            raise FileNotFoundError(f"No {sig_name}.npz files found under {analysis_root}")

        by_lr = {}
        for run in runs:
            by_lr.setdefault(run["lr"], []).append(run)

        specs = []
        for lr in sorted(by_lr.keys()):
            lr_runs = sorted(by_lr[lr], key=lambda d: d["hls"])

            xs = []
            ys = []
            hls_values = []

            for run in lr_runs:
                hls = run["hls"]
                sig_mask = np.asarray(run[sig_name]["results"], dtype=bool)

                if sig_mask.ndim != 2:
                    raise ValueError(f"Expected sige results shape (M, E), got {sig_mask.shape}")

                first_sig = first_sig_epochs(run["analysis_dir"], sig_mask.shape[0], sig_mask.shape[1], mode=sig_mode)

                good = np.isfinite(first_sig)
                vals = first_sig[good]
                if vals.size == 0:
                    continue

                pts_x = spread_x(float(hls), vals.size, width=0.18)

                xs.extend(pts_x.tolist())
                ys.extend(vals.tolist())
                hls_values.append(hls)

            if not hls_values:
                continue

            specs.append(
                OutputSpec(
                    figure_id=f"sige_hls_lr{str(lr).replace('.', 'p')}",
                    title=f"First Significant Epoch vs Hidden Layer Size ({sig_mode}, LR={lr:g})",
                    x_label="Hidden Layer Size",
                    y_label="First Significant Epoch",
                    x_ticks=[float(h) for h in hls_values],
                    x_ticklabels=[str(h) for h in hls_values],
                    x_lim=[min(hls_values) - 0.6, max(hls_values) + 0.6],
                    y_lim=[0.0, 120.0],
                    grid=True,
                    legend_loc="best",
                    legend_ncol=1,
                    legend_fontsize=8,
                    figsize=(12, 8),
                    dpi=300,
                    series_list=[
                        Series(
                            kind=PlotKind.SCATTER,
                            label="Models",
                            x=[float(v) for v in xs],
                            y=[float(v) for v in ys],
                            color=Color.BLUE,
                            marker="o",
                            alpha=0.5,
                        )
                    ],
                )
            )

        return specs
