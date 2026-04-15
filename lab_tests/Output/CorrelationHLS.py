import numpy as np
from Output.schema.OutputSpec import *
from Statistics.StatHelper import stats_over_models
from Output.utils import first_sig_epochs, get_hyperparameter_runs_with_data, spread_x


class CorrelationHLSOutput:
    name = "Correlation-HLS"
    hyperd = True

    def generate_output(self, sub_cfg: dict, analysis_root: str) -> list[OutputSpec]:
        sig_mode = str(sub_cfg.get("sige_type", "sig")).lower()
        if sig_mode not in ("sig", "wb-sig"):
            raise ValueError(f"Unsupported sige_type: {sig_mode}. Expected 'sig' or 'wb-sig'.")

        sig_name = "sige" if sig_mode == "sig" else "wb-sige"
        runs = get_hyperparameter_runs_with_data(analysis_root, ["Correlation", sig_name])
        runs = [run for run in runs if "Correlation" in run and sig_name in run]

        if not runs:
            raise FileNotFoundError(f"No Correlation.npz + {sig_name}.npz runs found under {analysis_root}")

        source_name = str(sub_cfg.get("source", "hidden")).lower()
        source_idx = 0 if source_name in ("hidden", "hid") else 1

        by_lr = {}
        for run in runs:
            by_lr.setdefault(run["lr"], []).append(run)

        specs = []
        for lr in sorted(by_lr.keys()):
            lr_runs = sorted(by_lr[lr], key=lambda d: d["hls"])

            scatter_x = []
            scatter_y = []

            bar_x = []
            bar_y = []
            bar_yerr = []

            x_ticks = []
            x_labels = []

            for run in lr_runs:
                hls = run["hls"]
                corr_raw = np.asarray(run["Correlation"]["raw"], dtype=np.float64)
                n_models, n_epochs = corr_raw.shape[0], corr_raw.shape[1]
                sig_epochs = first_sig_epochs(run["analysis_dir"], n_models, n_epochs, mode=sig_mode)

                per_model_scores = _scores_at_first_sig_epoch(
                    corr_raw=corr_raw,
                    sig_epochs=sig_epochs,
                    source_idx=source_idx,
                )

                good = np.isfinite(per_model_scores)
                vals = per_model_scores[good]

                x_ticks.append(float(hls))
                x_labels.append(str(hls))

                if vals.size > 0:
                    xs = spread_x(float(hls), vals.size, width=0.18)
                    scatter_x.extend(xs.tolist())
                    scatter_y.extend(vals.tolist())

                    st = stats_over_models(vals[:, None], ci=0.95)
                    mean = float(st["mean"][0])
                    ci_lo = float(st["ci_lo"][0])
                    ci_hi = float(st["ci_hi"][0])

                    bar_x.append(float(hls))
                    bar_y.append(mean)
                    bar_yerr.append(max(mean - ci_lo, ci_hi - mean))

            src_label = "Hidden" if source_idx == 0 else "Output"
            mode_label = "sig" if sig_mode == "sig" else "wb-sig"
            title_base = f"{src_label} Correlation at First Significant Epoch vs HLS ({mode_label}, LR={lr:g})"

            specs.append(
                OutputSpec(
                    figure_id=f"correlation_hls_scatter_lr{str(lr).replace('.', 'p')}",
                    title=title_base + " — Scatter",
                    x_label="Hidden Layer Size",
                    y_label="Correlation at First Significant Epoch",
                    x_ticks=x_ticks,
                    x_ticklabels=x_labels,
                    x_lim=[min(x_ticks) - 0.6, max(x_ticks) + 0.6],
                    y_lim=[0.0, 1.0],
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
                            x=[float(v) for v in scatter_x],
                            y=[float(v) for v in scatter_y],
                            color=Color.BLUE,
                            marker="o",
                            alpha=0.5,
                        )
                    ],
                )
            )

            specs.append(
                OutputSpec(
                    figure_id=f"correlation_hls_bar_lr{str(lr).replace('.', 'p')}",
                    title=title_base + " — Mean ± 95% CI",
                    x_label="Hidden Layer Size",
                    y_label="Correlation at First Significant Epoch",
                    x_ticks=x_ticks,
                    x_ticklabels=x_labels,
                    x_lim=[min(x_ticks) - 0.6, max(x_ticks) + 0.6],
                    y_lim=[0.0, 1.0],
                    grid=True,
                    legend_loc="best",
                    legend_ncol=1,
                    legend_fontsize=8,
                    figsize=(12, 8),
                    dpi=300,
                    series_list=[
                        Series(
                            kind=PlotKind.BAR,
                            label="Mean",
                            x=[float(v) for v in bar_x],
                            y=[float(v) for v in bar_y],
                            yerr=[float(v) for v in bar_yerr],
                            color=Color.BLUE,
                            alpha=0.8,
                        )
                    ],
                )
            )

        return specs


def _scores_at_first_sig_epoch(corr_raw: np.ndarray, sig_epochs: np.ndarray, source_idx: int) -> np.ndarray:
    x = np.asarray(corr_raw, dtype=np.float64)
    epochs = np.asarray(sig_epochs, dtype=np.float64)

    if x.ndim != 5 or x.shape[2:] != (2, 2, 2):
        raise ValueError(f"Expected Correlation raw shape (M, E, 2, 2, 2), got {x.shape}")

    if epochs.ndim != 1 or epochs.shape[0] != x.shape[0]:
        raise ValueError(f"Expected first significant epochs shape (M,), got {epochs.shape}")

    M = x.shape[0]
    out = np.full((M,), np.nan, dtype=np.float64)

    for m in range(M):
        e0 = epochs[m]
        if not np.isfinite(e0):
            continue

        e0 = int(e0)
        mod_r = x[m, e0, source_idx, 0, 0]
        lat_r = x[m, e0, source_idx, 1, 0]

        if np.isfinite(mod_r) and np.isfinite(lat_r):
            out[m] = 0.5 * (mod_r + lat_r)

    return out
