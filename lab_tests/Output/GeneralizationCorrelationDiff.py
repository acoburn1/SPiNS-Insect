import numpy as np

from Output.schema.OutputSpec import *
from Output.utils import (
    corr_type_from_cfg,
    first_sig_epochs,
    fit_line_with_stats,
    load_correlation_raw,
    load_ratio_test_bundle,
    points_at_epochs,
    points_for_epoch_list,
    resolve_epoch_range,
    shared_limits,
    weighted_single_ratio,
)


class GeneralizationCorrelationDiffOutput:
    name = "GeneralizationCorrelationDiff"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        ratio_name = "3:3"
        corr_type = corr_type_from_cfg(sub_cfg)
        ratio_bundle = load_ratio_test_bundle(analysis_dir)

        raw_ratio = np.asarray(ratio_bundle["raw"], dtype=np.float64)  # (M,E,R,S)
        raw_corr = load_correlation_raw(analysis_dir, corr_type=corr_type)

        ratio_labels = ratio_bundle["ratio_labels"]
        trial_counts = np.asarray(ratio_bundle["trial_counts"], dtype=np.float64)

        if ratio_name not in ratio_labels:
            raise ValueError(f"Requested ratio '{ratio_name}' not found. Available: {ratio_labels}")

        r_idx = ratio_labels.index(ratio_name)
        pref_over_epochs = weighted_single_ratio(raw_ratio[:, :, r_idx, :], trial_counts[r_idx, :])

        if corr_type == "standard":
            hidden_mod = raw_corr[:, :, 0, 0, 0]
            hidden_lat = raw_corr[:, :, 0, 1, 0]
        else:
            hidden_mod = raw_corr[:, :, 0, 0]
            hidden_lat = raw_corr[:, :, 0, 1]
        hidden_diff = hidden_mod - hidden_lat

        mode = str(sub_cfg.get("epochs", "range")).lower()
        if mode in ("sig", "wb-sig"):
            sig_epochs = first_sig_epochs(analysis_dir, raw_corr.shape[0], raw_corr.shape[1], mode=mode)
            x, y = points_at_epochs(hidden_diff, pref_over_epochs, sig_epochs)
            mode_suffix = "sige" if mode == "sig" else "wb-sige"
            return [_build_spec(sub_cfg, x, y, figure_id=sub_cfg.get("name", f"gen_corrdiff_{mode_suffix}"), suffix=mode)]

        if mode == "range":
            epoch_indices = resolve_epoch_range(sub_cfg, raw_corr.shape[1], default_start=0)
            x_list, y_list = points_for_epoch_list(hidden_diff, pref_over_epochs, epoch_indices)
            x_lim = shared_limits(x_list, padding=0.05)
            specs = []
            for e, x, y in zip(epoch_indices, x_list, y_list):
                specs.append(
                    _build_spec(
                        sub_cfg,
                        x,
                        y,
                        figure_id=f"gen_corrdiff_e{e:03d}",
                        suffix=f"epoch {e}",
                        x_lim=x_lim,
                    )
                )
            return specs

        raise ValueError(f"Unsupported epochs mode: {mode}. Expected 'range', 'sig', or 'wb-sig'.")


def _build_spec(
    sub_cfg: dict,
    x: np.ndarray,
    y: np.ndarray,
    *,
    figure_id: str,
    suffix: str,
    x_lim: list[float] | None = None,
) -> OutputSpec:
    fit_x, fit_y, fit_label = fit_line_with_stats(x, y)

    series = [
        Series(
            kind=PlotKind.SCATTER,
            label="models",
            x=[float(v) for v in x],
            y=[float(v) for v in y],
            color=Color.BLUE,
            marker="o",
            alpha=0.45,
        )
    ]

    if fit_x.size > 0:
        series.append(
            Series(
                kind=PlotKind.LINE,
                label=fit_label,
                x=[float(v) for v in fit_x],
                y=[float(v) for v in fit_y],
                color=Color.RED,
                marker=None,
                linewidth=2.0,
                alpha=0.9,
            )
        )

    return OutputSpec(
        figure_id=figure_id,
        title=f"Modular Preference vs Mod-Lat Hidden Correlation ({suffix})",
        x_label="Mod-Lat Hidden Correlation",
        y_label="Modular Preference",
        x_lim=x_lim,
        y_lim=[0.0, 1.0],
        grid=True,
        legend_loc="best",
        legend_fontsize=sub_cfg.get("legend_fontsize", 8),
        figsize=tuple(sub_cfg.get("figsize", (12, 8))),
        dpi=int(sub_cfg.get("dpi", 300)),
        series_list=series,
    )
