import os
import numpy as np

from Output.schema.OutputSpec import *
from Output.utils import (
    finite_xy,
    first_sig_epochs,
    fit_line_with_stats,
    load_ratio_test_bundle,
    normalize_k95_raw,
    points_at_epochs,
    resolve_epoch_range,
    shared_limits,
    weighted_single_ratio,
)


class K95DiffGeneralizationOutput:
    name = "K95DiffGeneralization"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        ratio_name = "3:3"

        ratio_bundle = load_ratio_test_bundle(analysis_dir)
        k95_np = np.load(os.path.join(analysis_dir, "K95.npz"))

        raw_ratio = np.asarray(ratio_bundle["raw"], dtype=np.float64)  # (M,E,R,S)
        raw_k95 = normalize_k95_raw(np.asarray(k95_np["raw"], dtype=np.float64))

        ratio_labels = ratio_bundle["ratio_labels"]
        trial_counts = np.asarray(ratio_bundle["trial_counts"], dtype=np.float64)

        if ratio_name not in ratio_labels:
            raise ValueError(f"Requested ratio '{ratio_name}' not found. Available: {ratio_labels}")

        r_idx = ratio_labels.index(ratio_name)
        pref = weighted_single_ratio(raw_ratio[:, :, r_idx, :], trial_counts[r_idx, :])
        k95_diff = raw_k95[:, :, 0] - raw_k95[:, :, 1]

        mode = str(sub_cfg.get("epochs", "sig")).lower()
        if mode == "sig":
            sig_epochs = first_sig_epochs(analysis_dir, pref.shape[0], pref.shape[1])
            x, y = points_at_epochs(k95_diff, pref, sig_epochs)
            y_lim = [0.0, 1.0]
            x_lim = sub_cfg.get("x_bound", shared_limits([x]))
            return [
                _build_spec(
                    sub_cfg,
                    x,
                    y,
                    figure_id=sub_cfg.get("name", "k95diff_generalization_sig"),
                    suffix="sig",
                    x_lim=x_lim,
                    y_lim=y_lim,
                )
            ]

        if mode == "range":
            epoch_indices = resolve_epoch_range(sub_cfg, pref.shape[1], default_start=0)
            per_epoch = []
            for e in epoch_indices:
                x, y = finite_xy(k95_diff[:, e], pref[:, e])
                per_epoch.append((x, y))

            x_lim = [-3.5, 3.5] #shared_limits([p[0] for p in per_epoch])
            y_lim = [0.0, 1.0]

            specs = []
            for e, (x, y) in zip(epoch_indices, per_epoch):
                specs.append(
                    _build_spec(
                        sub_cfg,
                        x,
                        y,
                        figure_id=f"k95diff_generalization_e{e:03d}",
                        suffix=f"epoch {e}",
                        x_lim=x_lim,
                        y_lim=y_lim,
                    )
                )
            return specs

        raise ValueError(f"Unsupported epochs mode: {mode}. Expected 'range' or 'sig'.")


def _build_spec(
    sub_cfg: dict,
    x: np.ndarray,
    y: np.ndarray,
    *,
    figure_id: str,
    suffix: str,
    x_lim: list[float] | None,
    y_lim: list[float],
) -> OutputSpec:
    fit_x, fit_y, fit_label = fit_line_with_stats(x, y)
    series_list = [
        Series(
            kind=PlotKind.SCATTER,
            label="models",
            x=[float(v) for v in x],
            y=[float(v) for v in y],
            color=Color.BLUE,
            marker="o",
            alpha=0.5,
        )
    ]

    if fit_x.size > 0:
        series_list.append(
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
        title=f"Modular Preference vs Mod-Lat K95 ({suffix})",
        x_label="Mod-Lat K95",
        y_label="Modular Preference",
        x_lim=x_lim,
        y_lim=y_lim,
        grid=True,
        legend_loc="best",
        legend_fontsize=sub_cfg.get("legend_fontsize", 8),
        figsize=tuple(sub_cfg.get("figsize", (12, 8))),
        dpi=int(sub_cfg.get("dpi", 300)),
        series_list=series_list,
    )
