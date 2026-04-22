import numpy as np
from Output.schema.OutputSpec import *
from Statistics.StatHelper import stats_over_models
from Output.utils import (
    load_ratio_test_bundle,
    resolve_requested_set_indices,
    selected_sets_suffix,
    valid_set_indices_for_ratio,
)


class RatioOverEpochsOutput:
    name = "RatioOverEpochs"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        bundle = load_ratio_test_bundle(analysis_dir)
        raw = bundle["raw"]
        ratio_labels = bundle["ratio_labels"]
        set_labels = bundle["set_labels"]

        if not ratio_labels or not set_labels:
            raise ValueError("RatioTest.npz is missing ratio/set labels metadata.")

        ratio_name = str(sub_cfg.get("ratio", "3:3"))
        if ratio_name not in ratio_labels:
            raise ValueError(f"Requested ratio '{ratio_name}' not found. Available: {ratio_labels}")

        r_idx = ratio_labels.index(ratio_name)
        avg_sets = bool(sub_cfg.get("avg", False))
        requested_set_indices = resolve_requested_set_indices(sub_cfg, set_labels)

        valid_indices = set(valid_set_indices_for_ratio(raw, r_idx))
        valid_sets = [(s_idx, set_labels[s_idx]) for s_idx in requested_set_indices if s_idx in valid_indices]
        if not valid_sets:
            requested_labels = [set_labels[s_idx] for s_idx in requested_set_indices]
            raise ValueError(
                f"No valid set data found for ratio '{ratio_name}' from requested sets {requested_labels}."
            )

        n_epochs = raw.shape[1]
        epochs = np.arange(n_epochs, dtype=np.float64)

        colors = [
            Color.BLUE,
            Color.RED,
            Color.GREEN,
            Color.ORANGE,
            Color.PURPLE,
            Color.BROWN,
            Color.PINK,
            Color.CYAN,
            Color.GRAY,
            Color.OLIVE,
        ]

        if avg_sets:
            valid_indices_list = [s_idx for s_idx, _ in valid_sets]
            per_model_epoch = np.nanmean(raw[:, :, r_idx, valid_indices_list], axis=2)
            st = stats_over_models(per_model_epoch)
            mean = np.asarray(st["mean"], dtype=np.float64)
            ci_lo = np.asarray(st["ci_lo"], dtype=np.float64)
            ci_hi = np.asarray(st["ci_hi"], dtype=np.float64)
            series_list = [
                Series(
                    kind=PlotKind.LINE,
                    label=sub_cfg.get("line_label", "avg sets"),
                    x=[float(v) for v in epochs],
                    y=[float(v) for v in mean],
                    ci_lower=[float(v) for v in ci_lo],
                    ci_upper=[float(v) for v in ci_hi],
                    color=Color.BLUE,
                    y_axis=YAxis.LEFT,
                    marker="o",
                    markersize=4.0,
                    linewidth=2.0,
                )
            ]
        else:
            selected = raw[:, :, r_idx, :]
            st = stats_over_models(selected)
            mean = np.asarray(st["mean"], dtype=np.float64)
            ci_lo = np.asarray(st["ci_lo"], dtype=np.float64)
            ci_hi = np.asarray(st["ci_hi"], dtype=np.float64)
            series_list = []
            for i, (s_idx, label) in enumerate(valid_sets):
                c = colors[i % len(colors)]
                series_list.append(
                    Series(
                        kind=PlotKind.LINE,
                        label=label,
                        x=[float(v) for v in epochs],
                        y=[float(v) for v in np.asarray(mean[:, s_idx], dtype=np.float64)],
                        ci_lower=[float(v) for v in np.asarray(ci_lo[:, s_idx], dtype=np.float64)],
                        ci_upper=[float(v) for v in np.asarray(ci_hi[:, s_idx], dtype=np.float64)],
                        color=c,
                        y_axis=YAxis.LEFT,
                        marker="o",
                        markersize=4.0,
                        linewidth=2.0,
                    )
                )

        selected_labels = [label for _, label in valid_sets]
        set_suffix = selected_sets_suffix(selected_labels, set_labels)
        avg_suffix = "_avg" if avg_sets else ""
        spec = OutputSpec(
            figure_id=sub_cfg.get("name", f"ratio_over_epochs_{ratio_name.replace(':', '_')}{avg_suffix}{set_suffix}"),
            title=sub_cfg.get("title", f"{ratio_name} Ratio Modular Response Across Epochs - Hidden"),
            x_label="Epoch",
            y_label="% Modular Response",
            x_lim=[0, n_epochs-1],
            y_lim=[0.0, 1.0],
            legend_loc=sub_cfg.get("legend_loc", "upper right"),
            legend_fontsize=sub_cfg.get("legend_fontsize", 10),
            figsize=tuple(sub_cfg.get("figsize", (12, 8))),
            dpi=int(sub_cfg.get("dpi", 300)),
            y_ref=[
                RLine(
                val=0.5,
                color=Color.GRAY,
                linestyle=LineStyle.DASHED,
                linewidth=1.5,
                alpha=0.8,
                )
            ],
            series_list=series_list,
        )

        return [spec]
