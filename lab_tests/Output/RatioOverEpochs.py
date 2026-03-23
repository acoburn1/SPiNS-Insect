import os
import numpy as np
from Output.schema.OutputSpec import *
from Statistics.StatHelper import stats_over_models


class RatioOverEpochsOutput:
    name = "RatioOverEpochs"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        path = os.path.join(analysis_dir, "RatioTest.npz")
        data = np.load(path, allow_pickle=True)

        raw = np.asarray(data["raw"], dtype=np.float64)  # (M, E, R, S)

        metadata = None
        if "metadata" in data:
            metadata = data["metadata"].item()

        ratio_labels = None
        set_labels = None

        if metadata is not None:
            ratio_labels = list(metadata.get("ratio_labels", []))
            set_labels = list(metadata.get("set_labels", []))

        if (not ratio_labels) and "ratio_labels" in data:
            ratio_labels = [str(v) for v in data["ratio_labels"].tolist()]

        if (not set_labels) and "set_labels" in data:
            set_labels = [str(v) for v in data["set_labels"].tolist()]

        if not ratio_labels or not set_labels:
            raise ValueError("RatioTest.npz is missing ratio/set labels metadata.")

        ratio_name = str(sub_cfg.get("ratio", "3:3"))
        if ratio_name not in ratio_labels:
            raise ValueError(f"Requested ratio '{ratio_name}' not found. Available: {ratio_labels}")

        r_idx = ratio_labels.index(ratio_name)

        selected = raw[:, :, r_idx, :]
        st = stats_over_models(selected)

        mean = np.asarray(st["mean"], dtype=np.float64)
        ci_lo = np.asarray(st["ci_lo"], dtype=np.float64)
        ci_hi = np.asarray(st["ci_hi"], dtype=np.float64)

        n_epochs = mean.shape[0]
        epochs = np.arange(n_epochs, dtype=np.float64)

        valid_sets = []
        for s_idx, label in enumerate(set_labels):
            vals = mean[:, s_idx]
            if np.any(np.isfinite(vals)):
                valid_sets.append((s_idx, label))

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

        spec = OutputSpec(
            figure_id=sub_cfg.get("name", f"ratio_over_epochs_{ratio_name.replace(':', '_')}"),
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