import os
import numpy as np

from Output.schema.OutputSpec import OutputSpec, Series, PlotKind, Color
from Output.utils import load_mean_ci


class SeriesMFA:
    name = "SeriesMFA"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        path = os.path.join(analysis_dir, "MFA.npz")
        mean, ci_lo, ci_hi = load_mean_ci(path)
        data = np.load(path, allow_pickle=True)

        metadata = data["metadata"].item() if "metadata" in data else {}
        group_labels = list(metadata.get("group_labels", ["mod-core", "mod-per", "lat-core", "lat-per"]))

        n_epochs = int(mean.shape[0])
        epochs = np.arange(n_epochs, dtype=np.float64)

        palette = [Color.BLUE, Color.CYAN, Color.RED, Color.ORANGE]
        series_list = []
        for i, label in enumerate(group_labels):
            series_list.append(
                Series(
                    kind=PlotKind.LINE,
                    label=str(label),
                    x=[float(v) for v in epochs],
                    y=[float(v) for v in mean[:, i]],
                    ci_lower=[float(v) for v in ci_lo[:, i]],
                    ci_upper=[float(v) for v in ci_hi[:, i]],
                    color=palette[i % len(palette)],
                    marker="o",
                    markersize=3.5,
                    linewidth=2.0,
                )
            )

        spec = OutputSpec(
            figure_id=sub_cfg.get("name", "missing_feature_choice"),
            title=sub_cfg.get("title", "Missing-Feature Choice Accuracy"),
            x_label="Epoch",
            y_label="Accuracy",
            series_list=series_list,
            y_lim=[0.0, 1.0],
            x_lim=[0, n_epochs - 1],
            legend_loc="best",
            legend_fontsize=9,
        )

        return [spec]
