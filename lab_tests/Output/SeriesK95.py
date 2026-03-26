import os
import numpy as np

from Output.schema.OutputSpec import *
from Statistics.StatHelper import stats_over_models
from Output.utils import normalize_k95_raw


class SeriesK95Output:
    name = "SeriesK95"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        k95_np = np.load(os.path.join(analysis_dir, "K95.npz"))
        raw_k95 = normalize_k95_raw(np.asarray(k95_np["raw"], dtype=np.float64))  # (M,E,2)

        st = stats_over_models(raw_k95)

        mean = np.asarray(st["mean"], dtype=np.float64)  # (E,2)
        ci_lo = np.asarray(st["ci_lo"], dtype=np.float64)
        ci_hi = np.asarray(st["ci_hi"], dtype=np.float64)

        n_epochs = mean.shape[0]
        epochs = np.arange(n_epochs, dtype=np.float64)

        y_vals = np.concatenate([mean[:, 0], mean[:, 1]])
        y_vals = y_vals[np.isfinite(y_vals)]
        y_lim = [0.0, 1.0] if y_vals.size == 0 else [max(0.0, float(np.nanmin(y_vals)) - 0.5), float(np.nanmax(y_vals)) + 0.5]

        return [
            OutputSpec(
                figure_id=sub_cfg.get("name", "series_k95"),
                title=sub_cfg.get("title", "K95 Across Epochs"),
                x_label="Epoch",
                y_label="K95",
                x_lim=[0.0, float(n_epochs - 1)],
                y_lim=y_lim,
                legend_loc=sub_cfg.get("legend_loc", "best"),
                legend_fontsize=sub_cfg.get("legend_fontsize", 9),
                figsize=tuple(sub_cfg.get("figsize", (12, 8))),
                dpi=int(sub_cfg.get("dpi", 300)),
                series_list=[
                    _make_line(
                        label="mod",
                        x=epochs,
                        y=mean[:, 0],
                        lo=ci_lo[:, 0],
                        hi=ci_hi[:, 0],
                        color=Color.BLUE,
                    ),
                    _make_line(
                        label="lat",
                        x=epochs,
                        y=mean[:, 1],
                        lo=ci_lo[:, 1],
                        hi=ci_hi[:, 1],
                        color=Color.RED,
                    ),
                ],
            )
        ]


def _make_line(*, label: str, x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray, color: Color) -> Series:
    return Series(
        kind=PlotKind.LINE,
        label=label,
        x=[float(v) for v in np.asarray(x, dtype=np.float64)],
        y=[float(v) for v in np.asarray(y, dtype=np.float64)],
        ci_lower=[float(v) for v in np.asarray(lo, dtype=np.float64)],
        ci_upper=[float(v) for v in np.asarray(hi, dtype=np.float64)],
        color=color,
        marker="o",
        markersize=4.0,
        linewidth=2.0,
    )
