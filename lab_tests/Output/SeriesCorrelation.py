import os
import numpy as np
from Output.schema.OutputSpec import *
from Output.utils import load_mean_ci
import copy

class SeriesCorrelationOutput:
    name = "SeriesCorrelation"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        corr_mean, corr_lo, corr_hi = load_mean_ci(os.path.join(analysis_dir, "Correlation.npz"))
        loss_mean, loss_lo, loss_hi = load_mean_ci(os.path.join(analysis_dir, "Loss.npz"))

        n_epochs = corr_mean.shape[0]
        epochs = np.arange(n_epochs, dtype=np.float64)

        spec_h = OutputSpec(
            figure_id=sub_cfg.get("name", "series_correlation_hidden"),
            title=sub_cfg.get("title", "Means Across Epochs"),
            x_label="Epoch",
            y_label="Correlation Value",
            y2_label="Loss",
            x_lim=[0, n_epochs - 1],
            y_lim=[0.0, 1.0],
            legend_loc="upper right",
            legend_fontsize=10,
            figsize=(12, 8),
            dpi=300,
            series_list=[
                _make_line(
                    label="M Hidden Corrs",
                    x=epochs,
                    y=corr_mean[:, 0, 0, 0],
                    lo=corr_lo[:, 0, 0, 0],
                    hi=corr_hi[:, 0, 0, 0],
                    color=Color.GREEN,
                    y_axis=YAxis.LEFT,
                ),
                _make_line(
                    label="L Hidden Corrs",
                    x=epochs,
                    y=corr_mean[:, 0, 1, 0],
                    lo=corr_lo[:, 0, 1, 0],
                    hi=corr_hi[:, 0, 1, 0],
                    color=Color.PINK,
                    y_axis=YAxis.LEFT,
                ),
                _make_line(
                    label="Losses",
                    x=epochs,
                    y=loss_mean,
                    lo=loss_lo,
                    hi=loss_hi,
                    color=Color.BLACK,
                    y_axis=YAxis.RIGHT,
                ),
            ],
        )

        spec_all = copy.deepcopy(spec_h)
        spec_all.figure_id = "series_correlation"
        spec_all.series_list.extend([
                _make_line(
                    label="M Output Corrs",
                    x=epochs,
                    y=corr_mean[:, 1, 0, 0],
                    lo=corr_lo[:, 1, 0, 0],
                    hi=corr_hi[:, 1, 0, 0],
                    color=Color.BLUE,
                    y_axis=YAxis.LEFT,
                ),
                _make_line(
                    label="L Output Corrs",
                    x=epochs,
                    y=corr_mean[:, 1, 1, 0],
                    lo=corr_lo[:, 1, 1, 0],
                    hi=corr_hi[:, 1, 1, 0],
                    color=Color.RED,
                    y_axis=YAxis.LEFT,
                )
            ]
        )

        return [spec_h, spec_all]


def _make_line(
    *,
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    color: Color,
    y_axis: YAxis,
) -> Series:
    return Series(
        kind=PlotKind.LINE,
        label=label,
        x=[float(v) for v in np.asarray(x, dtype=np.float64)],
        y=[float(v) for v in np.asarray(y, dtype=np.float64).reshape(-1)],
        ci_lower=[float(v) for v in np.asarray(lo, dtype=np.float64).reshape(-1)],
        ci_upper=[float(v) for v in np.asarray(hi, dtype=np.float64).reshape(-1)],
        color=color,
        y_axis=y_axis,
        marker="o",
        markersize=4.0,
        linewidth=2.0,
    )