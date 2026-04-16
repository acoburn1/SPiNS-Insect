import os
import copy
import numpy as np
from Output.schema.OutputSpec import *
from Output.utils import corr_type_from_cfg, load_correlation_raw, load_mean_ci
from Statistics.StatHelper import stats_over_models


class WithinVsBetweenCorrelationOutput:
    name = "WithinVsBetweenCorrelation"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        corr_type = corr_type_from_cfg(sub_cfg, default="wb")
        raw_corr = load_correlation_raw(analysis_dir, corr_type=corr_type)
        if corr_type == "standard":
            raw = np.asarray(raw_corr[:, :, :, :, 0], dtype=np.float64)
        else:
            raw = np.asarray(raw_corr, dtype=np.float64)

        mode = str(sub_cfg.get("epochs", "range")).lower()
        if mode == "wb-sig":
            sig_path = os.path.join(analysis_dir, "wb-sige.npz")
            sig_np = np.load(sig_path, allow_pickle=True)
            if "results" not in sig_np:
                raise ValueError("wb-sige.npz is missing 'results'.")
            sig_mask = np.asarray(sig_np["results"], dtype=bool)
            if sig_mask.shape != raw.shape[:2]:
                raise ValueError(f"Expected wb-sige shape {raw.shape[:2]}, got {sig_mask.shape}")
            raw = np.where(sig_mask[:, :, None, None], raw, np.nan)
        elif mode != "range":
            raise ValueError(f"Unsupported epochs mode: {mode}. Expected 'range' or 'wb-sig'.")

        loss_mean, loss_lo, loss_hi = load_mean_ci(os.path.join(analysis_dir, "Loss.npz"))
        st = stats_over_models(raw)
        mean = st["mean"]
        lo = st["ci_lo"]
        hi = st["ci_hi"]

        n_epochs = mean.shape[0]
        epochs = np.arange(n_epochs, dtype=np.float64)

        spec_h = OutputSpec(
            figure_id=sub_cfg.get("name", "within_vs_between_hidden"),
            title=sub_cfg.get("title", "Within vs Between Across Epochs"),
            x_label="Epoch",
            y_label="Within - Between",
            y2_label="Loss",
            x_lim=[0, n_epochs - 1],
            y_lim=[0,1],
            legend_loc="upper right",
            legend_fontsize=10,
            figsize=(12, 8),
            dpi=300,
            series_list=[
                _make_line(
                    label="M Hidden Structure",
                    x=epochs,
                    y=mean[:, 0, 0],
                    lo=lo[:, 0, 0],
                    hi=hi[:, 0, 0],
                    color=Color.GREEN,
                    y_axis=YAxis.LEFT,
                ),
                _make_line(
                    label="L Hidden Structure",
                    x=epochs,
                    y=mean[:, 0, 1],
                    lo=lo[:, 0, 1],
                    hi=hi[:, 0, 1],
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
        spec_all.figure_id = "within_vs_between"
        spec_all.series_list.extend([
            _make_line(
                label="M Output Structure",
                x=epochs,
                y=mean[:, 1, 0],
                lo=lo[:, 1, 0],
                hi=hi[:, 1, 0],
                color=Color.BLUE,
                y_axis=YAxis.LEFT,
            ),
            _make_line(
                label="L Output Structure",
                x=epochs,
                y=mean[:, 1, 1],
                lo=lo[:, 1, 1],
                hi=hi[:, 1, 1],
                color=Color.RED,
                y_axis=YAxis.LEFT,
            ),
        ])

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
