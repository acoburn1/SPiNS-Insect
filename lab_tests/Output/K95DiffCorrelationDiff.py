import os
import numpy as np

from Output.schema.OutputSpec import *
from Output.utils import finite_xy, fit_line_with_stats, normalize_k95_raw, resolve_epoch_range, shared_limits


class K95DiffCorrelationDiffOutput:
    name = "K95DiffCorrelationDiff"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        corr_np = np.load(os.path.join(analysis_dir, "Correlation.npz"))
        k95_np = np.load(os.path.join(analysis_dir, "K95.npz"))

        raw_corr = np.asarray(corr_np["raw"], dtype=np.float64)
        raw_k95 = normalize_k95_raw(np.asarray(k95_np["raw"], dtype=np.float64))

        if raw_corr.ndim != 5 or raw_corr.shape[2:] != (2, 2, 2):
            raise ValueError(f"Expected Correlation raw shape (M,E,2,2,2), got {raw_corr.shape}")

        corr_diff = raw_corr[:, :, 0, 0, 0] - raw_corr[:, :, 0, 1, 0]
        k95_diff = raw_k95[:, :, 0] - raw_k95[:, :, 1]

        epoch_indices = resolve_epoch_range(sub_cfg, raw_corr.shape[1], default_start=0)
        pts = []
        for e in epoch_indices:
            x, y = finite_xy(corr_diff[:, e], k95_diff[:, e])
            pts.append((x, y))

        x_lim = shared_limits([p[0] for p in pts])
        y_lim = shared_limits([p[1] for p in pts])

        specs = []
        for e, (x, y) in zip(epoch_indices, pts):
            fit_x, fit_y, fit_label = fit_line_with_stats(x, y)
            series_list = [
                Series(
                    kind=PlotKind.SCATTER,
                    label="models",
                    x=[float(v) for v in x],
                    y=[float(v) for v in y],
                    color=Color.PURPLE,
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
                        color=Color.PURPLE,
                        marker=None,
                        linewidth=2.0,
                        alpha=0.9,
                    )
                )

            specs.append(
                OutputSpec(
                    figure_id=f"k95diff_corrdiff_e{e:03d}",
                    title=f"Mod-Lat K95 vs Mod-Lat Hidden Correlation (epoch {e})",
                    x_label="Mod-Lat Hidden Correlation",
                    y_label="Mod-Lat K95",
                    x_lim=x_lim,
                    y_lim=y_lim,
                    grid=True,
                    legend_loc="best",
                    legend_fontsize=sub_cfg.get("legend_fontsize", 8),
                    figsize=tuple(sub_cfg.get("figsize", (12, 8))),
                    dpi=int(sub_cfg.get("dpi", 300)),
                    series_list=series_list,
                )
            )

        return specs
