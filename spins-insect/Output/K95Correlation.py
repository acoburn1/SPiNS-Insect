import os
import numpy as np

from Output.schema.OutputSpec import *
from Output.utils import (
    corr_type_from_cfg,
    finite_xy,
    first_sig_epochs,
    fit_line_with_stats,
    load_correlation_raw,
    normalize_k95_raw,
    points_at_epochs,
    resolve_epoch_range,
    shared_limits,
)


class K95CorrelationOutput:
    name = "K95Correlation"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        corr_type = corr_type_from_cfg(sub_cfg)
        raw_corr = load_correlation_raw(analysis_dir, corr_type=corr_type)
        k95_np = np.load(os.path.join(analysis_dir, "K95.npz"))

        raw_k95 = normalize_k95_raw(np.asarray(k95_np["raw"], dtype=np.float64))
        if corr_type == "standard":
            h_mod = raw_corr[:, :, 0, 0, 0]
            h_lat = raw_corr[:, :, 0, 1, 0]
        else:
            h_mod = raw_corr[:, :, 0, 0]
            h_lat = raw_corr[:, :, 0, 1]
        corr_token = "standard" if corr_type == "standard" else "wb"
        k_mod = raw_k95[:, :, 0]
        k_lat = raw_k95[:, :, 1]

        mode = str(sub_cfg.get("epochs", "sig")).lower()
        if mode in ("sig", "wb-sig"):
            sig_epochs = first_sig_epochs(analysis_dir, raw_corr.shape[0], raw_corr.shape[1], mode=mode)
            mod_x, mod_y = points_at_epochs(h_mod, k_mod, sig_epochs)
            lat_x, lat_y = points_at_epochs(h_lat, k_lat, sig_epochs)
            x_lim = shared_limits([mod_x, lat_x], fallback=[0.0, 1.0], clamp_01=True)
            y_lim = shared_limits([mod_y, lat_y], fallback=[0.0, 1.0], clamp_01=False)
            mode_suffix = "sige" if mode == "sig" else "wb-sige"
            return [
                _build_spec(
                    sub_cfg,
                    mod_x,
                    mod_y,
                    lat_x,
                    lat_y,
                    figure_id=sub_cfg.get("name", f"k95_correlation_{corr_token}_{mode_suffix}"),
                    suffix=mode,
                    x_lim=x_lim,
                    y_lim=y_lim,
                )
            ]

        if mode == "range":
            epoch_indices = resolve_epoch_range(sub_cfg, raw_corr.shape[1], default_start=0)
            per_epoch = []
            for e in epoch_indices:
                mod_x, mod_y = finite_xy(h_mod[:, e], k_mod[:, e])
                lat_x, lat_y = finite_xy(h_lat[:, e], k_lat[:, e])
                per_epoch.append((mod_x, mod_y, lat_x, lat_y))

            x_lim = shared_limits([v for p in per_epoch for v in (p[0], p[2])], fallback=[0.0, 1.0], clamp_01=True)
            y_lim = shared_limits([v for p in per_epoch for v in (p[1], p[3])], fallback=[0.0, 1.0], clamp_01=False)

            specs = []
            for e, (mod_x, mod_y, lat_x, lat_y) in zip(epoch_indices, per_epoch):
                specs.append(
                    _build_spec(
                        sub_cfg,
                        mod_x,
                        mod_y,
                        lat_x,
                        lat_y,
                        figure_id=f"k95_correlation_{corr_token}_e{e:03d}",
                        suffix=f"epoch {e}",
                        x_lim=x_lim,
                        y_lim=y_lim,
                    )
                )
            return specs

        raise ValueError(f"Unsupported epochs mode: {mode}. Expected 'range', 'sig', or 'wb-sig'.")


def _build_spec(
    sub_cfg: dict,
    mod_x: np.ndarray,
    mod_y: np.ndarray,
    lat_x: np.ndarray,
    lat_y: np.ndarray,
    *,
    figure_id: str,
    suffix: str,
    x_lim: list[float],
    y_lim: list[float],
) -> OutputSpec:
    mod_fit_x, mod_fit_y, mod_fit_label = fit_line_with_stats(mod_x, mod_y, label_prefix="mod fit")
    lat_fit_x, lat_fit_y, lat_fit_label = fit_line_with_stats(lat_x, lat_y, label_prefix="lat fit")

    series_list = [
        Series(
            kind=PlotKind.SCATTER,
            label="mod",
            x=[float(v) for v in mod_x],
            y=[float(v) for v in mod_y],
            color=Color.BLUE,
            marker="o",
            alpha=0.5,
        ),
        Series(
            kind=PlotKind.SCATTER,
            label="lat",
            x=[float(v) for v in lat_x],
            y=[float(v) for v in lat_y],
            color=Color.RED,
            marker="o",
            alpha=0.5,
        ),
    ]

    if mod_fit_x.size > 0:
        series_list.append(
            Series(
                kind=PlotKind.LINE,
                label=mod_fit_label,
                x=[float(v) for v in mod_fit_x],
                y=[float(v) for v in mod_fit_y],
                color=Color.BLUE,
                marker=None,
                linewidth=2.0,
                alpha=0.9,
            )
        )

    if lat_fit_x.size > 0:
        series_list.append(
            Series(
                kind=PlotKind.LINE,
                label=lat_fit_label,
                x=[float(v) for v in lat_fit_x],
                y=[float(v) for v in lat_fit_y],
                color=Color.RED,
                marker=None,
                linewidth=2.0,
                alpha=0.9,
            )
        )

    return OutputSpec(
        figure_id=figure_id,
        title=f"K95 vs Hidden Correlation ({suffix})",
        x_label="Hidden Correlation",
        y_label="Category K95",
        x_lim=x_lim,
        y_lim=y_lim,
        grid=True,
        legend_loc="best",
        legend_fontsize=sub_cfg.get("legend_fontsize", 8),
        figsize=tuple(sub_cfg.get("figsize", (12, 8))),
        dpi=int(sub_cfg.get("dpi", 300)),
        series_list=series_list,
    )
