import os
import numpy as np
from scipy.stats import linregress

from Output.schema.OutputSpec import *
from Output.utils import first_sig_epochs, normalize_k95_raw, resolve_epoch_range


class K95CorrelationOutput:
    name = "K95Correlation"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        corr_np = np.load(os.path.join(analysis_dir, "Correlation.npz"))
        k95_np = np.load(os.path.join(analysis_dir, "K95.npz"))

        raw_corr = np.asarray(corr_np["raw"], dtype=np.float64)
        raw_k95 = normalize_k95_raw(np.asarray(k95_np["raw"], dtype=np.float64))

        if raw_corr.ndim != 5 or raw_corr.shape[2:] != (2, 2, 2):
            raise ValueError(f"Expected Correlation raw shape (M,E,2,2,2), got {raw_corr.shape}")

        h_mod = raw_corr[:, :, 0, 0, 0]
        h_lat = raw_corr[:, :, 0, 1, 0]
        k_mod = raw_k95[:, :, 0]
        k_lat = raw_k95[:, :, 1]

        mode = str(sub_cfg.get("epochs", "sig")).lower()
        if mode == "sig":
            sig_epochs = first_sig_epochs(analysis_dir, raw_corr.shape[0], raw_corr.shape[1])
            mod_x, mod_y = _points_at_epochs(h_mod, k_mod, sig_epochs)
            lat_x, lat_y = _points_at_epochs(h_lat, k_lat, sig_epochs)
            x_lim = _shared_lim([mod_x, lat_x], fallback=[0.0, 1.0], clamp_01=True)
            y_lim = _shared_lim([mod_y, lat_y], fallback=[0.0, 1.0], clamp_01=False)
            return [
                _build_spec(
                    sub_cfg,
                    mod_x,
                    mod_y,
                    lat_x,
                    lat_y,
                    figure_id=sub_cfg.get("name", "k95_correlation_sig"),
                    suffix="sig",
                    x_lim=x_lim,
                    y_lim=y_lim,
                )
            ]

        if mode == "range":
            epoch_indices = resolve_epoch_range(sub_cfg, raw_corr.shape[1], default_start=0)
            per_epoch = []
            for e in epoch_indices:
                mod_x, mod_y = _finite_pair(h_mod[:, e], k_mod[:, e])
                lat_x, lat_y = _finite_pair(h_lat[:, e], k_lat[:, e])
                per_epoch.append((mod_x, mod_y, lat_x, lat_y))

            x_lim = _shared_lim([v for p in per_epoch for v in (p[0], p[2])], fallback=[0.0, 1.0], clamp_01=True)
            y_lim = _shared_lim([v for p in per_epoch for v in (p[1], p[3])], fallback=[0.0, 1.0], clamp_01=False)

            specs = []
            for e, (mod_x, mod_y, lat_x, lat_y) in zip(epoch_indices, per_epoch):
                specs.append(
                    _build_spec(
                        sub_cfg,
                        mod_x,
                        mod_y,
                        lat_x,
                        lat_y,
                        figure_id=f"k95_correlation_e{e:03d}",
                        suffix=f"epoch {e}",
                        x_lim=x_lim,
                        y_lim=y_lim,
                    )
                )
            return specs

        raise ValueError(f"Unsupported epochs mode: {mode}. Expected 'range' or 'sig'.")


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
    mod_fit_x, mod_fit_y, mod_fit_label = _fit_line(mod_x, mod_y, "mod")
    lat_fit_x, lat_fit_y, lat_fit_label = _fit_line(lat_x, lat_y, "lat")

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


def _points_at_epochs(x_arr: np.ndarray, y_arr: np.ndarray, epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.full((x_arr.shape[0],), np.nan, dtype=np.float64)
    y = np.full((y_arr.shape[0],), np.nan, dtype=np.float64)

    for m in range(x_arr.shape[0]):
        e = epochs[m]
        if not np.isfinite(e):
            continue
        ei = int(e)
        x[m] = x_arr[m, ei]
        y[m] = y_arr[m, ei]

    return _finite_pair(x, y)


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def _shared_lim(arrays: list[np.ndarray], *, fallback: list[float], clamp_01: bool) -> list[float]:
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size > 0]) if arrays else np.asarray([], dtype=np.float64)
    if vals.size == 0:
        return fallback

    if clamp_01:
        return [0.0, 1.0]

    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if lo == hi:
        return [lo - 0.5, hi + 0.5]

    span = hi - lo
    return [lo - 0.05 * span, hi + 0.05 * span]


def _fit_line(x: np.ndarray, y: np.ndarray, prefix: str) -> tuple[np.ndarray, np.ndarray, str]:
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), f"{prefix} fit unavailable"

    xv = x[good]
    yv = y[good]
    reg = linregress(xv, yv)

    x0 = float(np.nanmin(xv))
    x1 = float(np.nanmax(xv))
    if x0 == x1:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), f"{prefix} fit unavailable"

    fx = np.asarray([x0, x1], dtype=np.float64)
    fy = reg.slope * fx + reg.intercept
    label = f"{prefix} fit (r={reg.rvalue:.3f}, p={reg.pvalue:.3g})"
    return fx, fy, label
