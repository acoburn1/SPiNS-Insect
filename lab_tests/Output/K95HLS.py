import os
import re
import numpy as np
from scipy.stats import ttest_rel
from Output.schema.OutputSpec import *

class K95HLSOutput:
    name = "K95-HLS"
    hyperd = True

    def generate_output(self, spec_cfg: dict, analysis_root: str) -> list[OutputSpec]:
        runs = _discover_k95_runs(analysis_root)
        if not runs:
            raise FileNotFoundError(f"No K95.npz files found under {analysis_root}")

        by_lr = {}
        for run in runs:
            by_lr.setdefault(run["lr"], []).append(run)

        specs = []
        for lr in sorted(by_lr.keys()):
            lr_runs = sorted(by_lr[lr], key=lambda d: d["hls"])

            mod_x = []
            mod_y = []
            lat_x = []
            lat_y = []

            mod_mean_x = []
            mod_mean_y = []
            lat_mean_x = []
            lat_mean_y = []

            sig_x = []
            sig_y = []

            hls_values = []
            ymax = -np.inf

            for run in lr_runs:
                hls = run["hls"]
                raw = run["raw"]  # expected (M, E, 2, 1) or (M, E, 2)

                per_model = _avg_over_epochs_per_model(raw)  # (M, 2)
                mod_vals = per_model[:, 0]
                lat_vals = per_model[:, 1]

                hls_values.append(hls)

                x_mod_center = hls - 0.6
                x_lat_center = hls + 0.6

                x_mod_pts = _spread_x(x_mod_center, mod_vals.size, width=0.10)
                x_lat_pts = _spread_x(x_lat_center, lat_vals.size, width=0.10)

                mod_x.extend(x_mod_pts.tolist())
                mod_y.extend(mod_vals.tolist())

                lat_x.extend(x_lat_pts.tolist())
                lat_y.extend(lat_vals.tolist())

                mod_mean = float(np.nanmean(mod_vals))
                lat_mean = float(np.nanmean(lat_vals))

                mod_mean_x.append(float(x_mod_center))
                mod_mean_y.append(mod_mean)

                lat_mean_x.append(float(x_lat_center))
                lat_mean_y.append(lat_mean)

                local_max = np.nanmax(np.concatenate([mod_vals, lat_vals]))
                ymax = max(ymax, local_max)

                p = _paired_p_value(mod_vals, lat_vals)
                if np.isfinite(p) and p < float(spec_cfg.get("alpha", 0.05)):
                    sig_x.append(float(hls))
                    sig_y.append(float(local_max + spec_cfg.get("sig_y_pad", 0.35)))

            if not np.isfinite(ymax):
                ymax = 1.0

            y_top = max(
                ymax + float(spec_cfg.get("top_pad", 0.8)),
                max(sig_y) + float(spec_cfg.get("sig_top_extra", 0.25)) if sig_y else ymax + 0.8,
            )

            figure_id = _lr_figure_id(spec_cfg.get("name", "k95_hls"), lr)
            title = spec_cfg.get("title", f"K95 vs Hidden Layer Size (LR={_fmt_lr(lr)})")
            if "title" not in spec_cfg:
                title = f"K95 vs Hidden Layer Size (LR={_fmt_lr(lr)})"

            specs.append(
                OutputSpec(
                    figure_id=figure_id,
                    title=title,
                    x_label="Hidden Layer Size",
                    y_label="Mean K95 Across Epochs",
                    x_ticks=[float(h) for h in hls_values],
                    x_ticklabels=[str(h) for h in hls_values],
                    x_lim=[min(hls_values) - 0.6, max(hls_values) + 0.6],
                    y_lim=[
                        float(spec_cfg.get("y_min", 0.0)),
                        float(spec_cfg.get("y_max", y_top)),
                    ],
                    grid=True,
                    legend_loc=spec_cfg.get("legend_loc", "upper left"),
                    legend_fontsize=spec_cfg.get("legend_fontsize", 9),
                    figsize=tuple(spec_cfg.get("figsize", (12, 8))),
                    dpi=int(spec_cfg.get("dpi", 300)),
                    series_list=[
                        _marker_series(
                            label="Mod models",
                            x=mod_x,
                            y=mod_y,
                            color=Color.BLUE,
                            marker="o",
                            markersize=float(spec_cfg.get("model_marker_size", 5.0)),
                            alpha=float(spec_cfg.get("model_alpha", 0.35)),
                        ),
                        _marker_series(
                            label="Lat models",
                            x=lat_x,
                            y=lat_y,
                            color=Color.RED,
                            marker="o",
                            markersize=float(spec_cfg.get("model_marker_size", 5.0)),
                            alpha=float(spec_cfg.get("model_alpha", 0.35)),
                        ),
                        _marker_series(
                            label="Mod mean",
                            x=mod_mean_x,
                            y=mod_mean_y,
                            color=Color.BLUE,
                            marker="x",
                            markersize=float(spec_cfg.get("mean_marker_size", 10.0)),
                            alpha=1.0,
                        ),
                        _marker_series(
                            label="Lat mean",
                            x=lat_mean_x,
                            y=lat_mean_y,
                            color=Color.RED,
                            marker="x",
                            markersize=float(spec_cfg.get("mean_marker_size", 10.0)),
                            alpha=1.0,
                        ),
                        _marker_series(
                            label=f"p < {spec_cfg.get('alpha', 0.05)}",
                            x=sig_x,
                            y=sig_y,
                            color=Color.BLACK,
                            marker="*",
                            markersize=float(spec_cfg.get("sig_marker_size", 12.0)),
                            alpha=1.0,
                        ),
                    ],
                )
            )

        return specs


def _discover_k95_runs(analysis_root: str) -> list[dict]:
    """
    Find directories like:
        <analysis_root>_hls10_lr0p04
    containing K95.npz
    """
    parent = os.path.dirname(analysis_root.rstrip("/\\"))
    base = os.path.basename(analysis_root.rstrip("/\\"))
    if not parent:
        parent = "."

    pattern = re.compile(rf"^{re.escape(base)}_hls(\d+)_lr([A-Za-z0-9p\-]+)$")

    runs = []
    for name in os.listdir(parent):
        full = os.path.join(parent, name)
        if not os.path.isdir(full):
            continue

        m = pattern.match(name)
        if not m:
            continue

        k95_path = os.path.join(full, "K95.npz")
        if not os.path.exists(k95_path):
            continue

        hls = int(m.group(1))
        lr = _parse_lr_token(m.group(2))

        data = np.load(k95_path)
        raw = np.asarray(data["raw"], dtype=np.float64)

        runs.append(
            {
                "analysis_dir": full,
                "hls": hls,
                "lr": lr,
                "raw": raw,
            }
        )

    return runs


def _avg_over_epochs_per_model(raw: np.ndarray) -> np.ndarray:
    """
    Accepts:
        (M, E, 2, 1) or (M, E, 2)

    Returns:
        (M, 2)
    """
    x = np.asarray(raw, dtype=np.float64)

    if x.ndim == 4:
        if x.shape[2] != 2 or x.shape[3] != 1:
            raise ValueError(f"Expected K95 raw shape (M, E, 2, 1), got {x.shape}")
        x = x[..., 0]
    elif x.ndim == 3:
        if x.shape[2] != 2:
            raise ValueError(f"Expected K95 raw shape (M, E, 2), got {x.shape}")
    else:
        raise ValueError(f"Expected K95 raw ndim 3 or 4, got shape {x.shape}")

    return np.nanmean(x, axis=1)


def _paired_p_value(mod_vals: np.ndarray, lat_vals: np.ndarray) -> float:
    mod_vals = np.asarray(mod_vals, dtype=np.float64)
    lat_vals = np.asarray(lat_vals, dtype=np.float64)

    good = np.isfinite(mod_vals) & np.isfinite(lat_vals)
    if np.sum(good) < 2:
        return np.nan

    res = ttest_rel(mod_vals[good], lat_vals[good], nan_policy="omit")
    p = getattr(res, "pvalue", np.nan)
    return float(p) if np.isfinite(p) else np.nan


def _spread_x(center: float, n: int, width: float = 0.10) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    if n == 1:
        return np.asarray([center], dtype=np.float64)
    return np.linspace(center - width / 2.0, center + width / 2.0, n, dtype=np.float64)


def _marker_series(
    *,
    label: str,
    x: list[float],
    y: list[float],
    color: Color,
    marker: str,
    markersize: float,
    alpha: float,
) -> Series:
    return Series(
        kind=PlotKind.LINE,
        label=label,
        x=[float(v) for v in x],
        y=[float(v) for v in y],
        color=color,
        marker=marker,
        markersize=float(markersize),
        linewidth=0.0,
        alpha=float(alpha),
    )


def _parse_lr_token(token: str) -> float:
    return float(token.replace("p", "."))


def _fmt_lr(lr: float) -> str:
    return f"{lr:g}"


def _lr_figure_id(base: str, lr: float) -> str:
    return f"{base}_lr{str(lr).replace('.', 'p')}"