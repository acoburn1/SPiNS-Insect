import os
import re
import numpy as np
from Output.schema.OutputSpec import *
from Statistics.StatHelper import stats_over_models


class CorrelationHLSOutput:
    name = "Correlation-HLS"
    hyperd = True

    def generate_output(self, spec_cfg: dict, analysis_root: str) -> list[OutputSpec]:
        runs = _discover_correlation_runs(analysis_root)
        if not runs:
            raise FileNotFoundError(f"No Correlation.npz + sige.npz runs found under {analysis_root}")

        source_name = str(spec_cfg.get("source", "hidden")).lower()
        source_idx = 0 if source_name in ("hidden", "hid") else 1

        by_lr = {}
        for run in runs:
            by_lr.setdefault(run["lr"], []).append(run)

        specs = []
        for lr in sorted(by_lr.keys()):
            lr_runs = sorted(by_lr[lr], key=lambda d: d["hls"])

            scatter_x = []
            scatter_y = []

            bar_x = []
            bar_y = []
            bar_yerr = []

            x_ticks = []
            x_labels = []

            ymax = -np.inf

            for run in lr_runs:
                hls = run["hls"]
                per_model_scores = _scores_at_first_sig_epoch(
                    corr_raw=run["corr_raw"],
                    sig_mask=run["sig_mask"],
                    source_idx=source_idx,
                )

                good = np.isfinite(per_model_scores)
                vals = per_model_scores[good]

                x_ticks.append(float(hls))
                x_labels.append(str(hls))

                if vals.size > 0:
                    xs = _spread_x(float(hls), vals.size, width=float(spec_cfg.get("jitter_width", 0.18)))
                    scatter_x.extend(xs.tolist())
                    scatter_y.extend(vals.tolist())

                    st = stats_over_models(vals[:, None], ci=float(spec_cfg.get("ci", 0.95)))
                    mean = float(st["mean"][0])
                    ci_lo = float(st["ci_lo"][0])
                    ci_hi = float(st["ci_hi"][0])

                    bar_x.append(float(hls))
                    bar_y.append(mean)
                    bar_yerr.append(max(mean - ci_lo, ci_hi - mean))

                    ymax = max(ymax, np.nanmax(vals), ci_hi)

            if not np.isfinite(ymax):
                ymax = 1.0

            title_base = spec_cfg.get("title")
            if title_base is None:
                src_label = "Hidden" if source_idx == 0 else "Output"
                title_base = f"{src_label} Correlation at First Significant Epoch vs HLS (LR={_fmt_lr(lr)})"

            legend_loc = spec_cfg.get("legend_loc", "best")
            legend_ncol = spec_cfg.get("legend_ncol", 1)

            specs.append(
                OutputSpec(
                    figure_id=_lr_figure_id(spec_cfg.get("name", "correlation_hls_scatter"), lr, suffix="scatter"),
                    title=title_base + " — Scatter",
                    x_label="Hidden Layer Size",
                    y_label="Correlation at First Significant Epoch",
                    x_ticks=x_ticks,
                    x_ticklabels=x_labels,
                    x_lim=[min(x_ticks) - 0.6, max(x_ticks) + 0.6],
                    y_lim=[0.0, 1.0],
                    grid=True,
                    legend_loc=legend_loc,
                    legend_ncol=legend_ncol,
                    legend_fontsize=spec_cfg.get("legend_fontsize", 8),
                    figsize=tuple(spec_cfg.get("figsize", (12, 8))),
                    dpi=int(spec_cfg.get("dpi", 300)),
                    series_list=[
                        Series(
                            kind=PlotKind.SCATTER,
                            label="Models",
                            x=[float(v) for v in scatter_x],
                            y=[float(v) for v in scatter_y],
                            color=Color.BLUE,
                            marker="o",
                            alpha=float(spec_cfg.get("scatter_alpha", 0.5)),
                        )
                    ],
                )
            )

            specs.append(
                OutputSpec(
                    figure_id=_lr_figure_id(spec_cfg.get("name", "correlation_hls_bar"), lr, suffix="bar"),
                    title=title_base + " — Mean ± 95% CI",
                    x_label="Hidden Layer Size",
                    y_label="Correlation at First Significant Epoch",
                    x_ticks=x_ticks,
                    x_ticklabels=x_labels,
                    x_lim=[min(x_ticks) - 0.6, max(x_ticks) + 0.6],
                    y_lim=[0.0, 1.0],
                    grid=True,
                    legend_loc=legend_loc,
                    legend_ncol=legend_ncol,
                    legend_fontsize=spec_cfg.get("legend_fontsize", 8),
                    figsize=tuple(spec_cfg.get("figsize", (12, 8))),
                    dpi=int(spec_cfg.get("dpi", 300)),
                    series_list=[
                        Series(
                            kind=PlotKind.BAR,
                            label="Mean",
                            x=[float(v) for v in bar_x],
                            y=[float(v) for v in bar_y],
                            yerr=[float(v) for v in bar_yerr],
                            color=Color.BLUE,
                            alpha=float(spec_cfg.get("bar_alpha", 0.8)),
                        )
                    ],
                )
            )

        return specs


def _discover_correlation_runs(analysis_root: str) -> list[dict]:
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

        corr_path = os.path.join(full, "Correlation.npz")
        sige_path = os.path.join(full, "sige.npz")
        if not (os.path.exists(corr_path) and os.path.exists(sige_path)):
            continue

        corr_npz = np.load(corr_path)
        sige_npz = np.load(sige_path)

        runs.append(
            {
                "analysis_dir": full,
                "hls": int(m.group(1)),
                "lr": _parse_lr_token(m.group(2)),
                "corr_raw": np.asarray(corr_npz["raw"], dtype=np.float64),
                "sig_mask": np.asarray(sige_npz["results"]).astype(bool),
            }
        )

    return runs


def _scores_at_first_sig_epoch(corr_raw: np.ndarray, sig_mask: np.ndarray, source_idx: int) -> np.ndarray:
    """
    corr_raw expected shape: (M, E, 2, 2, 2)
        axis2: source [hid, out]
        axis3: category [mod, lat]
        axis4: stat [r, p]

    sig_mask expected shape: (M, E)

    returns:
        (M,) score per model, NaN when no significant epoch exists
    """
    x = np.asarray(corr_raw, dtype=np.float64)
    sig = np.asarray(sig_mask, dtype=bool)

    if x.ndim != 5 or x.shape[2:] != (2, 2, 2):
        raise ValueError(f"Expected Correlation raw shape (M, E, 2, 2, 2), got {x.shape}")

    if sig.ndim != 2 or sig.shape[0] != x.shape[0] or sig.shape[1] != x.shape[1]:
        raise ValueError(f"Expected sige shape (M, E) matching correlation raw, got corr={x.shape}, sige={sig.shape}")

    M = x.shape[0]
    out = np.full((M,), np.nan, dtype=np.float64)

    for m in range(M):
        idx = np.flatnonzero(sig[m])
        if idx.size == 0:
            continue

        e0 = int(idx[0])
        mod_r = x[m, e0, source_idx, 0, 0]
        lat_r = x[m, e0, source_idx, 1, 0]

        if np.isfinite(mod_r) and np.isfinite(lat_r):
            out[m] = 0.5 * (mod_r + lat_r)

    return out


def _spread_x(center: float, n: int, width: float = 0.18) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    if n == 1:
        return np.asarray([center], dtype=np.float64)
    return np.linspace(center - width / 2.0, center + width / 2.0, n, dtype=np.float64)


def _parse_lr_token(token: str) -> float:
    return float(token.replace("p", "."))


def _fmt_lr(lr: float) -> str:
    return f"{lr:g}"


def _lr_figure_id(base: str, lr: float, suffix: str) -> str:
    return f"{base}_lr{str(lr).replace('.', 'p')}_{suffix}"