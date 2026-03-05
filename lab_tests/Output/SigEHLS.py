import os
import re
import numpy as np
from Output.schema.OutputSpec import *


class SigEHLSOutput:
    name = "SigE-HLS"
    hyperd = True

    def generate_output(self, spec_cfg: dict, analysis_root: str) -> list[OutputSpec]:
        runs = _discover_sige_runs(analysis_root)
        if not runs:
            raise FileNotFoundError(f"No sige.npz files found under {analysis_root}")

        by_lr = {}
        for run in runs:
            by_lr.setdefault(run["lr"], []).append(run)

        specs = []
        for lr in sorted(by_lr.keys()):
            lr_runs = sorted(by_lr[lr], key=lambda d: d["hls"])

            xs = []
            ys = []
            hls_values = []

            ymax = -np.inf

            for run in lr_runs:
                hls = run["hls"]
                first_sig = _first_sig_epochs(run["sig_mask"])  # (M,)

                good = np.isfinite(first_sig)
                vals = first_sig[good]
                if vals.size == 0:
                    continue

                pts_x = _spread_x(
                    center=float(hls),
                    n=vals.size,
                    width=float(spec_cfg.get("jitter_width", 0.18)),
                )

                xs.extend(pts_x.tolist())
                ys.extend(vals.tolist())
                hls_values.append(hls)
                ymax = max(ymax, float(np.nanmax(vals)))

            if not hls_values:
                continue

            if not np.isfinite(ymax):
                ymax = 1.0

            specs.append(
                OutputSpec(
                    figure_id=_lr_figure_id(spec_cfg.get("name", "sige_hls"), lr),
                    title=spec_cfg.get(
                        "title",
                        f"First Significant Epoch vs Hidden Layer Size (LR={_fmt_lr(lr)})",
                    ),
                    x_label="Hidden Layer Size",
                    y_label="First Significant Epoch",
                    x_ticks=[float(h) for h in hls_values],
                    x_ticklabels=[str(h) for h in hls_values],
                    x_lim=[min(hls_values) - 0.6, max(hls_values) + 0.6],
                    y_lim=[
                        float(spec_cfg.get("y_min", 0.0)),
                        float(spec_cfg.get("y_max", ymax + float(spec_cfg.get("top_pad", 1.0)))),
                    ],
                    grid=True,
                    legend_loc=spec_cfg.get("legend_loc", "best"),
                    legend_ncol=spec_cfg.get("legend_ncol", 1),
                    legend_fontsize=spec_cfg.get("legend_fontsize", 8),
                    figsize=tuple(spec_cfg.get("figsize", (12, 8))),
                    dpi=int(spec_cfg.get("dpi", 300)),
                    series_list=[
                        Series(
                            kind=PlotKind.SCATTER,
                            label="Models",
                            x=[float(v) for v in xs],
                            y=[float(v) for v in ys],
                            color=Color.BLUE,
                            marker="o",
                            alpha=float(spec_cfg.get("alpha", 0.5)),
                        )
                    ],
                )
            )

        return specs


def _discover_sige_runs(analysis_root: str) -> list[dict]:
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

        sige_path = os.path.join(full, "sige.npz")
        if not os.path.exists(sige_path):
            continue

        data = np.load(sige_path)
        sig_mask = np.asarray(data["results"]).astype(bool)

        runs.append(
            {
                "analysis_dir": full,
                "hls": int(m.group(1)),
                "lr": _parse_lr_token(m.group(2)),
                "sig_mask": sig_mask,
            }
        )

    return runs


def _first_sig_epochs(sig_mask: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig_mask, dtype=bool)
    if sig.ndim != 2:
        raise ValueError(f"Expected sige results shape (M, E), got {sig.shape}")

    M = sig.shape[0]
    out = np.full((M,), np.nan, dtype=np.float64)

    for m in range(M):
        idx = np.flatnonzero(sig[m])
        if idx.size > 0:
            out[m] = float(idx[0])

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


def _lr_figure_id(base: str, lr: float) -> str:
    return f"{base}_lr{str(lr).replace('.', 'p')}"