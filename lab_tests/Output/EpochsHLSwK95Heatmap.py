import os
import re
import numpy as np

from Output.schema.OutputSpec import OutputSpec


class EpochsHLSwK95HeatmapOutput:
    name = "Epochs-HLSwK95Heatmap"
    hyperd = True

    def generate_output(self, spec_cfg: dict, analysis_root: str) -> list[OutputSpec]:
        runs = _discover_k95_runs(analysis_root)
        if not runs:
            raise FileNotFoundError(f"No K95.npz files found under {analysis_root}")

        by_lr = {}
        for run in runs:
            by_lr.setdefault(run["lr"], []).append(run)

        step = int(spec_cfg.get("epoch_step", 3))

        specs = []
        for lr in sorted(by_lr.keys()):
            lr_runs = sorted(by_lr[lr], key=lambda d: d["hls"])
            if not lr_runs:
                continue

            sample_epochs = _sample_epochs(lr_runs[0]["raw"], step=step)

            hls_values = [run["hls"] for run in lr_runs]
            mean_avg_mat = np.full((len(sample_epochs), len(lr_runs)), np.nan, dtype=np.float64)
            diff_mat = np.full((len(sample_epochs), len(lr_runs)), np.nan, dtype=np.float64)

            for j, run in enumerate(lr_runs):
                raw = run["raw"]  # expected (M,E,2,1) or (M,E,2)
                per_epoch_mean_avg, per_epoch_diff = _epoch_k95_summaries(raw)

                mean_avg_mat[:, j] = per_epoch_mean_avg[sample_epochs]
                diff_mat[:, j] = per_epoch_diff[sample_epochs]

            lr_str = _fmt_lr(lr)

            specs.append(
                OutputSpec(
                    figure_id=_lr_figure_id(
                        spec_cfg.get("name", "epochs_hls_k95_mean"),
                        lr,
                        suffix="meanavg",
                    ),
                    title=spec_cfg.get(
                        "title_mean",
                        f"Mean K95 Across Categories by Epoch and HLS (LR={lr_str})",
                    ),
                    x_label="Hidden Layer Size",
                    y_label="Epoch",
                    matrix=mean_avg_mat.tolist(),
                    grid=False,
                    figsize=tuple(spec_cfg.get("figsize", (10, 8))),
                    dpi=int(spec_cfg.get("dpi", 300)),
                    x_ticks=[float(i) for i in range(len(hls_values))],
                    x_ticklabels=[str(h) for h in hls_values],
                    y_ticks=[float(i) for i in range(len(sample_epochs))],
                    y_ticklabels=[str(e) for e in sample_epochs]
                )
            )

            specs.append(
                OutputSpec(
                    figure_id=_lr_figure_id(
                        spec_cfg.get("name", "epochs_hls_k95_diff"),
                        lr,
                        suffix="mod_minus_lat",
                    ),
                    title=spec_cfg.get(
                        "title_diff",
                        f"K95 Mod - Lat by Epoch and HLS (LR={lr_str})",
                    ),
                    x_label="Hidden Layer Size",
                    y_label="Epoch",
                    matrix=diff_mat.tolist(),
                    grid=False,
                    figsize=tuple(spec_cfg.get("figsize", (10, 8))),
                    dpi=int(spec_cfg.get("dpi", 300)),
                    x_ticks=[float(i) for i in range(len(hls_values))],
                    x_ticklabels=[str(h) for h in hls_values],
                    y_ticks=[float(i) for i in range(len(sample_epochs))],
                    y_ticklabels=[str(e) for e in sample_epochs],
                    matrix_split=0.0 
                )
            )

        return specs


def _discover_k95_runs(analysis_root: str) -> list[dict]:
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

        data = np.load(k95_path)
        raw = np.asarray(data["raw"], dtype=np.float64)

        runs.append(
            {
                "analysis_dir": full,
                "hls": int(m.group(1)),
                "lr": _parse_lr_token(m.group(2)),
                "raw": raw,
            }
        )

    return runs


def _epoch_k95_summaries(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Input:
        raw shape (M,E,2,1) or (M,E,2)

    Returns:
        per_epoch_mean_avg : shape (E,)
            mean across models of ((mod + lat) / 2)

        per_epoch_diff : shape (E,)
            mean across models of (mod - lat)
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

    mod = x[:, :, 0]  # (M,E)
    lat = x[:, :, 1]  # (M,E)

    mean_avg = np.nanmean((mod + lat) / 2.0, axis=0)  # (E,)
    diff = np.nanmean(mod - lat, axis=0)              # (E,)

    return mean_avg, diff


def _sample_epochs(raw: np.ndarray, step: int = 3) -> np.ndarray:
    x = np.asarray(raw)
    if x.ndim < 2:
        raise ValueError(f"Expected raw with epoch axis, got shape {x.shape}")

    n_epochs = int(x.shape[1])
    last = n_epochs // 2
    return np.arange(0, last + 1, step, dtype=np.int64)


def _parse_lr_token(token: str) -> float:
    return float(token.replace("p", "."))


def _fmt_lr(lr: float) -> str:
    return f"{lr:g}"


def _lr_figure_id(base: str, lr: float, suffix: str) -> str:
    return f"{base}_lr{str(lr).replace('.', 'p')}_{suffix}"