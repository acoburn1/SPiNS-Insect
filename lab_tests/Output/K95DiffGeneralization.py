import os
import numpy as np
from scipy.stats import linregress

from Output.schema.OutputSpec import *
from Output.utils import first_sig_epochs, normalize_k95_raw, resolve_epoch_range, weighted_single_ratio


class K95DiffGeneralizationOutput:
    name = "K95DiffGeneralization"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        ratio_name = str(sub_cfg.get("ratio", "3:3"))

        ratio_np = np.load(os.path.join(analysis_dir, "RatioTest.npz"), allow_pickle=True)
        k95_np = np.load(os.path.join(analysis_dir, "K95.npz"))

        raw_ratio = np.asarray(ratio_np["raw"], dtype=np.float64)  # (M,E,R,S)
        raw_k95 = normalize_k95_raw(np.asarray(k95_np["raw"], dtype=np.float64))

        ratio_labels = _get_ratio_labels(ratio_np)
        trial_counts = _get_trial_counts(ratio_np)

        if ratio_name not in ratio_labels:
            raise ValueError(f"Requested ratio '{ratio_name}' not found. Available: {ratio_labels}")

        r_idx = ratio_labels.index(ratio_name)
        pref = weighted_single_ratio(raw_ratio[:, :, r_idx, :], trial_counts[r_idx, :])
        k95_diff = raw_k95[:, :, 0] - raw_k95[:, :, 1]

        mode = str(sub_cfg.get("epochs", "sig")).lower()
        if mode == "sig":
            sig_epochs = first_sig_epochs(analysis_dir, pref.shape[0], pref.shape[1])
            x, y = _points_at_epochs(k95_diff, pref, sig_epochs)
            y_lim = [0.0, 1.0]
            x_lim = _shared_lim([x])
            return [
                _build_spec(
                    sub_cfg,
                    x,
                    y,
                    figure_id=sub_cfg.get("name", "k95diff_generalization_sig"),
                    suffix="sig",
                    x_lim=x_lim,
                    y_lim=y_lim,
                )
            ]

        if mode == "range":
            epoch_indices = resolve_epoch_range(sub_cfg, pref.shape[1], default_start=0)
            per_epoch = []
            for e in epoch_indices:
                x, y = _finite_pair(k95_diff[:, e], pref[:, e])
                per_epoch.append((x, y))

            x_lim = _shared_lim([p[0] for p in per_epoch])
            y_lim = [0.0, 1.0]

            specs = []
            for e, (x, y) in zip(epoch_indices, per_epoch):
                specs.append(
                    _build_spec(
                        sub_cfg,
                        x,
                        y,
                        figure_id=f"k95diff_generalization_e{e:03d}",
                        suffix=f"epoch {e}",
                        x_lim=x_lim,
                        y_lim=y_lim,
                    )
                )
            return specs

        raise ValueError(f"Unsupported epochs mode: {mode}. Expected 'range' or 'sig'.")


def _build_spec(
    sub_cfg: dict,
    x: np.ndarray,
    y: np.ndarray,
    *,
    figure_id: str,
    suffix: str,
    x_lim: list[float] | None,
    y_lim: list[float],
) -> OutputSpec:
    fit_x, fit_y, fit_label = _fit_line(x, y)
    series_list = [
        Series(
            kind=PlotKind.SCATTER,
            label="models",
            x=[float(v) for v in x],
            y=[float(v) for v in y],
            color=Color.BLUE,
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
                color=Color.RED,
                marker=None,
                linewidth=2.0,
                alpha=0.9,
            )
        )

    return OutputSpec(
        figure_id=figure_id,
        title=f"Modular Preference vs Mod-Lat K95 ({suffix})",
        x_label="Mod-Lat K95",
        y_label="Modular Preference",
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


def _shared_lim(arrays: list[np.ndarray]) -> list[float] | None:
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size > 0]) if arrays else np.asarray([], dtype=np.float64)
    if vals.size == 0:
        return None

    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if lo == hi:
        return [lo - 0.5, hi + 0.5]

    span = hi - lo
    return [lo - 0.05 * span, hi + 0.05 * span]


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), "fit unavailable"

    xv = x[good]
    yv = y[good]
    reg = linregress(xv, yv)

    x0 = float(np.nanmin(xv))
    x1 = float(np.nanmax(xv))
    if x0 == x1:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), "fit unavailable"

    fx = np.asarray([x0, x1], dtype=np.float64)
    fy = reg.slope * fx + reg.intercept
    label = f"fit (r={reg.rvalue:.3f}, p={reg.pvalue:.3g})"
    return fx, fy, label


def _get_ratio_labels(ratio_np: np.lib.npyio.NpzFile) -> list[str]:
    metadata = ratio_np["metadata"].item() if "metadata" in ratio_np else None
    labels = list(metadata.get("ratio_labels", [])) if metadata is not None else []
    if not labels and "ratio_labels" in ratio_np:
        labels = [str(v) for v in ratio_np["ratio_labels"].tolist()]
    return labels


def _get_trial_counts(ratio_np: np.lib.npyio.NpzFile) -> np.ndarray:
    metadata = ratio_np["metadata"].item() if "metadata" in ratio_np else None
    tc = None
    if metadata is not None and "trial_counts" in metadata:
        tc = np.asarray(metadata["trial_counts"], dtype=np.float64)
    elif "trial_counts" in ratio_np:
        tc = np.asarray(ratio_np["trial_counts"], dtype=np.float64)

    if tc is None:
        raise ValueError("RatioTest.npz is missing trial_counts metadata.")

    return tc
