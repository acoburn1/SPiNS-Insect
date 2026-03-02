import os
import numpy as np
import matplotlib.pyplot as plt
from Output.OutputSpec import OutputSpec, PlotKind

def save_s_curve_output(spec: "OutputSpec", save_dir: str, *, alt: bool = False):
    """
    Custom plotter for S-curve specs. Matches the style of your old plot_s_curve():
      - x positions fixed to 0..6
      - errorbar with circle markers, thick line, caps
      - xlabel '# mod feats', ylabel '% mod resp'
      - y in [0,1], grid
      - dashed hline at 0.5, dashed vline at 3
      - saves <figure_id>.png
    """
    os.makedirs(save_dir, exist_ok=True)

    if spec.series_list is None or len(spec.series_list) == 0:
        raise ValueError("S-curve OutputSpec must contain series_list.")

    # take first series (your S-curve spec should only have one)
    s = spec.series_list[0]

    x = np.asarray(s.x, dtype=float)
    y = np.asarray(s.y, dtype=float)
    yerr = None
    if getattr(s, "yerr", None) is not None:
        yerr = np.asarray(s.yerr, dtype=float)

    # enforce canonical x positions 0..6 and align y/yerr accordingly
    x_positions = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)

    if x.size != 7 or y.size != 7:
        raise ValueError(f"S-curve expects exactly 7 points; got x={x.size}, y={y.size}")

    x_int = np.rint(x).astype(int)
    if set(x_int.tolist()) == set(range(7)):
        order = np.argsort(x_int)
        x_int = x_int[order]
        y = y[order]
        if yerr is not None and yerr.size == 7:
            yerr = yerr[order]

        y_binned = np.full((7,), np.nan, dtype=float)
        yerr_binned = np.full((7,), np.nan, dtype=float) if yerr is not None else None

        for i in range(7):
            y_binned[int(x_int[i])] = float(y[i])
            if yerr is not None:
                yerr_binned[int(x_int[i])] = float(yerr[i])

        y = y_binned
        yerr = yerr_binned
    else:
        # fallback: sort by x and assume it corresponds to 0..6
        order = np.argsort(x)
        y = y[order]
        if yerr is not None and yerr.size == 7:
            yerr = yerr[order]

    fig, ax = plt.subplots(figsize=(10, 8))

    color = "blue"
    if getattr(s, "color", None) is not None:
        try:
            color = s.color.value
        except Exception:
            pass

    ax.errorbar(
        x_positions,
        y,
        yerr=yerr,
        marker="o",
        markersize=8,
        linewidth=2,
        capsize=5,
        capthick=2,
        color=color,
        label=getattr(s, "label", None) if getattr(s, "label", None) else None,
    )

    ax.set_xlabel("# mod feats", fontsize=14)
    ax.set_ylabel("% mod resp", fontsize=14)
    ax.set_title(spec.title or "Modular Response by Feature Composition", fontsize=14)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(i) for i in range(7)])

    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=3, color="gray", linestyle="--", alpha=0.7)

    if getattr(s, "label", None):
        ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{spec.figure_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_output(spec: "OutputSpec", save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    has_series = spec.series_list is not None and len(spec.series_list) > 0
    has_matrix = spec.matrix is not None and len(spec.matrix) > 0

    if has_series == has_matrix:
        raise ValueError("OutputSpec must contain either series_list or matrix, but not both.")

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = None

    def _is_loss_series(s) -> bool:
        lab = (getattr(s, "label", "") or "").lower()
        return ("loss" in lab) or (lab.strip() == "loss") or ("losses" in lab)

    def _get_ci_arrays(s):
        lo = getattr(s, "ci_lower", None)
        hi = getattr(s, "ci_upper", None)
        if lo is None or hi is None:
            return None, None
        if len(lo) == 0 or len(hi) == 0:
            return None, None
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    scatter_x_all = []
    scatter_y_all = []

    # collect all non-loss y values to decide whether to clamp to [0,1]
    nonloss_y_all = []

    corr_lines = []
    loss_lines = []

    if has_series:
        for s in spec.series_list:
            x = np.asarray(s.x, dtype=float)
            y = np.asarray(s.y, dtype=float)

            if not _is_loss_series(s) and y.size:
                nonloss_y_all.append(y)

            kwargs = {}
            if getattr(s, "color", None):
                kwargs["color"] = s.color.value
            if getattr(s, "linestyle", None):
                kwargs["linestyle"] = s.linestyle.value

            target_ax = ax1
            if _is_loss_series(s):
                if ax2 is None:
                    ax2 = ax1.twinx()
                target_ax = ax2

            if s.kind == PlotKind.LINE:
                lo, hi = _get_ci_arrays(s)

                if lo is not None and hi is not None and len(lo) == len(x) == len(hi):
                    line = target_ax.plot(
                        x, y,
                        linewidth=2,
                        marker="o",
                        markersize=4,
                        label=s.label,
                        **kwargs
                    )
                    for xi, l, u in zip(x, lo, hi):
                        if np.isfinite(xi) and np.isfinite(l) and np.isfinite(u):
                            target_ax.plot([xi, xi], [l, u], linewidth=1, alpha=0.7, **kwargs)
                else:
                    yerr = getattr(s, "yerr", None)
                    if yerr is not None and len(yerr) == len(x):
                        target_ax.errorbar(
                            x, y,
                            yerr=np.asarray(yerr, dtype=float),
                            linewidth=2,
                            marker="o",
                            markersize=4,
                            label=s.label,
                            **kwargs
                        )
                        line = []
                    else:
                        line = target_ax.plot(
                            x, y,
                            linewidth=2,
                            marker="o",
                            markersize=4,
                            label=s.label,
                            **kwargs
                        )

                if target_ax is ax2:
                    loss_lines.extend(line if isinstance(line, list) else [])
                else:
                    corr_lines.extend(line if isinstance(line, list) else [])

            elif s.kind == PlotKind.SCATTER:
                target_ax.scatter(x, y, label=s.label, alpha=0.25, **kwargs)
                if target_ax is ax1 and x.size:
                    scatter_x_all.append(x.astype(float, copy=False))
                    scatter_y_all.append(y.astype(float, copy=False))

            elif s.kind == PlotKind.BAR:
                target_ax.bar(x, y, label=s.label, **kwargs)

        ax1.set_xlabel(spec.x_label or "Epoch", fontsize=12)
        ax1.set_ylabel(spec.y_label or "Value", fontsize=12, color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.grid(True, alpha=0.3)

        # ---- IMPORTANT FIX: clamp to [0,1] ONLY if data actually looks like correlations ----
        if nonloss_y_all:
            yy = np.concatenate(nonloss_y_all)
            yy = yy[np.isfinite(yy)]
            if yy.size:
                y_min = float(np.min(yy))
                y_max = float(np.max(yy))
                # clamp only if it's basically within correlation bounds
                if (y_min >= -0.05) and (y_max <= 1.05):
                    ax1.set_ylim(0, 1)

        if ax2 is not None:
            ax2.set_ylabel("Loss", fontsize=12, color="black")
            ax2.set_ylim(0, 2)
            ax2.tick_params(axis="y", labelcolor="black")

        # ---- epoch ticks styling ONLY when x-axis is actually epochs ----
        xlab = (spec.x_label or "").lower()
        if "epoch" in xlab:
            x_for_ticks = None
            for s in spec.series_list:
                if getattr(s, "x", None) is not None and len(s.x) > 0:
                    x_for_ticks = np.asarray(s.x, dtype=float)
                    break

            if x_for_ticks is not None and x_for_ticks.size:
                finite_x = x_for_ticks[np.isfinite(x_for_ticks)]
                if finite_x.size:
                    min_epoch = int(np.min(np.round(finite_x)))
                    max_epoch = int(np.max(np.round(finite_x)))

                    major_ticks = list(range(10, max_epoch + 1, 10))
                    if min_epoch not in major_ticks:
                        major_ticks = [min_epoch] + major_ticks
                    if max_epoch not in major_ticks and max_epoch % 10 != 0:
                        major_ticks.append(max_epoch)

                    medium_ticks = [t for t in range(5, max_epoch + 1, 5) if t not in major_ticks and t >= min_epoch]
                    all_int_ticks = list(range(min_epoch, max_epoch + 1))
                    minor_ticks = [t for t in all_int_ticks if t not in major_ticks and t not in medium_ticks]

                    ax1.set_xticks(major_ticks)
                    ax1.set_xticks(medium_ticks, minor=False)
                    ax1.set_xticks(minor_ticks, minor=True)
                    ax1.tick_params(which="major", length=8, width=2, labelsize=10)
                    ax1.tick_params(which="minor", length=4, width=1)

                    for t in medium_ticks:
                        ax1.axvline(x=t, ymin=0, ymax=0.02, color="black", linewidth=1.5, clip_on=False)

                    ax1.set_xlim(min_epoch, max_epoch)

        # mean-per-x cross overlay for scatter points
        if scatter_x_all:
            sx = np.concatenate(scatter_x_all)
            sy = np.concatenate(scatter_y_all)
            m = np.isfinite(sx) & np.isfinite(sy)
            sx = sx[m]
            sy = sy[m]
            if sx.size:
                uniq_x = np.unique(sx)
                mean_x = []
                mean_y = []
                for xv in uniq_x:
                    mm = sx == xv
                    if np.any(mm):
                        mean_x.append(float(xv))
                        mean_y.append(float(np.mean(sy[mm])))
                ax1.scatter(
                    np.asarray(mean_x),
                    np.asarray(mean_y),
                    marker="x",
                    s=120,
                    linewidths=2,
                    label="mean per x",
                    color="black",
                    alpha=0.9,
                )

        # legend smaller
        if ax2 is not None:
            all_lines = corr_lines + loss_lines
            if all_lines:
                labels = [l.get_label() for l in all_lines]
                ax1.legend(all_lines, labels, loc="best", fontsize=8)
            else:
                ax1.legend(loc="best", fontsize=8)
        else:
            ax1.legend(loc="best", fontsize=8)

    else:
        mat = np.array(spec.matrix)
        im = ax1.imshow(mat, aspect="auto")
        fig.colorbar(im, ax=ax1)

    ax1.set_title(spec.title, fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{spec.figure_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)