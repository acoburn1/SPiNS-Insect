import os
import numpy as np
import matplotlib.pyplot as plt
from Output.OutputSpec import OutputSpec, PlotKind, YAxis, Aspect


def plot_output(spec: OutputSpec, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)

    has_series = bool(spec.series_list)
    has_matrix = spec.matrix is not None

    if has_series == has_matrix:
        raise ValueError("OutputSpec must contain either series_list or matrix, but not both.")

    fig, ax1 = plt.subplots(figsize=spec.figsize)
    ax2 = None

    def _apply_ref_lines(ax, lines, *, vertical: bool):
        if not lines:
            return
        for rl in lines:
            if vertical:
                ax.axvline(
                    x=float(rl.val),
                    linestyle=rl.linestyle.value,
                    color=rl.color.value,
                    alpha=float(rl.alpha),
                    linewidth=float(rl.linewidth),
                )
            else:
                ax.axhline(
                    y=float(rl.val),
                    linestyle=rl.linestyle.value,
                    color=rl.color.value,
                    alpha=float(rl.alpha),
                    linewidth=float(rl.linewidth),
                )

    def _get_color(s):
        if getattr(s, "color", None) is None:
            return {}
        try:
            return {"color": s.color.value}
        except Exception:
            return {}

    def _get_linestyle(s):
        if getattr(s, "linestyle", None) is None:
            return {}
        try:
            return {"linestyle": s.linestyle.value}
        except Exception:
            return {}

    def _get_ci(s):
        lo = getattr(s, "ci_lower", None)
        hi = getattr(s, "ci_upper", None)
        if lo is None or hi is None:
            return None, None
        if len(lo) == 0 or len(hi) == 0:
            return None, None
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    def _get_err(s):
        yerr = getattr(s, "yerr", None)
        if yerr is None:
            return None
        if len(yerr) == 0:
            return None
        return np.asarray(yerr, dtype=float)

    if has_series:
        need_ax2 = any(s.y_axis == YAxis.RIGHT for s in spec.series_list)
        if need_ax2:
            ax2 = ax1.twinx()

        left_lines = []
        right_lines = []

        for s in spec.series_list:
            ax = ax2 if (ax2 is not None and s.y_axis == YAxis.RIGHT) else ax1

            x = np.asarray(s.x, dtype=float)
            y = np.asarray(s.y, dtype=float)

            style = {}
            style.update(_get_color(s))
            style.update(_get_linestyle(s))

            marker = getattr(s, "marker", None)
            markersize = getattr(s, "markersize", None)
            linewidth = getattr(s, "linewidth", None)
            alpha = getattr(s, "alpha", None)

            if marker is not None:
                style["marker"] = marker
            if markersize is not None:
                style["markersize"] = float(markersize)
            if linewidth is not None:
                style["linewidth"] = float(linewidth)
            if alpha is not None:
                style["alpha"] = float(alpha)

            if s.kind == PlotKind.LINE:
                lo, hi = _get_ci(s)
                yerr = _get_err(s)

                if lo is not None and hi is not None and len(lo) == len(x) == len(hi):
                    lines = ax.plot(x, y, label=s.label, **style)
                    for xi, l, u in zip(x, lo, hi):
                        if np.isfinite(xi) and np.isfinite(l) and np.isfinite(u):
                            ax.plot([xi, xi], [l, u], linewidth=1, alpha=min(0.9, style.get("alpha", 1.0)), **{k: v for k, v in style.items() if k in ("color",)})
                elif yerr is not None and len(yerr) == len(x):
                    ax.errorbar(x, y, yerr=yerr, label=s.label, capsize=4, **style)
                    lines = []
                else:
                    lines = ax.plot(x, y, label=s.label, **style)

                if ax is ax1:
                    if isinstance(lines, list):
                        left_lines.extend(lines)
                else:
                    if isinstance(lines, list):
                        right_lines.extend(lines)

            elif s.kind == PlotKind.SCATTER:
                alpha_s = spec.scatter_alpha if spec.scatter_alpha is not None else 0.25
                marker_s = spec.scatter_marker if spec.scatter_marker is not None else style.get("marker", None)
                size_s = spec.scatter_size if spec.scatter_size is not None else None

                kw = {}
                kw.update(_get_color(s))
                kw.update(_get_linestyle(s))
                kw["alpha"] = float(alpha_s)
                if marker_s is not None:
                    kw["marker"] = marker_s
                if size_s is not None:
                    kw["s"] = float(size_s)

                ax.scatter(x, y, label=s.label, **kw)

            elif s.kind == PlotKind.BAR:
                kw = {}
                kw.update(_get_color(s))
                ax.bar(x, y, label=s.label, **kw)

        ax1.set_xlabel(spec.x_label or "", fontsize=12)
        ax1.set_ylabel(spec.y_label or "", fontsize=12)

        if ax2 is not None:
            ax2.set_ylabel(spec.y2_label or "", fontsize=12)

        if spec.x_lim is not None:
            ax1.set_xlim(spec.x_lim[0], spec.x_lim[1])
        if spec.y_lim is not None:
            ax1.set_ylim(spec.y_lim[0], spec.y_lim[1])
        if ax2 is not None and spec.y2_lim is not None:
            ax2.set_ylim(spec.y2_lim[0], spec.y2_lim[1])

        if spec.x_ticks is not None:
            ax1.set_xticks([float(v) for v in spec.x_ticks])
        if spec.x_ticklabels is not None:
            ax1.set_xticklabels([str(v) for v in spec.x_ticklabels])

        if spec.y_ticks is not None:
            ax1.set_yticks([float(v) for v in spec.y_ticks])
        if spec.y_ticklabels is not None:
            ax1.set_yticklabels([str(v) for v in spec.y_ticklabels])

        if spec.grid:
            ax1.grid(True, alpha=0.3)

        if spec.aspect == Aspect.EQUAL:
            ax1.set_aspect("equal", adjustable="box")

        _apply_ref_lines(ax1, spec.x_ref, vertical=True)
        _apply_ref_lines(ax1, spec.y_ref, vertical=False)
        if ax2 is not None:
            _apply_ref_lines(ax2, spec.y2_ref, vertical=False)

        ax1.set_title(spec.title or "", fontsize=14)

        leg_fs = spec.legend_fontsize if spec.legend_fontsize is not None else 8
        leg_loc = spec.legend_loc or "best"
        leg_ncol = spec.legend_ncol

        if ax2 is not None:
            all_lines = left_lines + right_lines
            if all_lines:
                labels = [l.get_label() for l in all_lines]
                ax1.legend(all_lines, labels, loc=leg_loc, fontsize=leg_fs, ncol=leg_ncol)
            else:
                ax1.legend(loc=leg_loc, fontsize=leg_fs, ncol=leg_ncol)
        else:
            ax1.legend(loc=leg_loc, fontsize=leg_fs, ncol=leg_ncol)

    else:
        mat = np.asarray(spec.matrix, dtype=float)
        im = ax1.imshow(mat, aspect="auto")
        fig.colorbar(im, ax=ax1)
        ax1.set_title(spec.title or "", fontsize=14)
        ax1.set_xlabel(spec.x_label or "", fontsize=12)
        ax1.set_ylabel(spec.y_label or "", fontsize=12)

        if spec.grid:
            ax1.grid(False)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{spec.figure_id}.png")
    plt.savefig(save_path, dpi=int(spec.dpi), bbox_inches="tight")
    plt.close(fig)
    return save_path