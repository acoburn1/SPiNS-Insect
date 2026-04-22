import numpy as np
from Output.schema.OutputSpec import *
from Statistics.StatHelper import stats_over_models
from Output.utils import (
    first_sig_epochs,
    load_ratio_test_bundle,
    mod_count_from_ratio,
    resolve_requested_set_indices,
    resolve_epoch_range,
    selected_sets_suffix,
    weighted_ratio_average,
)

SET_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.ORANGE,
    Color.PURPLE,
    Color.BROWN,
    Color.PINK,
    Color.CYAN,
    Color.GRAY,
    Color.OLIVE,
]


class SCurveOutput:
    name = "SCurve"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        bundle = load_ratio_test_bundle(analysis_dir)
        raw = bundle["raw"]
        ratio_labels = bundle["ratio_labels"]
        set_labels = bundle["set_labels"]
        trial_counts = bundle["trial_counts"]

        if not ratio_labels or not set_labels:
            raise ValueError("RatioTest.npz is missing ratio/set labels metadata.")
        if trial_counts is None:
            raise ValueError("RatioTest.npz is missing trial_counts metadata.")

        if trial_counts.shape != raw.shape[2:4]:
            raise ValueError(
                f"trial_counts shape {trial_counts.shape} does not match raw ratio/set shape {raw.shape[2:4]}"
            )

        set_indices = resolve_requested_set_indices(sub_cfg, set_labels)
        selected_set_labels = [set_labels[s_idx] for s_idx in set_indices]
        by_set = bool(sub_cfg.get("by_set", False))
        default_id_suffix = ("_by_set" if by_set else "") + selected_sets_suffix(selected_set_labels, set_labels)

        raw = raw[:, :, :, set_indices]
        trial_counts = trial_counts[:, set_indices]

        x_vals = np.asarray([mod_count_from_ratio(r) for r in ratio_labels], dtype=np.float64)

        mode = str(sub_cfg.get("epochs", "range")).lower()
        if mode == "range":
            epoch_indices = resolve_epoch_range(sub_cfg, raw.shape[1], default_start=0)
            return _build_range_epoch_specs(
                sub_cfg,
                raw,
                x_vals,
                trial_counts,
                epoch_indices,
                selected_set_labels=selected_set_labels,
                by_set=by_set,
                default_id_suffix=default_id_suffix,
            )
        elif mode in ("sig", "wb-sig"):
            return [
                _build_sig_spec(
                    sub_cfg,
                    analysis_dir,
                    raw,
                    x_vals,
                    trial_counts,
                    selected_set_labels=selected_set_labels,
                    by_set=by_set,
                    default_id_suffix=default_id_suffix,
                    mode=mode,
                )
            ]
        else:
            raise ValueError(f"Unsupported epochs mode: {mode}. Expected 'range', 'sig', or 'wb-sig'.")


def _build_range_epoch_specs(
    sub_cfg: dict,
    raw: np.ndarray,
    x_vals: np.ndarray,
    trial_counts: np.ndarray,
    epoch_indices: list[int],
    *,
    selected_set_labels: list[str],
    by_set: bool,
    default_id_suffix: str,
) -> list[OutputSpec]:
    specs = []

    for e in epoch_indices:
        if by_set:
            series_list = []
            for s_idx, set_label in enumerate(selected_set_labels):
                st = stats_over_models(raw[:, e, :, s_idx])
                mean = np.asarray(st["mean"], dtype=np.float64)
                ci_lo = np.asarray(st["ci_lo"], dtype=np.float64)
                ci_hi = np.asarray(st["ci_hi"], dtype=np.float64)
                series_list.append(
                    Series(
                        kind=PlotKind.LINE,
                        label=set_label,
                        x=[float(v) for v in x_vals],
                        y=[float(v) for v in mean],
                        ci_lower=[float(v) for v in ci_lo],
                        ci_upper=[float(v) for v in ci_hi],
                        ci_caps=True,
                        color=SET_COLORS[s_idx % len(SET_COLORS)],
                        y_axis=YAxis.LEFT,
                        marker="o",
                        markersize=5.0,
                        linewidth=2.0,
                    )
                )
        else:
            per_model_curve = weighted_ratio_average(raw[:, e, :, :], trial_counts)
            st = stats_over_models(per_model_curve)
            mean = np.asarray(st["mean"], dtype=np.float64)
            ci_lo = np.asarray(st["ci_lo"], dtype=np.float64)
            ci_hi = np.asarray(st["ci_hi"], dtype=np.float64)
            series_list = [
                Series(
                    kind=PlotKind.LINE,
                    label=sub_cfg.get("line_label", f"epoch {e}"),
                    x=[float(v) for v in x_vals],
                    y=[float(v) for v in mean],
                    ci_lower=[float(v) for v in ci_lo],
                    ci_upper=[float(v) for v in ci_hi],
                    ci_caps=True,
                    color=Color.BLUE,
                    y_axis=YAxis.LEFT,
                    marker="o",
                    markersize=5.0,
                    linewidth=2.0,
                )
            ]

        specs.append(
            OutputSpec(
                figure_id=f"s_curve{default_id_suffix}_e{e:03d}",
                title="Modular Response by Feature Composition - Hidden",
                x_label="# mod feats",
                y_label="% mod resp",
                x_lim=[-0.3, 6.3],
                y_lim=[0.0, 1.0],
                legend_loc=sub_cfg.get("legend_loc"),
                legend_fontsize=sub_cfg.get("legend_fontsize"),
                figsize=(12, 8),
                dpi=300,
                x_ref=[
                    RLine(
                        val=float(sub_cfg.get("mid_x", 3.0)),
                        color=Color.GRAY,
                        linestyle=LineStyle.DASHED,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                ],
                y_ref=[
                    RLine(
                        val=0.5,
                        color=Color.GRAY,
                        linestyle=LineStyle.DASHED,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                ],
                series_list=series_list,
            )
        )

    return specs


def _build_sig_spec(
    sub_cfg: dict,
    analysis_dir: str,
    raw: np.ndarray,
    x_vals: np.ndarray,
    trial_counts: np.ndarray,
    *,
    selected_set_labels: list[str],
    by_set: bool,
    default_id_suffix: str,
    mode: str = "sig",
) -> OutputSpec:
    sig_epochs = first_sig_epochs(analysis_dir, raw.shape[0], raw.shape[1], mode=mode)

    if by_set:
        per_model_set_curves = np.full((raw.shape[0], raw.shape[2], raw.shape[3]), np.nan, dtype=np.float64)
        for m in range(raw.shape[0]):
            e = sig_epochs[m]
            if not np.isfinite(e):
                continue
            per_model_set_curves[m, :, :] = raw[m, int(e), :, :]

        series_list = []
        for s_idx, set_label in enumerate(selected_set_labels):
            st = stats_over_models(per_model_set_curves[:, :, s_idx])
            mean = np.asarray(st["mean"], dtype=np.float64)
            ci_lo = np.asarray(st["ci_lo"], dtype=np.float64)
            ci_hi = np.asarray(st["ci_hi"], dtype=np.float64)
            series_list.append(
                Series(
                    kind=PlotKind.LINE,
                    label=set_label,
                    x=[float(v) for v in x_vals],
                    y=[float(v) for v in mean],
                    ci_lower=[float(v) for v in ci_lo],
                    ci_upper=[float(v) for v in ci_hi],
                    ci_caps=True,
                    color=SET_COLORS[s_idx % len(SET_COLORS)],
                    y_axis=YAxis.LEFT,
                    marker="o",
                    markersize=5.0,
                    linewidth=2.0,
                )
            )
    else:
        per_model_curves = np.full((raw.shape[0], raw.shape[2]), np.nan, dtype=np.float64)
        for m in range(raw.shape[0]):
            e = sig_epochs[m]
            if not np.isfinite(e):
                continue
            e = int(e)
            per_model_curves[m, :] = weighted_ratio_average(raw[m, e, :, :][None, ...], trial_counts)[0]

        st = stats_over_models(per_model_curves)
        mean = np.asarray(st["mean"], dtype=np.float64)
        ci_lo = np.asarray(st["ci_lo"], dtype=np.float64)
        ci_hi = np.asarray(st["ci_hi"], dtype=np.float64)
        series_list = [
            Series(
                kind=PlotKind.LINE,
                label=sub_cfg.get("line_label", "first sig epoch"),
                x=[float(v) for v in x_vals],
                y=[float(v) for v in mean],
                ci_lower=[float(v) for v in ci_lo],
                ci_upper=[float(v) for v in ci_hi],
                ci_caps=True,
                color=Color.BLUE,
                y_axis=YAxis.LEFT,
                marker="o",
                markersize=5.0,
                linewidth=2.0,
            )
        ]

    mode_suffix = "sige" if mode == "sig" else "wb-sige"
    default_name = f"s_curve_{mode_suffix}{default_id_suffix}"
    return OutputSpec(
        figure_id=sub_cfg.get("name", default_name),
        title=sub_cfg.get("title", "Modular Response by Feature Composition - Hidden"),
        x_label="# mod feats",
        y_label="% mod resp",
        x_lim=sub_cfg.get("x_lim", [-0.3, 6.3]),
        y_lim=sub_cfg.get("y_lim", [0.0, 1.0]),
        legend_loc=sub_cfg.get("legend_loc"),
        legend_fontsize=sub_cfg.get("legend_fontsize"),
        figsize=tuple(sub_cfg.get("figsize", (12, 8))),
        dpi=int(sub_cfg.get("dpi", 300)),
        x_ref=[
            RLine(
                val=float(sub_cfg.get("mid_x", 3.0)),
                color=Color.GRAY,
                linestyle=LineStyle.DASHED,
                linewidth=1.5,
                alpha=0.8,
            )
        ],
        y_ref=[
            RLine(
                val=float(sub_cfg.get("mid_y", 0.5)),
                color=Color.GRAY,
                linestyle=LineStyle.DASHED,
                linewidth=1.5,
                alpha=0.8,
            )
        ],
        series_list=series_list,
    )
