import numpy as np
from Output.schema.OutputSpec import *
from Output.utils import (
    first_sig_epochs,
    load_ratio_test_bundle,
    mod_count_from_ratio,
    weighted_ratio_average,
)


class AllModelsSCurveOutput:
    name = "AllModelsSCurve"
    hyperd = False

    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        bundle = load_ratio_test_bundle(analysis_dir)
        raw = bundle["raw"]
        ratio_labels = bundle["ratio_labels"]
        trial_counts = bundle["trial_counts"]

        if not ratio_labels:
            raise ValueError("RatioTest.npz is missing ratio_labels metadata.")
        if trial_counts is None:
            raise ValueError("RatioTest.npz is missing trial_counts metadata.")

        if trial_counts.shape != raw.shape[2:4]:
            raise ValueError(
                f"trial_counts shape {trial_counts.shape} does not match raw ratio/set shape {raw.shape[2:4]}"
            )

        x_vals = np.asarray([mod_count_from_ratio(r) for r in ratio_labels], dtype=np.float64)

        mode = str(sub_cfg.get("epochs", "all")).lower()
        if mode == "all":
            return _build_all_epoch_specs(sub_cfg, raw, x_vals, trial_counts)
        elif mode == "sig":
            return [_build_sig_spec(sub_cfg, analysis_dir, raw, x_vals, trial_counts)]
        else:
            raise ValueError(f"Unsupported epochs mode: {mode}. Expected 'all' or 'sig'.")


def _build_all_epoch_specs(sub_cfg: dict, raw: np.ndarray, x_vals: np.ndarray, trial_counts: np.ndarray) -> list[OutputSpec]:
    M, E, R, S = raw.shape
    specs = []

    for e in range(E):
        per_model_curve = weighted_ratio_average(raw[:, e, :, :], trial_counts)

        series_list = []
        for m in range(M):
            y = per_model_curve[m]
            if m == 0:
                label = "models"
            else:
                label = "_nolabel_"
            series_list.append(
                Series(
                        kind=PlotKind.LINE,
                        label=label,
                        x=[float(v) for v in x_vals],
                        y=[float(v) for v in y],
                        color=Color.BLUE,
                        y_axis=YAxis.LEFT,
                        marker="o",
                        markersize=3.0,
                        linewidth=.8,
                        alpha=.4
                    )
                )

        specs.append(
            OutputSpec(
                figure_id=f"all_models_{'s_curve'}_e{e:03d}",
                title="Modular Response by Feature Composition - Hidden",
                x_label="# mod feats",
                y_label="% mod resp",
                x_lim=[-0.3, 6.3],
                y_lim=[0.0, 1.0],
                legend_loc="legend_loc",
                legend_fontsize="legend_fontsize",
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
                series_list=series_list
            )
        )

    return specs


def _build_sig_spec(sub_cfg: dict, analysis_dir: str, raw: np.ndarray, x_vals: np.ndarray, trial_counts: np.ndarray) -> OutputSpec:
    sig_epochs = first_sig_epochs(analysis_dir, raw.shape[0], raw.shape[1])

    per_model_curves = np.full((raw.shape[0], raw.shape[2]), np.nan, dtype=np.float64)

    for m in range(raw.shape[0]):
        e = sig_epochs[m]
        if not np.isfinite(e):
            continue
        e = int(e)
        per_model_curves[m, :] = weighted_ratio_average(raw[m, e, :, :][None, ...], trial_counts)[0]

    series_list = []
    for m in range(per_model_curves.shape[0]):
        y = per_model_curves[m]
        if m == 0:
            label = "models"
        else:
            label = "_nolabel_"
        series_list.append(
            Series(
                    kind=PlotKind.LINE,
                    label=label,
                    x=[float(v) for v in x_vals],
                    y=[float(v) for v in y],
                    color=Color.BLUE,
                    y_axis=YAxis.LEFT,
                    marker="o",
                    markersize=3.0,
                    linewidth=.8,
                    alpha=.4
                )
            )

    return OutputSpec(
        figure_id="all_models_s_curve_sig",
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
        series_list=series_list
    )

