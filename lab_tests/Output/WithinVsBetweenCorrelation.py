import os
import copy
import numpy as np
from Output.schema.OutputSpec import *
from Output.utils import load_mean_ci
from Statistics.StatHelper import stats_over_models


class WithinVsBetweenCorrelationOutput:
    name = "WithinVsBetweenCorrelation"
    hyperd = False

    def generate_output(self, spec_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        data = np.load(os.path.join(analysis_dir, "MatrixCorrelation.npz"))
        raw = np.asarray(data["raw"], dtype=np.float64)  # (M, E, 2, 2, 11, 11)

        if raw.ndim != 6 or raw.shape[2:4] != (2, 2) or raw.shape[-2:] != (11, 11):
            raise ValueError(f"Expected MatrixCorrelation raw shape (M, E, 2, 2, 11, 11), got {raw.shape}")

        loss_mean, loss_lo, loss_hi = load_mean_ci(os.path.join(analysis_dir, "Loss.npz"))

        # build per-model / per-epoch structure scores
        # out shape: (M, E, 2, 2)
        # C1=0 hid, C1=1 out
        # C2=0 mod, C2=1 lat
        scores = np.full(raw.shape[:4], np.nan, dtype=np.float64)

        for src in range(2):
            for m in range(raw.shape[0]):
                for e in range(raw.shape[1]):
                    scores[m, e, src, 0] = _mod_score(raw[m, e, src, 0])
                    scores[m, e, src, 1] = _lat_score(raw[m, e, src, 1])

        st = stats_over_models(scores)
        mean = st["mean"]   # (E, 2, 2)
        lo = st["ci_lo"]
        hi = st["ci_hi"]

        n_epochs = mean.shape[0]
        epochs = np.arange(1, n_epochs + 1, dtype=np.float64)

        spec_h = OutputSpec(
            figure_id=spec_cfg.get("name", "within_vs_between_hidden"),
            title=spec_cfg.get("title", "Within vs Between Across Epochs"),
            x_label="Epoch",
            y_label="Within - Between",
            y2_label="Loss",
            y_lim=[0,1],
            legend_loc="upper right",
            legend_fontsize=10,
            figsize=(12, 8),
            dpi=300,
            series_list=[
                _make_line(
                    label="M Hidden Structure",
                    x=epochs,
                    y=mean[:, 0, 0],
                    lo=lo[:, 0, 0],
                    hi=hi[:, 0, 0],
                    color=Color.GREEN,
                    y_axis=YAxis.LEFT,
                ),
                _make_line(
                    label="L Hidden Structure",
                    x=epochs,
                    y=mean[:, 0, 1],
                    lo=lo[:, 0, 1],
                    hi=hi[:, 0, 1],
                    color=Color.PINK,
                    y_axis=YAxis.LEFT,
                ),
                _make_line(
                    label="Losses",
                    x=epochs,
                    y=loss_mean,
                    lo=loss_lo,
                    hi=loss_hi,
                    color=Color.BLACK,
                    y_axis=YAxis.RIGHT,
                ),
            ],
        )

        spec_all = copy.deepcopy(spec_h)
        spec_all.figure_id = "within_vs_between"
        spec_all.series_list.extend([
            _make_line(
                label="M Output Structure",
                x=epochs,
                y=mean[:, 1, 0],
                lo=lo[:, 1, 0],
                hi=hi[:, 1, 0],
                color=Color.BLUE,
                y_axis=YAxis.LEFT,
            ),
            _make_line(
                label="L Output Structure",
                x=epochs,
                y=mean[:, 1, 1],
                lo=lo[:, 1, 1],
                hi=hi[:, 1, 1],
                color=Color.RED,
                y_axis=YAxis.LEFT,
            ),
        ])

        return [spec_h, spec_all]


def _make_line(
    *,
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    color: Color,
    y_axis: YAxis,
) -> Series:
    return Series(
        kind=PlotKind.LINE,
        label=label,
        x=[float(v) for v in np.asarray(x, dtype=np.float64)],
        y=[float(v) for v in np.asarray(y, dtype=np.float64).reshape(-1)],
        ci_lower=[float(v) for v in np.asarray(lo, dtype=np.float64).reshape(-1)],
        ci_upper=[float(v) for v in np.asarray(hi, dtype=np.float64).reshape(-1)],
        color=color,
        y_axis=y_axis,
        marker="o",
        markersize=4.0,
        linewidth=2.0,
    )


def _mod_score(cm: np.ndarray) -> float:
    """
    Uses only the last 8 features:
      A = [3:7], B = [7:11]
    score = mean(within A and within B, excluding diagonals)
          - mean(cross-block A<->B)
    """
    X = np.asarray(cm, dtype=np.float64)
    A = np.arange(3, 7)
    B = np.arange(7, 11)

    aa = X[np.ix_(A, A)]
    bb = X[np.ix_(B, B)]
    ab = X[np.ix_(A, B)]
    ba = X[np.ix_(B, A)]

    aa_vals = aa[~np.eye(len(A), dtype=bool)]
    bb_vals = bb[~np.eye(len(B), dtype=bool)]
    within_vals = np.concatenate([aa_vals, bb_vals])

    between_vals = np.concatenate([ab.reshape(-1), ba.reshape(-1)])

    return float(np.nanmean(within_vals) - np.nanmean(between_vals))


def _lat_score(cm: np.ndarray) -> float:
    """
    Uses only the last 8 features [3:, 3:] arranged as a ring.
    For each row, within = 2 left + 2 right neighbors (wrapping),
    between = the remaining 3 non-self positions.
    Final score = mean_row( mean(within row) - mean(between row) ).
    """
    X = np.asarray(cm, dtype=np.float64)[3:, 3:]  # (8, 8)
    if X.shape != (8, 8):
        raise ValueError(f"Expected 8x8 non-core lattice block, got {X.shape}")

    row_scores = []
    n = 8
    for i in range(n):
        within_idx = [((i - 2) % n), ((i - 1) % n), ((i + 1) % n), ((i + 2) % n)]
        between_idx = [j for j in range(n) if j != i and j not in within_idx]

        within_mean = np.nanmean(X[i, within_idx])
        between_mean = np.nanmean(X[i, between_idx])
        row_scores.append(within_mean - between_mean)

    return float(np.nanmean(row_scores))