import numpy as np
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import eigh
from Output.OutputSpec import *
from DriverUtils.Zarr import load_slice
from Eval.utils import *
from Eval.Pearson import get_significant_epoch
from scipy.stats import pearsonr

class K95BarsEpochEvaluator:
    name = "K95BarsEpoch"

    def run(self, cfg, zarr_path: str, save_path: str, epoch: int) -> list[OutputSpec]:
        onehot_ids = get_onehot_ids(cfg.probe_index)
        hid, _, _ = load_slice(zarr_path, probe_ids=onehot_ids)
        hid = np.asarray(hid, dtype=np.float64)

        M, P = int(hid.shape[0]), int(hid.shape[2])
        nf = cfg.num_features

        km = []
        kl = []
        for m in range(M):
            A = hid[m, epoch]
            a, b = get_pcns_mod_lat(A, nf)
            km.append(float(a))
            kl.append(float(b))

        epoch_display = epoch if cfg.include_e0 else epoch + 1

        km_s = StatsObject(np.asarray(km))
        kl_s = StatsObject(np.asarray(kl))

        return [
            OutputSpec(
                figure_id=f"k95_bars_e{epoch_display}",
                title=f"Average k95 at Epoch {epoch_display} — Hidden",
                x_label="Subset",
                y_label="k95 (components to reach 95% variance)",
                series_list=[
                    Series(
                        kind=PlotKind.BAR,
                        label="k95",
                        x=["Mod", "Lat"],
                        y=[km_s.mean, kl_s.mean],
                        yerr=[(km_s.ci_upper - km_s.mean), (kl_s.ci_upper - kl_s.mean)],
                        color=Color.BLUE,
                        linestyle=None,
                    )
                ],
                matrix=None,
            )
        ]


class K95OverEpochsEvaluator:
    name = "K95OverEpochs"

    def run(self, cfg, zarr_path: str, save_path: str) -> list[OutputSpec]:
        onehot_ids = get_onehot_ids(cfg.probe_index)
        hid, _, _ = load_slice(zarr_path, probe_ids=onehot_ids)
        hid = np.asarray(hid, dtype=np.float64)  # (M,E,P,H)

        M, E, P = int(hid.shape[0]), int(hid.shape[1]), int(hid.shape[2])
        nf = cfg.num_features

        km = np.empty((M, E), dtype=np.float64)
        kl = np.empty((M, E), dtype=np.float64)

        for m in range(M):
            for e in range(E):
                A = hid[m, e]  # (P,H)
                km[m, e], kl[m, e] = get_pcns_mod_lat(A, nf)

        km_stats = get_epoch_stats(km)
        kl_stats = get_epoch_stats(kl)

        x = np.arange(E, dtype=np.int64)
        if not cfg.include_e0:
            x = x + 1

        return [
            OutputSpec(
                figure_id="k95_over_epochs",
                title="Average k95 over Epochs — Hidden",
                x_label="Epoch",
                y_label="k95 (components to reach 95% variance)",
                series_list=[
                    Series(
                        kind=PlotKind.LINE,
                        label="Mod k95",
                        x=x.tolist(),
                        y=np.asarray(km_stats.means).tolist(),
                        ci_lower=np.asarray(km_stats.ci_lowers).tolist(),
                        ci_upper=np.asarray(km_stats.ci_uppers).tolist(),
                        color=Color.BLUE,
                        linestyle=LineStyle.SOLID,
                    ),
                    Series(
                        kind=PlotKind.LINE,
                        label="Lat k95",
                        x=x.tolist(),
                        y=np.asarray(kl_stats.means).tolist(),
                        ci_lower=np.asarray(kl_stats.ci_lowers).tolist(),
                        ci_upper=np.asarray(kl_stats.ci_uppers).tolist(),
                        color=Color.ORANGE,
                        linestyle=LineStyle.SOLID,
                    ),
                ],
                matrix=None,
            )
        ]

def build_pca_scatter(
    *,
    cfg,
    root_dir: str,
    which: str,
    alpha: float = 0.05,
    diff_threshold: float = 0.05,
    figure_id: str = "pca_scatter_hls",
):
    which = str(which).lower().strip()
    if which not in ("mod", "lat"):
        raise ValueError("which must be 'mod' or 'lat'")

    zarr_paths = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if dirpath.endswith("activations.zarr"):
            zarr_paths.append(dirpath)
    zarr_paths.sort()

    totals_by_hls = {}
    included_by_hls = {}
    points_by_hls = {}
    sig_epochs_by_hls = {}

    for zp in zarr_paths:
        try:
            hid, _, _ = load_slice(zp, probe_ids=get_onehot_ids(cfg.probe_index))
            M, E, P, H = map(int, hid.shape)
            nf = int(P // 2)
            hls = int(H)

            totals_by_hls[hls] = totals_by_hls.get(hls, 0) + M

            hid_all = np.asarray(hid, dtype=np.float64)
        except Exception:
            continue

        for m in range(M):
            try:
                hidden_onehot = hid_all[m]  # (E,P,H)

                sig_e = get_significant_epoch(
                    hidden_onehot=hidden_onehot,
                    ref_mod=cfg.mod_rm,
                    ref_lat=cfg.lat_rm,
                    num_features=nf,
                    alpha=alpha,
                    diff_threshold=diff_threshold,
                )

                if sig_e is None:
                    continue

                A = hidden_onehot[sig_e]
                km, kl = get_pcns_mod_lat(A, nf)
                y = float(km if which == "mod" else kl)

                included_by_hls[hls] = included_by_hls.get(hls, 0) + 1

                if hls not in points_by_hls:
                    points_by_hls[hls] = {"x": [], "y": []}
                points_by_hls[hls]["x"].append(float(hls))
                points_by_hls[hls]["y"].append(y)

                if hls not in sig_epochs_by_hls:
                    sig_epochs_by_hls[hls] = []
                sig_epochs_by_hls[hls].append(int(sig_e))

            except Exception:
                continue

    color_cycle = [
        Color.BLUE, Color.ORANGE, Color.GREEN, Color.RED, Color.PURPLE,
        Color.BROWN, Color.PINK, Color.GRAY, Color.OLIVE, Color.CYAN
    ]

    series_list = []

    for i, (hls, pts) in enumerate(sorted(points_by_hls.items())):
        total = totals_by_hls.get(hls, 0)
        included = included_by_hls.get(hls, 0)
        pct = (100.0 * included / total) if total > 0 else 0.0

        ses = np.asarray(sig_epochs_by_hls.get(hls, []), dtype=np.float64)
        if ses.size == 0:
            mean_se = np.nan
            std_se = np.nan
        else:
            mean_se = float(np.mean(ses))
            std_se = float(np.std(ses, ddof=1)) if ses.size > 1 else 0.0

        label = f"HLS={hls} ({pct:.1f}%, sig_e={mean_se:.2f}±{std_se:.2f})"

        series_list.append(
            Series(
                kind=PlotKind.SCATTER,
                label=label,
                x=pts["x"],
                y=pts["y"],
                color=color_cycle[i % len(color_cycle)],
                linestyle=None,
            )
        )

    all_x = []
    all_y = []

    for pts in points_by_hls.values():
        all_x.extend(pts["x"])
        all_y.extend(pts["y"])

    all_x = np.asarray(all_x, dtype=np.float64)
    all_y = np.asarray(all_y, dtype=np.float64)

    mask = np.isfinite(all_x) & np.isfinite(all_y)

    if np.sum(mask) >= 2:
        x_fit = all_x[mask]
        y_fit = all_y[mask]

        # linear regression
        slope, intercept = np.polyfit(x_fit, y_fit, 1)

        x_line = np.linspace(np.min(x_fit), np.max(x_fit), 100)
        y_line = slope * x_line + intercept

        # correlation + p-value
        r, p = pearsonr(x_fit, y_fit)

        series_list.append(
            Series(
                kind=PlotKind.LINE,
                label=f"trend (r={r:.2f}, p={p:.2e})",
                x=x_line.tolist(),
                y=y_line.tolist(),
                color=Color.BLACK,
                linestyle=LineStyle.DOTTED,
            )
        )

    return OutputSpec(
        figure_id=figure_id + f"_{which}",
        title=f"PCA vs Hidden Layer Size ({which.upper()})",
        x_label="Hidden Layer Size",
        y_label=f"k95 {which}",
        series_list=series_list,
        matrix=None,
    )


def pca_embedding(A_rows, target=0.95):
    # Center the data
    Xc = A_rows - A_rows.mean(axis=0)
    # Covariance matrix
    C = np.cov(Xc, rowvar=False)
    # Eigendecomposition
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    # Projection
    X_proj = Xc @ evecs
    var_ratio = evals / evals.sum()
    cume = np.cumsum(var_ratio)
    k95 = np.searchsorted(cume, target) + 1
    return X_proj, evals, var_ratio, cume, k95


def get_pcns_mod_lat(activations, num_features):
    _, _, _, _, km = pca_embedding(activations[:num_features])
    _, _, _, _, kl = pca_embedding(activations[num_features:])
    return km, kl
    