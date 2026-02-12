import os, glob
import numpy as np
import matplotlib.pyplot as plt
from Eval.MDS import mds_average_embedding
from Statistics.StatsProducer import StatsProducer


def _plot_22_embedding(Y, n_components=3, title="", save_path=None, show=False, stats_text=None):
    group_color = np.array(["mod"] * 11 + ["lat"] * 11)

    shape_group = np.empty(22, dtype=int)
    shape_group[0:3] = 0
    shape_group[3:11] = 1
    shape_group[11:14] = 0
    shape_group[14:22] = 1

    color_map = {"mod": "tab:blue", "lat": "tab:orange"}
    marker_map = {0: "o", 1: "s"}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d") if n_components == 3 else fig.add_subplot(111)

    for gi in ["mod", "lat"]:
        for si in [0, 1]:
            idx = np.where((group_color == gi) & (shape_group == si))[0]
            if idx.size == 0:
                continue
            if n_components == 3:
                ax.scatter(Y[idx, 0], Y[idx, 1], Y[idx, 2], c=color_map[gi], marker=marker_map[si], s=80, alpha=0.85)
            else:
                ax.scatter(Y[idx, 0], Y[idx, 1], c=color_map[gi], marker=marker_map[si], s=80, alpha=0.85)

    if n_components == 3:
        ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2"); ax.set_zlabel("MDS-3")
    else:
        ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if stats_text is not None:
        ax.text2D(0.02, 0.02, stats_text, transform=ax.transAxes)

    if n_components != 3:
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0],[0], marker='o', color='w', label='Mod', markerfacecolor=color_map["mod"], markersize=10),
            Line2D([0],[0], marker='o', color='w', label='Lat', markerfacecolor=color_map["lat"], markersize=10),
            Line2D([0],[0], marker=marker_map[0], color='k', label='First 3 in group', linestyle='None', markersize=10),
            Line2D([0],[0], marker=marker_map[1], color='k', label='Remaining 8 in group', linestyle='None', markersize=10),
        ]
        ax.legend(handles=legend_elems, loc="best")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def _plot_distance_ci_heatmap(D_mean, D_half, title="", save_path=None, show=False):
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(D_mean, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Stimulus index (0-21)")
    ax.set_ylabel("Stimulus index (0-21)")
    ax.grid(False)

    if D_half is not None:
        iu = np.triu_indices_from(D_half, k=1)
        txt = f"Mean CI half-width (avg i<j): {float(np.mean(D_half[iu])):.4f}"
        ax.text(0.02, -0.08, txt, transform=ax.transAxes)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_onehot_mds(
    data_dir,
    epoch=None,
    n_components=3,
    metric="cosine",
    activation_key="hidden_activations",
    save_dir="Results/Analysis/Plots/MDS/onehot",
    show_plots=False,
    include_e0=False,
    random_state=0,
    n_init=8,
    max_iter=300,
    ci=0.95,
    plot_ci_heatmap=True
):
    os.makedirs(save_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        print(f"No NPZ files found in {data_dir}")
        return

    def _load_A(fpath, e):
        d = np.load(fpath, allow_pickle=True)
        if activation_key not in d:
            return None
        arr = d[activation_key]
        if e >= len(arr):
            return None
        A = np.asarray(arr[e], float)
        if A.shape != (22, 400):
            return None
        return A

    max_epochs = 0
    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            if activation_key in d:
                max_epochs = max(max_epochs, len(d[activation_key]))
        except Exception:
            continue
    if max_epochs == 0:
        print("No activation data found.")
        return

    epochs = [epoch] if epoch is not None else list(range(max_epochs))
    stats_producer = StatsProducer(ci=ci)

    results = []
    for e in epochs:
        mats = []
        for f in npz_files:
            A = _load_A(f, e)
            if A is None:
                continue
            mats.append(A)

        if len(mats) == 0:
            continue

        out = mds_average_embedding(
            nA_rows=mats,
            stats_producer=stats_producer,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter
        )

        epoch_display = e if include_e0 else e + 1
        out_path = os.path.join(save_dir, f"mds_onehot_e{epoch_display}.png")
        title = f"MDS one-hot (avg {metric} distance) — Epoch {epoch_display} — models={out['n_models']}"
        stats = f"ρ={out['rho']:.3f}, p={out['p']:.2g}, stress={out['stress']:.3g}"

        _plot_22_embedding(
            out["Y"],
            n_components=n_components,
            title=title,
            save_path=out_path,
            show=show_plots,
            stats_text=stats
        )

        heat_path = None
        if plot_ci_heatmap:
            heat_path = os.path.join(save_dir, f"mds_onehot_dist_mean_ci_e{epoch_display}.png")
            _plot_distance_ci_heatmap(
                out["D_mean"],
                out["D_ci_half"],
                title=f"Avg {metric} distance matrix — Epoch {epoch_display}",
                save_path=heat_path,
                show=show_plots
            )

        results.append({
            "epoch": e,
            "models_used": out["n_models"],
            "stress": out["stress"],
            "rho": out["rho"],
            "p": out["p"],
            "save_path": out_path,
            "dist_ci_heatmap_path": heat_path
        })

    if not results:
        print("No valid data to plot.")
        return

    return results
