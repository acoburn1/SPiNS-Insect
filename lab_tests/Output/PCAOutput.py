import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, t
from Eval.PCA import get_pcns_mod_lat

def plot_generalization_vs_dimensionality_diff(
    data_dir,
    epoch,
    num_features=11,
    save_dir="Results/Analysis/Plots/scatter_dimensionality",
    show_plots=False,
    include_e0=False
):
    """
    Scatter of generalization (%Mod response) vs Mod–Lat principal dimensionality (k95_mod − k95_lat).
    Each dot = one model at the given epoch.

    X-axis: (k95_mod - k95_lat) from PCA on hidden activations
    Y-axis: %Mod preference from 3:3 ratio tests
    """

    os.makedirs(save_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        print(f"No NPZ files found in {data_dir}")
        return

    test_data_key = 'hidden_ratio_tests'
    activation_key = 'hidden_activations'   # assumed key

    kdiff_per_model = []
    mod_pref_per_model = []

    for f in npz_files:
        d = np.load(f, allow_pickle=True)

        # --- X values: k95_mod - k95_lat ---
        if activation_key not in d or epoch >= len(d[activation_key]):
            continue
        A = np.asarray(d[activation_key][epoch], float)
        if A.shape[0] != 2 * num_features:
            continue
        km, kl = get_pcns_mod_lat(A, num_features)
        kdiff = km - kl

        # --- Y values: generalization from ratio tests ---
        if test_data_key not in d or epoch >= len(d[test_data_key]):
            continue
        test_epoch = d[test_data_key][epoch]
        if '3:3' not in test_epoch:
            continue

        ratio_data = test_epoch['3:3']
        # average %mod preference across all sets
        rates = []
        for set_name, set_data in ratio_data.items():
            mod_corrs = np.asarray(set_data["mod"], float)
            lat_corrs = np.asarray(set_data["lat"], float)
            avg_mod = np.mean(mod_corrs, axis=1)
            avg_lat = np.mean(lat_corrs, axis=1)
            mod_pref = np.mean(avg_mod > avg_lat)
            rates.append(mod_pref)
        if not rates:
            continue
        mod_pref_rate = np.mean(rates)

        kdiff_per_model.append(kdiff)
        mod_pref_per_model.append(mod_pref_rate)

    if not kdiff_per_model:
        print("No valid data at this epoch.")
        return

    # --- Plot ---
    layer = 'Hidden'
    epoch_display = epoch if include_e0 else epoch + 1
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(kdiff_per_model, mod_pref_per_model, s=60, alpha=0.7, zorder=3)

    if len(kdiff_per_model) >= 2:
        r, pval = pearsonr(kdiff_per_model, mod_pref_per_model)
        z = np.polyfit(kdiff_per_model, mod_pref_per_model, 1)
        p = np.poly1d(z)
        xs = np.linspace(min(kdiff_per_model), max(kdiff_per_model), 100)
        ax.plot(xs, p(xs), color='red', linewidth=2, alpha=0.8,
                label=f'r = {r:.3f}, p = {pval:.3f}')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1, label='Chance (50%)')
    ax.set_xlabel("Dimensionality Difference (k95_mod - k95_lat)", fontsize=12)
    ax.set_ylabel("Generalization (%Mod preference)", fontsize=12)
    ax.set_title(f"Generalization vs Dimensionality - {layer} (Epoch {epoch_display})", fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"scatter_dimensionality_kdiff_{layer[0].lower()}e{epoch_display}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()

    return {
        "epoch": epoch,
        "kdiff": kdiff_per_model,
        "generalization": mod_pref_per_model,
        "save_path": out_path
    }


def plot_k95_bars_epoch(
    data_dir,
    epoch,
    num_features=11,
    save_dir="Results/Analysis/Plots/PCA/k95_bars",
    show_plots=False,
    include_e0=False,
    ci=0.95
):
    """
    Compute average k95 for modular and lattice subsets across models at a given epoch
    and plot a bar chart with confidence intervals.

    Returns dict with raw values, means, CI bounds, and save path.
    """
    os.makedirs(save_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        print(f"No NPZ files found in {data_dir}")
        return

    activation_key = 'hidden_activations'  # using hidden activations

    km_list, kl_list = [], []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        if activation_key not in d or epoch >= len(d[activation_key]):
            continue
        A = np.asarray(d[activation_key][epoch], float)
        if A.ndim < 2 or A.shape[0] != 2 * num_features:
            continue
        km, kl = get_pcns_mod_lat(A, num_features)
        km_list.append(float(km))
        kl_list.append(float(kl))

    if not km_list or not kl_list:
        print("No valid activation data to compute k95 at this epoch.")
        return

    def _mean_ci(vals, ci):
        vals = np.asarray(vals, dtype=float)
        n = vals.shape[0]
        m = np.nanmean(vals)
        if n <= 1:
            return m, m, m, 0.0
        sd = np.nanstd(vals, ddof=1)
        se = sd / np.sqrt(n)
        tcrit = t.ppf((1 + ci) / 2.0, df=n - 1)
        half = tcrit * se
        return m, m - half, m + half, half

    km_mean, km_ci_l, km_ci_u, km_err = _mean_ci(km_list, ci)
    kl_mean, kl_ci_l, kl_ci_u, kl_err = _mean_ci(kl_list, ci)

    layer = 'Hidden'
    epoch_display = epoch if include_e0 else epoch + 1

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Mod', 'Lat']
    means = [km_mean, kl_mean]
    yerr = [km_err, kl_err]
    colors = ['tab:blue', 'tab:orange']

    ax.bar(labels, means, yerr=yerr, capsize=6, color=colors, alpha=0.85)
    ax.set_ylabel('k95 (components to reach 95% variance)', fontsize=12)
    ax.set_title(f'Average k95 at Epoch {epoch_display} — {layer}', fontsize=14)
    ax.set_ylim(0, max(num_features, max(means) + 1))
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"k95_bars_{layer[0].lower()}e{epoch_display}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    return {
        'epoch': epoch,
        'km_values': km_list,
        'kl_values': kl_list,
        'km_mean': km_mean,
        'kl_mean': kl_mean,
        'km_ci': (km_ci_l, km_ci_u),
        'kl_ci': (kl_ci_l, kl_ci_u),
        'save_path': out_path
    }


def plot_k95_over_epochs(
    data_dir,
    num_features=11,
    save_dir="Results/Analysis/Plots/PCA/k95_over_epochs",
    show_plots=False,
    include_e0=False,
    ci=0.95
):
    """
    Compute average k95 for modular and lattice subsets across models for each epoch
    and plot two lines (Mod and Lat) with confidence interval bars.

    Returns dict with per-epoch means and CI bounds and save path.
    """
    os.makedirs(save_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        print(f"No NPZ files found in {data_dir}")
        return

    activation_key = 'hidden_activations'  # using hidden activations

    # Determine maximum epochs available across models
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

    def _mean_ci(vals, ci):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        n = vals.shape[0]
        if n == 0:
            return np.nan, np.nan, np.nan, 0.0
        m = np.mean(vals)
        if n == 1:
            return m, m, m, 0.0
        sd = np.std(vals, ddof=1)
        se = sd / np.sqrt(n)
        tcrit = t.ppf((1 + ci) / 2.0, df=n - 1)
        half = tcrit * se
        return m, m - half, m + half, half

    km_means, km_ci_l, km_ci_u, km_errs = [], [], [], []
    kl_means, kl_ci_l, kl_ci_u, kl_errs = [], [], [], []

    for e in range(max_epochs):
        km_vals_e, kl_vals_e = [], []
        for f in npz_files:
            try:
                d = np.load(f, allow_pickle=True)
                if activation_key not in d or e >= len(d[activation_key]):
                    continue
                A = np.asarray(d[activation_key][e], float)
                if A.ndim < 2 or A.shape[0] != 2 * num_features:
                    continue
                km, kl = get_pcns_mod_lat(A, num_features)
                km_vals_e.append(float(km))
                kl_vals_e.append(float(kl))
            except Exception:
                continue

        m, l, u, err = _mean_ci(km_vals_e, ci)
        km_means.append(m); km_ci_l.append(l); km_ci_u.append(u); km_errs.append(err)
        m, l, u, err = _mean_ci(kl_vals_e, ci)
        kl_means.append(m); kl_ci_l.append(l); kl_ci_u.append(u); kl_errs.append(err)

    display_epochs = list(range(max_epochs)) if include_e0 else list(range(1, max_epochs + 1))

    layer = 'Hidden'

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot lines with CI bars using errorbar
    ax.errorbar(display_epochs, km_means, yerr=km_errs, color='tab:blue', marker='o',
                linewidth=2, markersize=4, capsize=4, label='Mod k95')
    ax.errorbar(display_epochs, kl_means, yerr=kl_errs, color='tab:orange', marker='s',
                linewidth=2, markersize=4, capsize=4, label='Lat k95')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('k95 (components to reach 95% variance)', fontsize=12)
    ax.set_title(f'Average k95 over Epochs — {layer}', fontsize=14)

    # y-limit bounded by possible maximum (num_features)
    y_max = np.nanmax([np.nanmax(km_means), np.nanmax(kl_means)])
    ax.set_ylim(0, max(num_features, (y_max if np.isfinite(y_max) else num_features)))

    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()

    suffix = 'h'
    out_path = os.path.join(save_dir, f"k95_over_epochs_{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    return {
        'epochs': list(range(max_epochs)),
        'km_means': km_means,
        'km_ci_lowers': km_ci_l,
        'km_ci_uppers': km_ci_u,
        'kl_means': kl_means,
        'kl_ci_lowers': kl_ci_l,
        'kl_ci_uppers': kl_ci_u,
        'save_path': out_path
    }
