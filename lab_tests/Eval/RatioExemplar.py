import numpy as np
from scipy.stats import pearsonr
from Output.OutputSpec import *
from Eval.utils import *
from Eval.Pearson import get_significant_epoch


import numpy as np
from scipy.stats import pearsonr
from Output.OutputSpec import *
from Eval.utils import *
from Eval.Pearson import get_significant_epoch


class SCurveEpochEvaluator:
    name = "SCurve"

    def run(self, cfg, zarr_path: str, save_path: str) -> list[OutputSpec]:
        alt = cfg.alt
        include_e0 = cfg.include_e0

        ratio_labels, ratio_to_pos = _ratio_positions(alt=alt)

        # exemplar ids
        mod_exemplar_ids = _ids_for_conditions(cfg.probe_index, source="exemplar", category="mod")
        lat_exemplar_ids = _ids_for_conditions(cfg.probe_index, source="exemplar", category="lat")

        # onehot ids (needed for significant epoch computation)
        onehot_ids = get_onehot_ids(cfg.probe_index)

        # ratio ids
        need_ratio_ids = []
        for r in ratio_labels:
            need_ratio_ids.extend(cfg.probe_index.get(f"ratio={r}", []))
        need_ratio_ids = np.asarray(sorted(set(need_ratio_ids)), dtype=np.int64)

        # union of all probes we need
        probe_ids = np.asarray(
            sorted(
                set(mod_exemplar_ids.tolist())
                | set(lat_exemplar_ids.tolist())
                | set(onehot_ids.tolist())
                | set(need_ratio_ids.tolist())
            ),
            dtype=np.int64,
        )

        loaded = load_slice(zarr_path, probe_ids=probe_ids)
        hid = loaded[0] if isinstance(loaded, tuple) else loaded
        reps = np.asarray(hid, dtype=np.float64)  # (M,E,Psel,D)

        idx_map = {int(pid): i for i, pid in enumerate(probe_ids.tolist())}

        mod_local = np.asarray([idx_map[int(pid)] for pid in mod_exemplar_ids if int(pid) in idx_map], dtype=np.int64)
        lat_local = np.asarray([idx_map[int(pid)] for pid in lat_exemplar_ids if int(pid) in idx_map], dtype=np.int64)
        onehot_local = np.asarray([idx_map[int(pid)] for pid in onehot_ids if int(pid) in idx_map], dtype=np.int64)

        M, E = int(reps.shape[0]), int(reps.shape[1])

        totals = M
        included = 0
        sig_epochs = []

        acc_by_pos = {i: [] for i in range(7)}

        for m in range(M):
            try:
                if mod_local.size == 0 or lat_local.size == 0 or onehot_local.size == 0:
                    continue

                hidden_onehot = np.take(reps[m], onehot_local, axis=1)  # (E,P,H)
                P_onehot = int(hidden_onehot.shape[1])
                nf = int(P_onehot // 2)

                sig_e = get_significant_epoch(
                    hidden_onehot=hidden_onehot,
                    ref_mod=cfg.mod_rm,
                    ref_lat=cfg.lat_rm,
                    num_features=nf,
                    alpha=getattr(cfg, "alpha", 0.05),
                    diff_threshold=getattr(cfg, "diff_threshold", 0.05),
                )
                if sig_e is None:
                    continue

                sig_e = int(sig_e)
                if sig_e < 0 or sig_e >= E:
                    continue

                mod_exemplars = reps[m, sig_e, mod_local, :]
                lat_exemplars = reps[m, sig_e, lat_local, :]

                if mod_exemplars.shape[0] == 0 or lat_exemplars.shape[0] == 0:
                    continue

                for ratio in ratio_labels:
                    r_ids = get_ratio_ids(cfg.probe_index, ratio)
                    if r_ids.size == 0:
                        continue

                    r_local = np.asarray([idx_map[int(pid)] for pid in r_ids if int(pid) in idx_map], dtype=np.int64)
                    if r_local.size == 0:
                        continue

                    trials = reps[m, sig_e, r_local, :]
                    rate = _mod_pref_rate(trials, mod_exemplars, lat_exemplars)

                    pos = ratio_to_pos[ratio]
                    if np.isfinite(rate):
                        acc_by_pos[pos].append(rate)

                included += 1
                sig_epochs.append(sig_e)

            except Exception:
                continue

        means = []
        stderrs = []
        for pos in range(7):
            vals = np.asarray(acc_by_pos[pos], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                means.append(np.nan)
                stderrs.append(np.nan)
            else:
                means.append(float(np.mean(vals)))
                stderrs.append(float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)

        pct = (100.0 * included / totals) if totals > 0 else 0.0
        ses = np.asarray(sig_epochs, dtype=np.float64)
        if ses.size == 0:
            mean_se = np.nan
            std_se = np.nan
        else:
            mean_se = float(np.mean(ses))
            std_se = float(np.std(ses, ddof=1)) if ses.size > 1 else 0.0

        x = list(range(7))
        label = f"mean ({pct:.1f}%, sig_e={mean_se:.2f}±{std_se:.2f})"

        return [
            OutputSpec(
                figure_id=f"s_curve_h_sig_e",
                title="Modular Preference by Feature Composition (hidden vs exemplars @ sig epoch/model)",
                x_label="# mod feats",
                y_label="% mod-pref (avg corr to mod exemplars > lat exemplars)",
                series_list=[
                    Series(
                        kind=PlotKind.LINE,
                        label=label,
                        x=x,
                        y=means,
                        yerr=stderrs,
                        color=Color.BLUE,
                        linestyle=LineStyle.SOLID,
                    )
                ],
                matrix=None,
            )
        ]


class RatioSetOverEpochsEvaluator:
    name = "RatioSetOverEpochs"

    def run(self, cfg, zarr_path: str, save_path: str) -> list[OutputSpec]:
        return []


def _ids_for_conditions(probe_index: dict, **conds) -> np.ndarray:
    hit = None
    for k, v in conds.items():
        ids = probe_index.get(f"{k}={v}", [])
        s = set(int(i) for i in ids)
        hit = s if hit is None else (hit & s)
    if not hit:
        return np.asarray([], dtype=np.int64)
    return np.asarray(sorted(hit), dtype=np.int64)


def _exemplar_corrs(trials: np.ndarray, exemplars: np.ndarray) -> np.ndarray:
    T = np.asarray(trials, dtype=np.float64)
    E = np.asarray(exemplars, dtype=np.float64)
    n_trials = int(T.shape[0])
    n_exemplar = int(E.shape[0])
    out = np.empty((n_trials, n_exemplar), dtype=np.float64)
    for i in range(n_trials):
        for j in range(n_exemplar):
            r, _ = pearsonr(T[i], E[j])
            out[i, j] = float(r)
    return out


def _mod_pref_rate(trials: np.ndarray, mod_exemplars: np.ndarray, lat_exemplars: np.ndarray) -> float:
    if trials.shape[0] == 0:
        return np.nan
    if mod_exemplars.shape[0] == 0 or lat_exemplars.shape[0] == 0:
        return np.nan

    mod_corrs = _exemplar_corrs(trials, mod_exemplars)
    lat_corrs = _exemplar_corrs(trials, lat_exemplars)
    avg_mod = np.mean(mod_corrs, axis=1)
    avg_lat = np.mean(lat_corrs, axis=1)
    return float(np.mean(avg_mod > avg_lat))


def _ratio_positions(alt: bool):
    if not alt:
        ratios = ["0:6", "1:5", "2:4", "3:3", "4:2", "5:1", "6:0"]
        return ratios, {r: i for i, r in enumerate(ratios)}
    ratios = ["0:5", "1:4", "2:3", "2:2", "3:2", "4:1", "5:0"]
    return ratios, {r: i for i, r in enumerate(ratios)}