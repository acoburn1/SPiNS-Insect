import os
import numpy as np
from DriverUtils.Zarr import load_slice
from Eval.utils import get_ratio_ids


class RatioTestEvaluator:
    name = "RatioTest"

    def run(self, cfg, zarr_path: str, vis=None) -> np.ndarray:
        alt = bool(getattr(cfg, "alt", False))
        ratio_labels, ratio_to_pos = _ratio_positions(alt=alt)
        n_sets = 7

        mod_exemplar_ids = _ids_for_conditions(cfg.probe_index, source="exemplar", category="mod")
        lat_exemplar_ids = _ids_for_conditions(cfg.probe_index, source="exemplar", category="lat")

        ratio_ids_by_label = {r: get_ratio_ids(cfg.probe_index, r) for r in ratio_labels}

        all_ratio_ids = np.concatenate([v for v in ratio_ids_by_label.values() if v.size], axis=0) if ratio_ids_by_label else np.asarray([], dtype=np.int64)

        probe_ids = np.asarray(
            sorted(
                set(mod_exemplar_ids.tolist())
                | set(lat_exemplar_ids.tolist())
                | set(all_ratio_ids.tolist())
            ),
            dtype=np.int64,
        )

        hid, _, _ = load_slice(zarr_path, probe_ids=probe_ids)
        reps = np.asarray(hid, dtype=np.float64)  # (M,E,Psel,H)

        M, E = int(reps.shape[0]), int(reps.shape[1])
        out = np.full((M, E, n_sets), np.nan, dtype=np.float64)

        mod_local = _to_local(probe_ids, mod_exemplar_ids)
        lat_local = _to_local(probe_ids, lat_exemplar_ids)

        ratio_locals = []
        ratio_pos_for_rows = []

        for r in ratio_labels:
            ids = ratio_ids_by_label[r]
            if ids.size == 0:
                continue
            local = _to_local(probe_ids, ids)
            if local.size == 0:
                continue
            pos = ratio_to_pos[r]
            ratio_locals.append(local)
            ratio_pos_for_rows.append(np.full((local.size,), pos, dtype=np.int64))

        if mod_local.size == 0 or lat_local.size == 0 or not ratio_locals:
            return out

        all_ratio_local = np.concatenate(ratio_locals, axis=0).astype(np.int64, copy=False)
        pos_of_trial_row = np.concatenate(ratio_pos_for_rows, axis=0).astype(np.int64, copy=False)

        for m in range(M):
            for e in range(E):
                if vis is not None:
                    vis.update(self.name, m, e)

                mod_ex = reps[m, e, mod_local, :]  # (Em,H)
                lat_ex = reps[m, e, lat_local, :]  # (El,H)
                trials = reps[m, e, all_ratio_local, :]  # (T,H)

                pref = _trial_prefers_mod(trials, mod_ex, lat_ex)  # (T,) bool or empty
                if pref.size == 0:
                    continue

                pref_i = pref.astype(np.int64, copy=False)
                sums = np.bincount(pos_of_trial_row, weights=pref_i, minlength=n_sets).astype(np.float64, copy=False)
                cnts = np.bincount(pos_of_trial_row, minlength=n_sets).astype(np.float64, copy=False)

                with np.errstate(divide="ignore", invalid="ignore"):
                    rates = sums / cnts

                out[m, e, :] = rates

        return out

def _to_local(sorted_probe_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return np.asarray([], dtype=np.int64)

    ids = np.unique(ids)
    idx = np.searchsorted(sorted_probe_ids, ids)
    ok = (idx < sorted_probe_ids.size) & (sorted_probe_ids[idx] == ids)
    return idx[ok].astype(np.int64, copy=False)


def _zscore_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd == 0.0, np.nan, sd)
    return (X - mu) / sd


def _trial_prefers_mod(trials: np.ndarray, mod_exemplars: np.ndarray, lat_exemplars: np.ndarray) -> np.ndarray:
    T = np.asarray(trials, dtype=np.float64)
    M = np.asarray(mod_exemplars, dtype=np.float64)
    L = np.asarray(lat_exemplars, dtype=np.float64)

    if T.shape[0] == 0 or M.shape[0] == 0 or L.shape[0] == 0:
        return np.asarray([], dtype=bool)

    zT = _zscore_rows(T)
    zM = _zscore_rows(M)
    zL = _zscore_rows(L)

    H = float(T.shape[1])
    if H <= 0:
        return np.asarray([], dtype=bool)

    mod_corrs = (zT @ zM.T) / H  # (T,Em)
    lat_corrs = (zT @ zL.T) / H  # (T,El)

    avg_mod = np.nanmean(mod_corrs, axis=1)
    avg_lat = np.nanmean(lat_corrs, axis=1)

    good = np.isfinite(avg_mod) & np.isfinite(avg_lat)
    out = np.zeros((T.shape[0],), dtype=bool)
    out[good] = avg_mod[good] > avg_lat[good]
    return out


def _ids_for_conditions(probe_index: dict, **conds) -> np.ndarray:
    hit = None
    for k, v in conds.items():
        ids = probe_index.get(f"{k}={v}", [])
        s = set(int(i) for i in ids)
        hit = s if hit is None else (hit & s)
    if not hit:
        return np.asarray([], dtype=np.int64)
    return np.asarray(sorted(hit), dtype=np.int64)


def _ratio_positions(alt: bool):
    if not alt:
        ratios = ["0:6", "1:5", "2:4", "3:3", "4:2", "5:1", "6:0"]
        return ratios, {r: i for i, r in enumerate(ratios)}
    ratios = ["0:5", "1:4", "2:3", "2:2", "3:2", "4:1", "5:0"]
    return ratios, {r: i for i, r in enumerate(ratios)}