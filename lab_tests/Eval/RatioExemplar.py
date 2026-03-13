import numpy as np
from DriverUtils.Zarr import load_slice
from Eval.utils import *


class RatioTestEvaluator:
    name = "RatioTest"

    def run(self, cfg, zarr_path: str, vis=None):
        ratio_labels, ratio_to_pos = _ratio_positions(alt=cfg.alt)
        set_labels = _set_axis_labels(cfg.probe_index)

        n_ratios = len(ratio_labels)
        n_sets = len(set_labels)

        mod_exemplar_ids = _ids_for_conditions(
            cfg.probe_index,
            source="exemplar",
            category="mod",
        )
        lat_exemplar_ids = _ids_for_conditions(
            cfg.probe_index,
            source="exemplar",
            category="lat",
        )

        ratio_set_ids = {}
        trial_counts = np.zeros((n_ratios, n_sets), dtype=np.int64)

        for r in ratio_labels:
            r_idx = ratio_to_pos[r]
            for s_idx, s in enumerate(set_labels):
                ids = _ids_for_conditions(
                    cfg.probe_index,
                    source="ratio",
                    ratio=r,
                    sets=s,
                )
                ratio_set_ids[(r, s)] = ids
                trial_counts[r_idx, s_idx] = int(ids.size)

        all_ratio_ids = [ids for ids in ratio_set_ids.values() if ids.size > 0]
        all_ratio_ids = (
            np.concatenate(all_ratio_ids, axis=0)
            if all_ratio_ids
            else np.asarray([], dtype=np.int64)
        )

        probe_ids = np.asarray(
            sorted(
                set(mod_exemplar_ids.tolist())
                | set(lat_exemplar_ids.tolist())
                | set(all_ratio_ids.tolist())
            ),
            dtype=np.int64,
        )

        hid, _, _ = load_slice(zarr_path, probe_ids=probe_ids)
        reps = np.asarray(hid, dtype=np.float64)  # (M, E, Psel, H)

        M, E = int(reps.shape[0]), int(reps.shape[1])
        assert_data_shape([M, E], [cfg.num_models, cfg.eval_epochs], ["M", "E"])

        out = np.full((M, E, n_ratios, n_sets), np.nan, dtype=np.float64)

        mod_local = _to_local(probe_ids, mod_exemplar_ids)
        lat_local = _to_local(probe_ids, lat_exemplar_ids)

        ratio_set_locals = {}
        for r in ratio_labels:
            for s in set_labels:
                ratio_set_locals[(r, s)] = _to_local(probe_ids, ratio_set_ids[(r, s)])

        metadata = {
            "ratio_labels": ratio_labels,
            "set_labels": set_labels,
            "trial_counts": trial_counts,
        }

        if mod_local.size == 0 or lat_local.size == 0:
            return out, metadata

        for m in range(M):
            for e in range(E):
                if vis is not None:
                    vis.update(m, e)

                mod_ex = reps[m, e, mod_local, :]
                lat_ex = reps[m, e, lat_local, :]

                for r in ratio_labels:
                    r_idx = ratio_to_pos[r]

                    for s_idx, s in enumerate(set_labels):
                        local = ratio_set_locals[(r, s)]
                        if local.size == 0:
                            continue

                        trials = reps[m, e, local, :]
                        pref = _trial_prefers_mod(trials, mod_ex, lat_ex)

                        if pref.size == 0:
                            continue

                        out[m, e, r_idx, s_idx] = float(np.mean(pref.astype(np.float64, copy=False)))

        return out, metadata

def _set_axis_labels(probe_index: dict) -> list[str]:
    preferred = [
        "both-core",
        "both-core-whole",
        "both-wrong",
        "lat-core",
        "mod-core",
        "no-core",
    ]

    present = []
    for s in preferred:
        ids = _ids_for_conditions(probe_index, source="ratio", sets=s)
        if ids.size > 0:
            present.append(s)

    extras = []
    for k in probe_index.keys():
        if not k.startswith("sets="):
            continue
        s = k.split("=", 1)[1].strip("'\"")
        if s in present or s in preferred:
            continue
        ids = _ids_for_conditions(probe_index, source="ratio", sets=s)
        if ids.size > 0:
            extras.append(s)

    return present + sorted(extras)


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

    mod_corrs = (zT @ zM.T) / H
    lat_corrs = (zT @ zL.T) / H

    avg_mod = np.nanmean(mod_corrs, axis=1)
    avg_lat = np.nanmean(lat_corrs, axis=1)

    good = np.isfinite(avg_mod) & np.isfinite(avg_lat)
    out = np.zeros((T.shape[0],), dtype=bool)
    out[good] = avg_mod[good] > avg_lat[good]
    return out


def _ids_for_conditions(probe_index: dict, **conds) -> np.ndarray:
    hit = None
    for k, v in conds.items():
        ids = _get_ids_for_value(probe_index, k, v)
        s = set(int(i) for i in ids)
        hit = s if hit is None else (hit & s)

    if not hit:
        return np.asarray([], dtype=np.int64)

    return np.asarray(sorted(hit), dtype=np.int64)


def _get_ids_for_value(probe_index: dict, key: str, value) -> np.ndarray:
    candidates = [f"{key}={value}"]
    if isinstance(value, str):
        candidates.append(f"{key}='{value}'")
        candidates.append(f'{key}="{value}"')

    found = set()
    for cand in candidates:
        ids = probe_index.get(cand, [])
        found.update(int(i) for i in ids)

    if not found:
        return np.asarray([], dtype=np.int64)

    return np.asarray(sorted(found), dtype=np.int64)


def _ratio_positions(alt: bool):
    if not alt:
        ratios = ["0:6", "1:5", "2:4", "3:3", "4:2", "5:1", "6:0"]
        return ratios, {r: i for i, r in enumerate(ratios)}

    ratios = ["0:5", "1:4", "2:3", "2:2", "3:2", "4:1", "5:0"]
    return ratios, {r: i for i, r in enumerate(ratios)}