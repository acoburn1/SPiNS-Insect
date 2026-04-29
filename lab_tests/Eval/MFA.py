import numpy as np

from DriverUtils.Zarr import load_slice
from Eval.utils import assert_data_shape


class MFA:
    name = "MFA"

    def run(self, cfg, zarr_path: str, vis=None) -> tuple[np.ndarray, dict]:
        meta = cfg.probe_metadata.astype(str)

        trial_ids = np.where(meta["source"].to_numpy() == "missing_feature")[0].astype(np.int64)

        if trial_ids.size == 0:
            M = int(cfg.num_models)
            E = int(cfg.eval_epochs)
            out = np.full((M, E, int(trial_ids.size)), np.nan, dtype=np.float64)
            return out, {"trial_ids": trial_ids}

        _, out_act, _ = load_slice(zarr_path, probe_ids=trial_ids)
        reps = np.asarray(out_act, dtype=np.float64)  # (M, E, T, O)

        M, E = int(reps.shape[0]), int(reps.shape[1])
        assert_data_shape([M, E], [cfg.num_models, cfg.eval_epochs], ["M", "E"])

        trial_meta = meta.iloc[trial_ids]
        correct_idx = trial_meta["option_a_idx"].astype(int).to_numpy()
        incorrect_idx = trial_meta["option_b_idx"].astype(int).to_numpy()

        T = int(trial_ids.size)
        trial_scores = np.zeros((M, E, T), dtype=np.float64)

        for t in range(T):
            c = int(correct_idx[t])
            w = int(incorrect_idx[t])
            trial_scores[:, :, t] = (reps[:, :, t, c] > reps[:, :, t, w]).astype(np.float64, copy=False)

        group_labels = ["mod-core", "mod-per", "lat-core", "lat-per"]
        group_masks = {
            "mod-core": ((trial_meta["category"] == "mod") & (trial_meta["structure"] == "core")).to_numpy(),
            "mod-per": ((trial_meta["category"] == "mod") & (trial_meta["structure"] == "per")).to_numpy(),
            "lat-core": ((trial_meta["category"] == "lat") & (trial_meta["structure"] == "core")).to_numpy(),
            "lat-per": ((trial_meta["category"] == "lat") & (trial_meta["structure"] == "per")).to_numpy(),
        }

        out = np.full((M, E, len(group_labels)), np.nan, dtype=np.float64)
        trial_counts = np.zeros((len(group_labels),), dtype=np.int64)

        for g_idx, g in enumerate(group_labels):
            mask = np.asarray(group_masks[g], dtype=bool)
            trial_counts[g_idx] = int(mask.sum())
            if mask.any():
                out[:, :, g_idx] = np.mean(trial_scores[:, :, mask], axis=2)

        if vis is not None:
            for m in range(M):
                for e in range(E):
                    vis.update(m, e)

        metadata = {
            "trial_ids": trial_ids,
            "group_labels": group_labels,
            "trial_counts": trial_counts,
        }

        return out, metadata
