from scipy.stats import pearsonr
import numpy as np
from Eval.utils import get_onehot_ids
from DriverUtils.Zarr import load_slice
from DriverUtils.Visual import DataVisualInfo


class CorrelationEvaluator:
    name = "Correlation"

    def run(self, cfg, zarr_path: str, vis: DataVisualInfo = None) -> np.ndarray:
        onehot_ids = get_onehot_ids(cfg.probe_index)
        hid, _, _ = load_slice(zarr_path, probe_ids=onehot_ids)
        hid = np.asarray(hid, dtype=np.float64)  # (M,E,P,H)

        M, E, P, H = map(int, hid.shape)
        nf = int(P // 2)

        out = np.full((M, E, 2, 2), np.nan, dtype=np.float64)  # C=[mod,lat], D=[r,p]

        for m in range(M):
            for e in range(E):
                if vis is not None:
                    vis.update(self.name, m, e)

                A = hid[m, e]  # (P,H)

                mod_vecs = A[:nf]
                lat_vecs = A[nf:]

                mod_cm = _pairwise_corr_matrix(mod_vecs)
                lat_cm = _pairwise_corr_matrix(lat_vecs)

                r_m, p_m = _flat_corr_with_p(mod_cm, cfg.mod_rm)
                r_l, p_l = _flat_corr_with_p(lat_cm, cfg.lat_rm)

                out[m, e, 0, 0] = r_m
                out[m, e, 0, 1] = p_m
                out[m, e, 1, 0] = r_l
                out[m, e, 1, 1] = p_l

        return out

def _cut(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    n = int(m.shape[0])
    iu = np.triu_indices(n, k=1)
    return m[iu]

def _flat_corr_with_p(m1: np.ndarray, m2: np.ndarray) -> tuple[float, float]:
    a = _cut(m1)
    b = _cut(m2)
    if a.size < 2 or b.size < 2:
        return (np.nan, np.nan)
    r, p = pearsonr(a, b)
    return (float(r), float(p))

def _pairwise_corr_matrix(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(vectors, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(nrm, eps)
    return X @ X.T