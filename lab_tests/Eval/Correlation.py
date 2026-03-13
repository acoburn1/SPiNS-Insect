from sys import hash_info
from scipy.stats import pearsonr
import numpy as np
from Eval.utils import *
from DriverUtils.Zarr import load_slice
from DriverUtils.Visual import EvalVisualInfo


class CorrelationEvaluator:
    name = "Correlation"

    def run(self, cfg, zarr_path: str, vis: EvalVisualInfo = None) -> np.ndarray:
        """
          C1=0 hid, C1=1 out
          C2=0 mod, C2=1 lat
          D=0 r,   D=1 p
        """
        onehot_ids = get_onehot_ids(cfg.probe_index)
        hid, out, _ = load_slice(zarr_path, probe_ids=onehot_ids)
        hid = np.asarray(hid, dtype=np.float64)  # (M,E,P,H)
        out = np.asarray(out, dtype=np.float64)  # (M,E,P,O)    

        M, E, P, H = map(int, hid.shape)
        nf = cfg.num_features
        assert_data_shape([M, E, P], [cfg.num_models, cfg.eval_epochs, nf*2], ["M", "E", "P"])

        res = np.full((M, E, 2, 2, 2), np.nan, dtype=np.float64)  # C1=[hid,out], C2=[mod,lat], D=[r,p]

        for m in range(M):
            for e in range(E):
                if vis is not None:
                    vis.update(m, e)

                H = hid[m, e]  # (P,H)
                O = out[m, e]  # (P,O)

                h_mod_vecs = H[:nf]
                h_lat_vecs = H[nf:]
                o_mod_vecs = O[:nf]
                o_lat_vecs = O[nf:]

                h_mod_cm = _pairwise_corr_matrix(h_mod_vecs)
                h_lat_cm = _pairwise_corr_matrix(h_lat_vecs)

                h_r_m, h_p_m = _flat_corr_with_p(h_mod_cm, cfg.mod_rm)
                h_r_l, h_p_l = _flat_corr_with_p(h_lat_cm, cfg.lat_rm)
                o_r_m, o_p_m = _flat_corr_with_p(o_mod_vecs, cfg.mod_rm)
                o_r_l, o_p_l = _flat_corr_with_p(o_lat_vecs, cfg.lat_rm)

                res[m, e, 0, 0, 0] = h_r_m
                res[m, e, 0, 0, 1] = h_p_m
                res[m, e, 0, 1, 0] = h_r_l
                res[m, e, 0, 1, 1] = h_p_l
                res[m, e, 1, 0, 0] = o_r_m
                res[m, e, 1, 0, 1] = o_p_m
                res[m, e, 1, 1, 0] = o_r_l
                res[m, e, 1, 1, 1] = o_p_l

        return res, None

class MatrixCorrelationEvaluator:
    name = "MatrixCorrelation"
    def run(self, cfg, zarr_path: str, vis: EvalVisualInfo = None) -> np.ndarray:
        """
          C1=0 hid, C1=1 out
          C2=0 mod, C2=1 lat
          D=(nf, nf) pairwise correlation matrix of hidden onehot activations
        """
        onehot_ids = get_onehot_ids(cfg.probe_index)
        hid, out, _ = load_slice(zarr_path, probe_ids=onehot_ids)
        hid = np.asarray(hid, dtype=np.float64)  # (M,E,P,H)
        out = np.asarray(out, dtype=np.float64)  # (M,E,P,O)

        M, E, P, H = map(int, hid.shape)
        _, _, _, O = map(int, out.shape)
        nf = cfg.num_features
        assert_data_shape([M, E, P], [cfg.num_models, cfg.eval_epochs, nf*2], ["M", "E", "P"])

        res = np.full((M, E, 2, 2, nf, nf), np.nan, dtype=np.float64)  # C1=[hid,out], C2=[mod,lat], D=(nf, nf)

        for m in range(M):
            for e in range(E):
                if vis is not None:
                    vis.update(m, e)
                H = hid[m, e]  # (P,H)
                O = out[m, e]  # (P,O)
                h_mod_vecs = H[:nf]
                h_lat_vecs = H[nf:]
                o_mod_vecs = O[:nf]
                o_lat_vecs = O[nf:]
                res[m, e, 0, 0, :, :] = _pairwise_corr_matrix(h_mod_vecs)
                res[m, e, 0, 1, :, :] = _pairwise_corr_matrix(h_lat_vecs)
                res[m, e, 1, 0, :, :] = o_mod_vecs
                res[m, e, 1, 1, :, :] = o_lat_vecs

        return res, None

class LossEvaluator:
    name = "Loss"
    def run(self, cfg, zarr_path: str, vis: EvalVisualInfo = None) -> np.ndarray:
        _, _, losses = load_slice(zarr_path)
        return np.asarray(losses, dtype=np.float64), None  # (M, E)

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