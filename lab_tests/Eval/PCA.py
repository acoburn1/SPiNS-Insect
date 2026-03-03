import numpy as np
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import eigh
from DriverUtils.Zarr import load_slice
from Eval.utils import *

class K95Evaluator:
    name = "K95"

    def run(self, cfg, zarr_path: str, vis=None) -> np.ndarray:
        """
        Data shape is (M, E, C=2, D=1) with C = [mod, lat]
        """
        onehot_ids = get_onehot_ids(cfg.probe_index)
        hid, _, _ = load_slice(zarr_path, probe_ids=onehot_ids)
        hid = np.asarray(hid, dtype=np.float64)  # (M,E,P,H)

        M, E, P = int(hid.shape[0]), int(hid.shape[1]), int(hid.shape[2])
        nf = cfg.num_features
        assert_data_shape([M, E, P], [cfg.num_models, cfg.eval_epochs, nf*2], ["M", "E", "P"])

        out = np.empty((M, E, 2, 1), dtype=np.int16)

        for m in range(M):
            for e in range(E):
                A = hid[m, e]  # (P,H)
                km, kl = get_pcns_mod_lat(A, nf)
                out[m, e, 0, 0] = int(km)
                out[m, e, 1, 0] = int(kl)
                if vis is not None:
                    vis.update(m, e)

        return out
        

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
    