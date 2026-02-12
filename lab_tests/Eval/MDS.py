import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from scipy.stats import spearmanr


def distance_matrix(A_rows, metric="cosine"):
    A = np.asarray(A_rows, float)
    return pairwise_distances(A, metric=metric)


def mds_from_distance(D, n_components=3, random_state=0, n_init=8, max_iter=300):
    D = np.asarray(D, float)
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter
    )
    Y = mds.fit_transform(D)
    return Y, float(mds.stress_)


def embedding_fidelity_rho_p(D_orig, Y_emb):
    D_orig = np.asarray(D_orig, float)
    Y = np.asarray(Y_emb, float)
    D_emb = pairwise_distances(Y, metric="euclidean")
    iu = np.triu_indices_from(D_orig, k=1)
    rho, p = spearmanr(D_orig[iu], D_emb[iu])
    return float(rho), float(p)


def mds_average_embedding(
    nA_rows,
    stats_producer,
    n_components=3,
    metric="cosine",
    random_state=0,
    n_init=8,
    max_iter=300
):
    if nA_rows is None or len(nA_rows) == 0:
        raise ValueError("nA_rows is empty")

    Ds = [distance_matrix(A_rows, metric=metric) for A_rows in nA_rows]
    D_mean, D_lo, D_hi, D_half = stats_producer.mean_ci_matrices(Ds)

    Y, stress = mds_from_distance(
        D_mean,
        n_components=n_components,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter
    )

    rho, p = embedding_fidelity_rho_p(D_mean, Y)

    return {
        "Y": Y,
        "stress": float(stress),
        "rho": float(rho),
        "p": float(p),
        "D_mean": D_mean,
        "D_ci_lo": D_lo,
        "D_ci_hi": D_hi,
        "D_ci_half": D_half,
        "n_models": int(len(nA_rows))
    }
