from scipy.stats import pearsonr
import torch
import numpy as np
from Output.OutputSpec import *
from Eval.utils import *

class SeriesCorrelationEvaluator:
    name = "SeriesCorrelation"
    def run(self, cfg, zarr_path: str, save_path: str, within: bool=False) -> list[OutputSpec]:
        ref_mod = cfg.mod_rm
        ref_lat = cfg.lat_rm
        if ref_mod is None or ref_lat is None:
            raise ValueError("SeriesCorrelationEvaluator requires ref_mod and ref_lat on cfg (or dict keys).")

        ref_mod = np.asarray(ref_mod, dtype=np.float64)
        ref_lat = np.asarray(ref_lat, dtype=np.float64)

        onehot_ids = get_onehot_ids(cfg.probe_index)
        hid, out, loss = load_slice(zarr_path, probe_ids=onehot_ids)
        hid = np.asarray(hid, dtype=np.float64)  # (M,E,P,H)
        out = np.asarray(out, dtype=np.float64)  # (M,E,P,O)
        loss = np.asarray(loss, dtype=np.float64)  # (M,E)

        M, E, P = int(out.shape[0]), int(out.shape[1]), int(out.shape[2])
        nf = cfg.num_features

        m_out = np.empty((M, E), dtype=np.float64)
        l_out = np.empty((M, E), dtype=np.float64)
        m_hid = np.empty((M, E), dtype=np.float64)
        l_hid = np.empty((M, E), dtype=np.float64)

        for m in range(M):
            for e in range(E):
                out_mat = out[m, e, :, :]
                m_out[m, e] = _flat_corr(out_mat[:nf, :nf], ref_mod)
                l_out[m, e] = _flat_corr(out_mat[nf:2 * nf, nf:2 * nf], ref_lat)

                hid_corr = _pairwise_corr_matrix(hid[m, e, :, :])
                m_hid[m, e] = _flat_corr(hid_corr[:nf, :nf], ref_mod)
                l_hid[m, e] = _flat_corr(hid_corr[nf:2 * nf, nf:2 * nf], ref_lat)

        m_out_s = get_epoch_stats(m_out)
        l_out_s = get_epoch_stats(l_out)
        m_hid_s = get_epoch_stats(m_hid)
        l_hid_s = get_epoch_stats(l_hid)
        loss_s = get_epoch_stats(loss)

        x = np.arange(E, dtype=np.int64)
        if not cfg.include_e0:
            x = x + 1

        return [
            OutputSpec(
                figure_id="series_correlations",
                title="Means Across Epochs",
                x_label="Epoch",
                y_label="Correlation Value",
                series_list=[
                    Series(PlotKind.LINE, "Mod Output Corr", x.tolist(), np.asarray(m_out_s.means).tolist(),
                           ci_lower=np.asarray(m_out_s.ci_lowers).tolist(), ci_upper=np.asarray(m_out_s.ci_uppers).tolist(),
                           color=Color.BLUE, linestyle=LineStyle.SOLID),
                    Series(PlotKind.LINE, "Lat Output Corr", x.tolist(), np.asarray(l_out_s.means).tolist(),
                           ci_lower=np.asarray(l_out_s.ci_lowers).tolist(), ci_upper=np.asarray(l_out_s.ci_uppers).tolist(),
                           color=Color.RED, linestyle=LineStyle.SOLID),
                    Series(PlotKind.LINE, "Mod Hidden Corr", x.tolist(), np.asarray(m_hid_s.means).tolist(),
                           ci_lower=np.asarray(m_hid_s.ci_lowers).tolist(), ci_upper=np.asarray(m_hid_s.ci_uppers).tolist(),
                           color=Color.GREEN, linestyle=LineStyle.SOLID),
                    Series(PlotKind.LINE, "Lat Hidden Corr", x.tolist(), np.asarray(l_hid_s.means).tolist(),
                           ci_lower=np.asarray(l_hid_s.ci_lowers).tolist(), ci_upper=np.asarray(l_hid_s.ci_uppers).tolist(),
                           color=Color.PURPLE, linestyle=LineStyle.SOLID),
                    Series(PlotKind.LINE, "Loss", x.tolist(), np.asarray(loss_s.means).tolist(),
                           ci_lower=np.asarray(loss_s.ci_lowers).tolist(), ci_upper=np.asarray(loss_s.ci_uppers).tolist(),
                           color=Color.GRAY, linestyle=LineStyle.SOLID),
                ],
                matrix=None,
            )
        ]

class MatrixCorrelationEpochEvaluator:
    name = "MatrixCorrelation"

    def run(self, cfg, zarr_path: str, save_path: str, epoch: int) -> list[OutputSpec]:
        include_e0 = cfg.include_e0

        onehot_ids = get_onehot_ids(cfg.probe_index)
        hid, out, _ = load_slice(zarr_path, probe_ids=onehot_ids)
        hid = np.asarray(hid, dtype=np.float64)
        out = np.asarray(out, dtype=np.float64)

        M, E, P = int(out.shape[0]), int(out.shape[1]), int(out.shape[2])
        if epoch < 0:
            epoch = E + epoch
        epoch = max(0, min(epoch, E - 1))

        nf = cfg.num_features

        mod_out = []
        lat_out = []
        mod_hid = []
        lat_hid = []

        for m in range(M):
            out_mat = out[m, epoch, :, :]
            mod_out.append(out_mat[:nf, :nf])
            lat_out.append(out_mat[nf:2 * nf, nf:2 * nf])

            hid_corr = _pairwise_corr_matrix(hid[m, epoch, :, :])
            mod_hid.append(hid_corr[:nf, :nf])
            lat_hid.append(hid_corr[nf:2 * nf, nf:2 * nf])

        mod_out_mean = np.mean(np.stack(mod_out, axis=0), axis=0)
        lat_out_mean = np.mean(np.stack(lat_out, axis=0), axis=0)
        mod_hid_mean = np.mean(np.stack(mod_hid, axis=0), axis=0)
        lat_hid_mean = np.mean(np.stack(lat_hid, axis=0), axis=0)

        epoch_display = epoch if include_e0 else epoch + 1

        return [
            OutputSpec(
                figure_id=f"heatmap_output_mod_e{epoch_display}",
                title=f"Output Similarity Heatmap — Mod (Epoch {epoch_display})",
                x_label="Feature",
                y_label="Feature",
                series_list=None,
                matrix=mod_out_mean.tolist(),
            ),
            OutputSpec(
                figure_id=f"heatmap_output_lat_e{epoch_display}",
                title=f"Output Similarity Heatmap — Lat (Epoch {epoch_display})",
                x_label="Feature",
                y_label="Feature",
                series_list=None,
                matrix=lat_out_mean.tolist(),
            ),
            OutputSpec(
                figure_id=f"heatmap_hidden_mod_e{epoch_display}",
                title=f"Hidden Similarity Heatmap — Mod (Epoch {epoch_display})",
                x_label="Feature",
                y_label="Feature",
                series_list=None,
                matrix=mod_hid_mean.tolist(),
            ),
            OutputSpec(
                figure_id=f"heatmap_hidden_lat_e{epoch_display}",
                title=f"Hidden Similarity Heatmap — Lat (Epoch {epoch_display})",
                x_label="Feature",
                y_label="Feature",
                series_list=None,
                matrix=lat_hid_mean.tolist(),
            ),
        ]

def _cut(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    n = int(m.shape[0])
    iu = np.triu_indices(n, k=1)
    return m[iu]


def _flat_corr(m1: np.ndarray, m2: np.ndarray) -> float:
    a = _cut(m1)
    b = _cut(m2)
    if a.size < 2 or b.size < 2:
        return np.nan
    r, _ = pearsonr(a, b)
    return float(r)


def _pairwise_corr_matrix(vectors: np.ndarray) -> np.ndarray:
    X = np.asarray(vectors, dtype=np.float64)
    n = int(X.shape[0])
    out = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            r, _ = pearsonr(X[i], X[j])
            out[i, j] = float(r)
            out[j, i] = float(r)
    return out

def _flat_corr_with_p(m1: np.ndarray, m2: np.ndarray) -> tuple[float, float]:
    a = _cut(m1)
    b = _cut(m2)
    if a.size < 2 or b.size < 2:
        return (np.nan, np.nan)
    r, p = pearsonr(a, b)
    return (float(r), float(p))

def get_significant_epoch(
    *,
    hidden_onehot: np.ndarray,
    ref_mod: np.ndarray,
    ref_lat: np.ndarray,
    num_features: int,
    alpha: float = 0.05,
    diff_threshold: float = 0.05,
) -> int | None:
    """
    hidden_onehot: (E, 2F, H) hidden activations for one model (onehot probes only)
    ref_mod/ref_lat: (F, F) reference matrices (triangle semantics implied by _cut)
    """
    hidden_onehot = np.asarray(hidden_onehot)
    if hidden_onehot.ndim != 3:
        raise ValueError(f"hidden_onehot must be (E,P,H), got {hidden_onehot.shape}")

    E, P, _ = hidden_onehot.shape
    nf = int(num_features)
    if P != 2 * nf:
        raise ValueError(f"Expected P==2*num_features, got P={P}, num_features={nf}")

    ref_mod = np.asarray(ref_mod, dtype=np.float64)
    ref_lat = np.asarray(ref_lat, dtype=np.float64)
    if ref_mod.shape != (nf, nf) or ref_lat.shape != (nf, nf):
        raise ValueError(f"ref matrices must be ({nf},{nf}), got {ref_mod.shape} and {ref_lat.shape}")

    for e in range(E):
        C = _pairwise_corr_matrix(hidden_onehot[e])  # (2F,2F)

        mod_block = C[:nf, :nf]
        lat_block = C[nf:2 * nf, nf:2 * nf]

        r_m, p_m = _flat_corr_with_p(mod_block, ref_mod)
        r_l, p_l = _flat_corr_with_p(lat_block, ref_lat)

        if not (np.isfinite(r_m) and np.isfinite(r_l) and np.isfinite(p_m) and np.isfinite(p_l)):
            continue
        if p_m < alpha and p_l < alpha and abs(r_m - r_l) < diff_threshold:
            return int(e)

    return None