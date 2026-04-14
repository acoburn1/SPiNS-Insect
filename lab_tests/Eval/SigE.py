import numpy as np

class SignificantEpochEvaluator:
    def run(self, corr_np: np.ndarray, diff_threshold: float = 0.05) -> tuple[np.ndarray, dict]:
        """
        corr_np: (M,E,C1,C2,D) from CorrelationEvaluator where
          C1=0 hid, C1=1 out
          C2=0 mod, C2=1 lat
          D=0 r,   D=1 p
        returns: (M,E) boolean mask
        """
        corr_np = np.asarray(corr_np, dtype=np.float64)

        mod_r = corr_np[:, :, 0, 0, 0]  # only using hidden
        mod_p = corr_np[:, :, 0, 0, 1]
        lat_r = corr_np[:, :, 0, 1, 0]
        lat_p = corr_np[:, :, 0, 1, 1]

        ok = (
            np.isfinite(mod_r) & np.isfinite(lat_r) &
            np.isfinite(mod_p) & np.isfinite(lat_p) &
            (mod_p < 0.05) & (lat_p < 0.05) &
            (np.absolute(mod_r - lat_r) <= diff_threshold)
        )

        return ok.astype(np.uint8), None


class WithinVsBetweenSignificantEpochEvaluator:
    def run(
        self,
        wb_corr_np: np.ndarray,
        diff_threshold: float = 0.05,
        min_corr: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        """
        wb_corr_np: (M,E,C1,C2) from WithinVsBetweenCorrelationEvaluator where
          C1=0 hid, C1=1 out
          C2=0 mod, C2=1 lat
        returns: (M,E) boolean mask
        """
        wb_corr_np = np.asarray(wb_corr_np, dtype=np.float64)

        mod_r = wb_corr_np[:, :, 0, 0]  # only using hidden
        lat_r = wb_corr_np[:, :, 0, 1]

        ok = (
            np.isfinite(mod_r) & np.isfinite(lat_r) &
            (mod_r > min_corr) & (lat_r > min_corr) &
            (np.absolute(mod_r - lat_r) <= diff_threshold)
        )

        return ok.astype(np.uint8), None
