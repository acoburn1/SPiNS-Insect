from scipy.stats import t, ttest_rel
import numpy as np
import warnings
from scipy.stats import pearsonr
from DriverUtils.WarningLog import append_stats_warning

def nanmean_logged(x: np.ndarray, axis=0, source: str = "") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=RuntimeWarning)
        out = np.nanmean(x, axis=axis)

    has_empty_slice = any(
        isinstance(w.message, RuntimeWarning) and "Mean of empty slice" in str(w.message)
        for w in caught
    )
    if has_empty_slice:
        append_stats_warning(
            event="nanmean_empty_slice",
            source=source,
            axis=axis,
            shape=x.shape,
            all_nan=int(np.isnan(x).all()),
            n_non_nan=int(np.sum(~np.isnan(x))),
        )
    return out


def pearsonr_logged(a: np.ndarray, b: np.ndarray, source: str = "") -> tuple[float, float]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r, p = pearsonr(aa, bb)

    warned = any("ConstantInputWarning" in str(w.message.__class__.__name__) or
                 "NearConstantInputWarning" in str(w.message.__class__.__name__) for w in caught)
    if warned:
        append_stats_warning(
            event="pearsonr_constant_input",
            source=source,
            a_shape=aa.shape,
            b_shape=bb.shape,
            a_std=float(np.nanstd(aa)),
            b_std=float(np.nanstd(bb)),
        )
    return float(r), float(p)

def stats_over_models(raw: np.ndarray, ci=0.95):
    """
    raw: (M, E, ...)
    returns dict with arrays of shape (E, ...)
    NaNs ignored.
    """
    x = np.asarray(raw, dtype=np.float64)

    finite = np.isfinite(x)
    n = np.sum(finite, axis=0).astype(np.int64)

    mean = nanmean_logged(x, axis=0, source="Statistics.StatHelper.stats_over_models")

    x0 = x - mean[None, ...]
    x0 = np.where(finite, x0, 0.0)
    ss = np.sum(x0 * x0, axis=0)

    denom = np.maximum(n - 1, 1)
    std = np.sqrt(ss / denom)
    std = np.where(n >= 2, std, 0.0)

    se = np.where(n > 0, std / np.sqrt(np.maximum(n, 1)), np.nan)

    df = np.maximum(n - 1, 1).astype(np.float64)
    tcrit = t.ppf((1 + ci) / 2, df=df)
    tcrit = np.where(n >= 2, tcrit, 0.0)

    ci_lo = mean - tcrit * se
    ci_hi = mean + tcrit * se

    mean = np.where(n > 0, mean, np.nan)
    std = np.where(n > 0, std, np.nan)
    se = np.where(n > 0, se, np.nan)
    ci_lo = np.where(n > 0, ci_lo, np.nan)
    ci_hi = np.where(n > 0, ci_hi, np.nan)

    return dict(mean=mean, std=std, se=se, ci_lo=ci_lo, ci_hi=ci_hi, n=n)

def paired_ttest(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    good = np.isfinite(a) & np.isfinite(b)
    if np.sum(good) < 2:
        return np.nan

    res = ttest_rel(a[good], b[good], nan_policy="omit")
    p = getattr(res, "pvalue", np.nan)
    return float(p) if np.isfinite(p) else np.nan
