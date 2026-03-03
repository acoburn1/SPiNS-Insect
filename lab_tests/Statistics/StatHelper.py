from scipy.stats import t
import numpy as np

def stats_over_models(raw: np.ndarray, ci=0.95):
    """
    raw: (M, E, ...)
    returns dict with arrays of shape (E, ...)
    NaNs ignored.
    """
    x = np.asarray(raw, dtype=np.float64)

    finite = np.isfinite(x)
    n = np.sum(finite, axis=0).astype(np.int64)

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(x, axis=0)

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