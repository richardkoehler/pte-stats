"""Module for functions regarding multiple comparison."""
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.stats.multitest import fdrcorrection

import pte_stats


def timeseries_pvals(
    x: np.ndarray,
    y: int | float | np.ndarray,
    n_perm: int,
    two_tailed: bool,
):
    """Calculate sample-wise p-values for array using permutation testing."""
    p_vals = np.empty(len(x))
    if isinstance(y, (int, float)):
        for t_p, pred in enumerate(x):
            _, p_vals[t_p] = pte_stats.permutation_onesample(
                data_a=pred, data_b=y, n_perm=n_perm, two_tailed=two_tailed
            )
    else:
        for i, (x_, y_) in enumerate(zip(x, y)):
            _, p_vals[i] = pte_stats.permutation_twosample(
                data_a=x_, data_b=y_, n_perm=n_perm, two_tailed=two_tailed
            )
    return p_vals


def correct_pvals(
    p_vals: np.ndarray,
    alpha: float = 0.05,
    correction_method: str = "cluster",
    n_perm: int = 10000,
):
    """Correct p-values for multiple comparisons."""
    if correction_method == "cluster_pvals":
        _, signif = pte_stats.cluster_analysis_from_pvals(
            p_values=p_vals, alpha=alpha, n_perm=n_perm, only_max_cluster=False
        )
        if len(signif) > 0:
            signif = np.hstack(signif)
        else:
            signif = np.array([])
    elif correction_method == "fdr":
        rejected, _ = fdrcorrection(
            pvals=p_vals, alpha=alpha, method="poscorr", is_sorted=False
        )
        signif = np.where(rejected)[0]
    else:
        raise ValueError(
            "`correction_method` must be one of either `cluster_pvals` or"
            f"`fdr`. Got:{correction_method}."
        )
    return signif


def handle_baseline(
    baseline: None | tuple[int | float | None, int | float | None] = None,
    sfreq: int | float | None = None,
) -> tuple[int | None, int | None]:
    """Return baseline start and end indices."""
    if not baseline:
        return None, None
    if any(baseline) and not sfreq:
        raise ValueError(
            "If `baseline` is any value other than `None`, or `(None, None)`,"
            f" `sfreq` must be provided. Got: {baseline=}"
        )
    if not sfreq:
        sfreq = 0.0
    if baseline[0] is None:
        base_start = 0
    else:
        base_start = int(baseline[0] * sfreq)
    if baseline[1] is None:
        base_end = None
    else:
        base_end = int(baseline[1] * sfreq)
    return base_start, base_end


def baseline_correct(
    data: np.ndarray,
    baseline_mode: str = "percent",
    base_start: int | None = None,
    base_end: int | None = None,
) -> np.ndarray:
    """Baseline correct data."""
    baseline = data[::, base_start:base_end]
    if baseline_mode == "percent":
        mean = np.mean(baseline, axis=1, keepdims=True)
        data = (data - mean) / (mean) * 100
        return data
    if baseline_mode == "zscore":
        mean = np.mean(baseline, axis=1, keepdims=True)
        data = (data - mean) / (np.std(baseline, axis=1, keepdims=True))
        return data
    if baseline_mode == "std":
        data /= np.std(baseline, axis=1, keepdims=True)
        return data
    raise ValueError(
        "`baseline_mode` must be one of either `percent`, `std` or `zscore`."
        f" Got: {baseline_mode}."
    )
