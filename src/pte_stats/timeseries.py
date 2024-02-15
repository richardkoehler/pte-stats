"""Module for statistical analysis of time series."""

import numpy as np

import pte_stats


def timeseries_pvals(
    x: np.ndarray,
    y: int | float | np.ndarray,
    n_perm: int,
    two_tailed: bool,
) -> np.ndarray:
    """Calculate sample-wise p-values for array using permutation testing."""
    p_vals = np.empty(len(x))
    if isinstance(y, int | float):
        for i, x_ in enumerate(x):
            _, p_vals[i] = pte_stats.permutation_onesample(
                data_a=x_, data_b=y, n_perm=n_perm, two_tailed=two_tailed
            )
    else:
        for i, (x_, y_) in enumerate(zip(x, y, strict=True)):
            _, p_vals[i] = pte_stats.permutation_twosample(
                data_a=x_, data_b=y_, n_perm=n_perm, two_tailed=two_tailed
            )
    return p_vals


def handle_baseline(
    baseline: None | tuple[int | float | None, int | float | None] = None,
    sfreq: int | float | None = None,
) -> tuple[int | None, int | None]:
    """Return baseline start and end indices."""
    if baseline is None:
        return None, None
    if any(baseline) and sfreq is None:
        raise ValueError(
            "If `baseline` is any value other than `None`, or `(None, None)`,"
            f" `sfreq` must be provided. Got: {baseline=}"
        )
    if sfreq is None:
        sfreq = 0.0
    base_start = 0 if baseline[0] is None else int(baseline[0] * sfreq)
    base_end = None if baseline[1] is None else int(baseline[1] * sfreq)
    return base_start, base_end


def handle_baseline_bytimes(
    baseline: None | tuple[int | float | None, int | float | None] = None,
    times: np.ndarray | None = None,
) -> tuple[int | None, int | None]:
    """Return baseline start and end indices."""
    if baseline is None:
        return None, None
    if any(baseline) and times is None:
        raise ValueError(
            "If `baseline` is any value other than `None`, or `(None, None)`,"
            f" `times` must be provided. Got: {baseline = }"
        )
    baseline_start, baseline_end = baseline[0], baseline[1]
    if baseline_start is None:
        ind_start = 0
    else:
        ind_start = np.where(baseline_start <= times)[0][0]  # type: ignore
    if baseline[1] is None:
        ind_end = None
    else:
        ind_end = np.where(times <= baseline_end)[0][-1]  # type: ignore
    return ind_start, ind_end


def baseline_correct(
    data: np.ndarray,
    baseline_mode: str = "percent",
    base_start: int | None = None,
    base_end: int | None = None,
    baseline_trialwise: bool = False,
) -> np.ndarray:
    """Baseline correct data."""
    axis: int | tuple[int, int] = (-2, -1)
    if baseline_trialwise:
        axis = -1

    baseline = data[::, base_start:base_end]

    if baseline_mode == "percent":
        mean = np.mean(baseline, axis=axis, keepdims=True)
        data = (data - mean) / (mean) * 100
        return data
    if baseline_mode == "zscore":
        mean = np.mean(baseline, axis=axis, keepdims=True)
        data = (data - mean) / (np.std(baseline, axis=1, keepdims=True))
        return data
    if baseline_mode == "std":
        data /= np.std(baseline, axis=axis, keepdims=True)
        return data
    raise ValueError(
        "`baseline_mode` must be one of either `percent`, `std` or `zscore`."
        f" Got: {baseline_mode}."
    )
