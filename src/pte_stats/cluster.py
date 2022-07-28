"""Module for cluster-based statistics."""
import numpy as np
from numba import njit
from skimage import measure

import pte_stats


def cluster_analysis_2d(
    data_a: np.ndarray,
    data_b: np.ndarray | int | float | None = None,
    alpha: float = 0.05,
    n_perm: int = 1000,
    only_max_cluster: bool = False,
    n_jobs: int = 1,
) -> tuple[list, list]:
    """Calculate significant clusters and their corresponding p-values."""
    if data_b is None:
        data_b = 0
    if isinstance(data_b, (int, float)):
        one_sample = True
    else:
        one_sample = False

    two_tailed = True

    # Get 2D clusters
    p_values = pte_stats.permutation_2d(
        data_a=data_a,
        data_b=data_b,
        n_perm=n_perm,
        two_tailed=two_tailed,
    )
    p_values_inv = np.asarray(1 - p_values)

    labels, num_clusters = measure.label(
        p_values <= alpha, return_num=True, connectivity=2
    )  # type: ignore

    null_distr = _null_distribution_2d(
        data_a=data_a,
        data_b=data_b,
        alpha=alpha,
        n_perm=n_perm,
        two_tailed=two_tailed,
        one_sample=one_sample,
        n_jobs=n_jobs,
    )

    p_sum_max = 0  # Is only used if only_max_cluster
    clusters = []
    cluster_pvals = []
    for i in range(num_clusters):
        # Cluster labels start at 1
        index_cluster = np.asarray(labels == i + 1).nonzero()
        p_sum = np.sum(p_values_inv[index_cluster])
        p_val = (n_perm - np.sum(p_sum >= null_distr) + 1) / (n_perm + 1)
        if p_val <= alpha:
            clusters.append(index_cluster)
            cluster_pvals.append(p_val)

        if only_max_cluster:
            if p_sum > p_sum_max:
                clusters.clear()
                clusters.append(index_cluster)
                cluster_pvals = [p_val]
                p_sum_max = p_sum

    return cluster_pvals, clusters


def _null_distribution_2d(
    data_a: np.ndarray,
    data_b: np.ndarray | int | float,
    alpha: float,
    n_perm: int,
    two_tailed: bool,
    one_sample: bool,
    n_jobs: int = 1,
) -> np.ndarray:
    """Calculate null distribution of clusters.

    Parameters
    ----------
    data_a :  np.ndarray
        Data of three dimensions (first dimension is the number of
        measurements), e.g. shape: (n_subjects, n_freqs, n_times)
    alpha_ : float
        Significance level (p-value)
    n_perm_ : int
        No. of random permutations

    Returns
    -------
    null_distribution : np.ndarray
        Null distribution of shape (_n_perm, )
    """
    kwargs = {
        "data_a": data_a,
        "data_b": data_b,
        "alpha": alpha,
        "n_perm": n_perm,
        "two_tailed": two_tailed,
        "one_sample": one_sample,
    }
    if n_jobs in (0, 1):
        null_distr = [_single_p_sum_2d(**kwargs) for _ in range(n_perm)]
    else:
        from joblib import Parallel, delayed

        null_distr = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(_single_p_sum_2d)(**kwargs) for _ in range(n_perm)
        )
    return np.array(null_distr)


def _single_p_sum_2d(
    data_a: np.ndarray,
    data_b: np.ndarray | int | float,
    alpha: float,
    n_perm: int,
    two_tailed: bool,
    one_sample: bool,
) -> float:
    """"""
    if one_sample:
        sign = np.random.choice(
            a=np.array([-1, 1]), size=data_a.shape[0], replace=True
        ).reshape(data_a.shape[0], 1, 1)
        data_a_final = data_a.copy() * sign
    else:
        data_a_final = data_a

    p_values = pte_stats.permutation_2d(
        data_a=data_a_final,
        data_b=data_b,
        n_perm=n_perm,
        two_tailed=two_tailed,
    )
    p_values_inv = np.asarray(1 - p_values)
    labels_, num_clusters = measure.label(
        p_values <= alpha, return_num=True, connectivity=2
    )  # type: ignore

    p_sum_max = 0
    if num_clusters > 0:
        for i in range(num_clusters):
            index_cluster = np.asarray(labels_ == i + 1).nonzero()
            p_sum = np.sum(p_values_inv[index_cluster])
            p_sum_max = max(p_sum, p_sum_max)
    return p_sum_max


def clusters_from_pvals(
    p_vals: np.ndarray,
    alpha: float,
    correction_method: str,
    n_perm: int,
    min_cluster_size: int = 1,
) -> tuple[np.ndarray, int]:
    """Return significant clusters from array of p-values."""
    if np.where(p_vals <= alpha)[0].size > 1:
        p_vals_signif = pte_stats.correct_pvals(
            p_vals=p_vals,
            alpha=alpha,
            correction_method=correction_method,
            n_perm=n_perm,
        )
        if p_vals_signif.size > 1:
            clusters_raw = np.array(
                [1 if i in p_vals_signif else 0 for i, _ in enumerate(p_vals)]
            )
            clusters, cluster_count = get_clusters_1d(
                data=clusters_raw, min_cluster_size=min_cluster_size
            )
            return (clusters, cluster_count)
    return (np.ndarray([]), 0)


@njit
def get_clusters_1d(
    data: np.ndarray, min_cluster_size: int = 1
) -> tuple[np.ndarray, int]:
    """Cluster 1-D array of boolean values.

    Parameters
    ----------
    iterable : array-like of bool
        Array to be clustered.
    min_cluster_size : integer
        Minimum size of clusters to consider. Must be at least 1.

    Returns
    -------
    cluster_labels : np.array
        Array of shape (len(iterable), 1), where each value indicates the
        number of the cluster. Values are 0 if the item does not belong to
        a cluster
    cluster_count : int
        Number of detected cluster. Corresponds to the highest value in
        cluster_labels
    """
    min_cluster_size = max(min_cluster_size, 1)
    cluster_labels = np.zeros_like(data, dtype=np.int32)
    cluster_count = 0
    cluster_len = 0
    for idx, item in enumerate(data):
        if item:
            cluster_len += 1
            cluster_labels[idx] = cluster_count + 1
        else:
            if cluster_len >= min_cluster_size:
                cluster_count += 1
            else:
                cluster_labels[max(0, idx - cluster_len) : idx] = 0
            cluster_len = 0
    if cluster_len >= min_cluster_size:
        cluster_count += 1
    else:
        cluster_labels[min(-1, cluster_len) :] = 0
    return cluster_labels, cluster_count


@njit
def _null_distribution_from_pvals(
    p_values: np.ndarray,
    alpha: float = 0.05,
    n_perm: int = 1000,
    min_cluster_size: int = 1,
) -> np.ndarray:
    """Calculate null distribution of clusters from given p-values.

    Parameters
    ----------
    p_values :  np.ndarray
        Array of p-values
    alpha_ : float
        Significance level (p-value)
    n_perm_ : int
        No. of random permutations

    Returns
    -------
    null_distribution : np.ndarray
        Null distribution of shape (_n_perm, )
    """
    # loop through random permutation cycles
    null_distribution = np.zeros(n_perm)
    for i in range(n_perm):
        r_per = np.random.randint(
            low=0, high=p_values.shape[0], size=p_values.shape[0]
        )
        pvals_perm = p_values[r_per]
        labels_, n_clusters = get_clusters_1d(
            data=pvals_perm <= alpha, min_cluster_size=min_cluster_size
        )

        cluster_ind = {}
        if n_clusters == 0:
            null_distribution[i] = 0
        else:
            p_sum = np.zeros(n_clusters)
            for ind in range(n_clusters):
                cluster_ind[ind] = np.where(labels_ == ind + 1)[0]
                p_sum[ind] = np.sum(
                    np.asarray(1 - pvals_perm)[cluster_ind[ind]]
                )
            null_distribution[i] = np.max(p_sum)
    return null_distribution


@njit
def _null_distribution_1d_onesample(
    data_a: np.ndarray,
    data_b: int | float,
    alpha: float,
    n_perm: int,
    two_tailed: bool,
    min_cluster_size: int,
) -> np.ndarray:
    """Calculate null distribution of clusters.

    Parameters
    ----------
    data_a :  np.ndarray
        Data of three dimensions (first dimension is the number of
        measurements), e.g. shape: (n_subjects, n_freqs, n_times)
    alpha_ : float
        Significance level (p-value)
    n_perm_ : int
        No. of random permutations

    Returns
    -------
    null_distribution : np.ndarray
        Null distribution of shape (_n_perm, )
    """
    null_distr = np.empty((n_perm,))
    for n in range(n_perm):
        sign = np.random.choice(
            a=np.array([-1, 1]), size=data_a.shape[0], replace=True
        ).reshape(data_a.shape[0], 1)
        data_a_final = data_a.copy() * sign

        p_values = pte_stats.permutation_1d_onesample(
            data_a=data_a_final,
            data_b=data_b,
            n_perm=n_perm,
            two_tailed=two_tailed,
        )
        p_values_inv = np.asarray(1 - p_values)

        labels, num_clusters = get_clusters_1d(
            data=p_values <= alpha, min_cluster_size=min_cluster_size
        )

        p_sum_max = 0
        if num_clusters > 0:
            for i in range(num_clusters):
                index_cluster = np.asarray(labels == i + 1).nonzero()
                p_sum = np.sum(p_values_inv[index_cluster])
                p_sum_max = max(p_sum, p_sum_max)
        null_distr[n] = p_sum_max
    return null_distr


@njit
def _null_distribution_1d_twosample(
    data_a: np.ndarray,
    data_b: np.ndarray,
    alpha: float,
    n_perm: int,
    two_tailed: bool,
    min_cluster_size: int,
) -> np.ndarray:
    """Calculate null distribution of clusters.

    Parameters
    ----------
    data_a :  np.ndarray
        Data of three dimensions (first dimension is the number of
        measurements), e.g. shape: (n_subjects, n_freqs, n_times)
    alpha : float
        Significance level (p-value)
    n_perm  : int
        No. of random permutations

    Returns
    -------
    null_distribution : np.ndarray
        Null distribution of shape (_n_perm, )
    """
    null_distr = np.empty((n_perm,))
    for n in range(n_perm):
        p_values = pte_stats.permutation_1d_twosample(
            data_a=data_a,
            data_b=data_b,
            n_perm=n_perm,
            two_tailed=two_tailed,
        )
        p_values_inv = np.asarray(1 - p_values)

        labels, num_clusters = get_clusters_1d(
            data=p_values <= alpha, min_cluster_size=min_cluster_size
        )

        p_sum_max = 0
        if num_clusters > 0:
            for i in range(num_clusters):
                index_cluster = np.asarray(labels == i + 1).nonzero()
                p_sum = np.sum(p_values_inv[index_cluster])
                p_sum_max = max(p_sum, p_sum_max)
        null_distr[n] = p_sum_max
    return null_distr


def cluster_analysis_1d(
    data_a: np.ndarray,
    data_b: np.ndarray | int | float | None = None,
    alpha: float = 0.05,
    n_perm: int = 1000,
    only_max_cluster: bool = False,
    two_tailed: bool = True,
    min_cluster_size: int = 1,
) -> tuple[list, list]:
    """Calculate significant clusters and their corresponding p-values."""
    if data_b is None:
        data_b = 0
    if isinstance(data_b, (int, float)):
        f_perm = pte_stats.permutation_1d_onesample
        f_null_distr = _null_distribution_1d_onesample
    else:
        f_perm = pte_stats.permutation_1d_twosample
        f_null_distr = _null_distribution_1d_twosample

    p_values = f_perm(
        data_a=data_a,
        data_b=data_b,
        n_perm=n_perm,
        two_tailed=two_tailed,
    )

    labels, num_clusters = get_clusters_1d(
        data=p_values <= alpha, min_cluster_size=min_cluster_size
    )

    if num_clusters < 1:
        return [], []

    p_values_inv = np.asarray(1 - p_values)

    null_distr = f_null_distr(
        data_a=data_a,
        data_b=data_b,
        alpha=alpha,
        n_perm=n_perm,
        two_tailed=two_tailed,
        min_cluster_size=min_cluster_size,
    )

    clusters = []
    cluster_pvals = []
    p_sum_max = 0  # Is only used if only_max_cluster
    for i in range(num_clusters):
        # Cluster labels start at 1
        index_cluster = np.asarray(labels == i + 1).nonzero()
        p_sum = np.sum(p_values_inv[index_cluster])
        p_val = (n_perm - np.sum(p_sum >= null_distr) + 1) / (n_perm + 1)
        if p_val <= alpha:
            clusters.append(index_cluster)
            cluster_pvals.append(p_val)

        if only_max_cluster:
            if p_sum > p_sum_max:
                clusters.clear()
                clusters.append(index_cluster)
                cluster_pvals = [p_val]
                p_sum_max = p_sum

    return cluster_pvals, clusters


@njit
def cluster_analysis_from_pvals(
    p_values: np.ndarray,
    alpha: float = 0.05,
    n_perm: int = 1000,
    only_max_cluster: bool = False,
    min_cluster_size: int = 1,
) -> tuple[list, list]:
    """Calculate significant clusters and their corresponding p-values.

    Based on:
    https://github.com/neuromodulation/wjn_toolbox/blob/
    4745557040ad26f3b8498ca5d0c5d5dece2d3ba1/mypcluster.m

    https://garstats.wordpress.com/2018/09/06/cluster/

    Arguments
    ---------
    p_values :  numpy array
        Array of p-values. WARNING: MUST be one-dimensional
    alpha : float
        Significance level
    n_perm : int
        No. of random permutations for building cluster null-distribution
    only_max_cluster : bool, default = False
        Set to True to only return the most significant cluster.

    Returns
    -------
    cluster_pvals : list of float(s)
        List of p-values for each cluster
    clusters : list of numpy array(s)
        List of indices of each significant cluster
    """
    p_values_inv = np.asarray(1 - p_values)

    labels, num_clusters = get_clusters_1d(
        data=p_values <= alpha, min_cluster_size=min_cluster_size
    )

    null_distr = _null_distribution_from_pvals(p_values, alpha, n_perm)
    # Loop through clusters of p_val series or image
    clusters = []
    # Initialize empty list with specific data type for numba to work
    cluster_pvals = [np.float64(x) for x in range(0)]
    max_cluster_sum = 0
    # Cluster labels start at 1
    for cluster_i in range(num_clusters):
        index_cluster = np.where(labels == cluster_i + 1)[0]
        p_cluster_sum = np.sum(p_values_inv[index_cluster])
        p_val = (n_perm - np.sum(p_cluster_sum >= null_distr) + 1) / n_perm

        if p_val <= alpha:
            clusters.append(index_cluster)
            cluster_pvals.append(p_val)

        if only_max_cluster:
            if max_cluster_sum == 0 or p_cluster_sum > max_cluster_sum:
                clusters.clear()
                clusters.append(index_cluster)
                cluster_pvals = [p_val]
                max_cluster_sum = p_cluster_sum

    return cluster_pvals, clusters
