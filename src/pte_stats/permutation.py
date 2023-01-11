"""Module for permutation testing."""
import random

import numpy as np
from numba import njit
import pandas as pd
import scipy.stats


def spearmans_rho_permutation(
    x: np.ndarray |pd.Series, y: np.ndarray |pd.Series, n_perm: int = 5000
) -> tuple[float, float]:
    """Calculate permutation test for multiple repetitions of Spearmans Rho

    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    Parameters
    ----------
    x (np array) : first distibution
    y (np array) : second distribution
    n_permp (int): number of permutations

    Returns
    -------
    gT (float) : estimated ground truth, here spearman's rho
    p_val (float) : p value of permutation test
    """

    # compute ground truth difference
    gT = scipy.stats.spearmanr(x, y)[0]
    #
    pV = np.array((x, y))
    # Initialize permutation:
    pD = []
    # Permutation loop:
    args_order = np.arange(0, pV.shape[1], 1)
    args_order_2 = np.arange(0, pV.shape[1], 1)
    for _ in range(n_perm):
        # Shuffle the data:
        random.shuffle(args_order)
        random.shuffle(args_order_2)
        # Compute permuted absolute difference of your two sampled
        # distributions and store it in pD:
        pD.append(
            scipy.stats.spearmanr(pV[0, args_order], pV[1, args_order_2])[0]
        )

    # calculate p value
    if gT < 0:
        p_val = (len(np.where(pD <= gT)[0]) + 1) / (n_perm + 1)
    else:
        p_val = (len(np.where(pD >= gT)[0]) + 1) / (n_perm + 1)

    return gT, p_val


def permutation_2d(
    data_a: np.ndarray,
    data_b: np.ndarray | int | float,
    n_perm: int = 1000,
    two_tailed: bool = True,
):
    """Perform permutation test with two-dimensional array.

    Parameters
    ----------
    x : array_like
        First distribution
    y : int or float
        Baseline against which to check for statistical significance
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution from baseline
    float
        P-value of permutation test
    """
    if isinstance(data_b, (int, float)):
        data_b = np.ones((1, *data_a.shape[1:])) * data_b
        return permutation_2d_onesample(
            data_a=data_a, data_b=data_b, n_perm=n_perm, two_tailed=two_tailed
        )

    if data_a.shape[1:] != data_b.shape[1:]:
        raise ValueError(
            f"If `data_b` is an array, it must have the same shape as"
            f" `data_a`. Got: {data_a.shape=} and {data_b.shape=}."
        )
    return permutation_2d_twosample(
        data_a=data_a, data_b=data_b, n_perm=n_perm, two_tailed=two_tailed
    )


def permutation_1d(
    data_a: np.ndarray,
    data_b: np.ndarray | int | float,
    n_perm: int = 1000,
    two_tailed: bool = True,
) -> np.ndarray:
    """Perform permutation test with one-dimensional array.

    Parameters
    ----------
    data_a : array_like
        First distribution
    data_b : int or float
        Baseline against which to check for statistical significance
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True

    Returns
    -------
    array
        P-values of permutation tests
    """
    if isinstance(data_b, (int, float)):
        return permutation_1d_onesample(
            data_a=data_a, data_b=data_b, n_perm=n_perm, two_tailed=two_tailed
        )
    return permutation_1d_twosample(
        data_a=data_a, data_b=data_b, n_perm=n_perm, two_tailed=two_tailed
    )


@njit
def permutation_1d_twosample(
    data_a: np.ndarray, data_b: np.ndarray, n_perm: int, two_tailed: bool
) -> np.ndarray:
    """"""
    p_values = np.empty((data_a.shape[1]))

    for i in np.arange(data_a.shape[1]):
        _, p = permutation_twosample(
            data_a[:, i], data_b[:, i], n_perm, two_tailed
        )
        p_values[i] = p
    return p_values


@njit
def permutation_1d_onesample(
    data_a: np.ndarray, data_b: int | float, n_perm: int, two_tailed: bool
) -> np.ndarray:
    """"""
    p_values = np.empty((data_a.shape[1]))

    for i in np.arange(data_a.shape[1]):
        _, p = permutation_onesample(data_a[:, i], data_b, n_perm, two_tailed)
        p_values[i] = p
    return p_values


@njit
def permutation_2d_twosample(
    data_a: np.ndarray, data_b: np.ndarray, n_perm: int, two_tailed: bool
) -> np.ndarray:
    """"""
    p_values = np.empty((data_a.shape[1], data_a.shape[2]))

    for i in np.arange(data_a.shape[1]):
        for j in np.arange(data_a.shape[2]):
            _, p = permutation_twosample(
                data_a[:, i, j], data_b[:, i, j], n_perm, two_tailed
            )
            p_values[i, j] = p
    return p_values


@njit
def permutation_2d_onesample(
    data_a: np.ndarray, data_b: np.ndarray, n_perm: int, two_tailed: bool
) -> np.ndarray:
    """"""
    p_values = np.empty((data_a.shape[1], data_a.shape[2]))

    for i in np.arange(data_a.shape[1]):
        for j in np.arange(data_a.shape[2]):
            _, p = permutation_onesample(
                data_a[:, i, j], data_b[:, i, j], n_perm, two_tailed
            )
            p_values[i, j] = p
    return p_values


@njit
def permutation_onesample(
    data_a: np.ndarray,
    data_b: int | float,
    n_perm: int = 10000,
    two_tailed: bool = True,
) -> tuple[float, float]:
    """Perform permutation test with one-sample distribution.

    Parameters
    ----------
    x : array_like
        First distribution
    y : int, float
        Baseline against which to check for statistical significance
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True. If False, tests if data is significantly larger than
        baseline.

    Returns
    -------
    float
        Estimated difference of distribution from baseline
    float
        P-value of permutation test
    """

    zeroed = data_a - data_b
    p = np.empty(n_perm)
    z = np.mean(zeroed)
    if two_tailed:
        z = np.abs(z)
    # Run the simulation n_perm times
    for i in np.arange(n_perm):
        sign = np.random.choice(
            a=np.array([-1.0, 1.0]), size=len(data_a), replace=True
        )
        val_perm = np.mean(zeroed * sign)
        if two_tailed:
            val_perm = np.abs(val_perm)
        p[i] = val_perm

    # Compute effect size (Cohen's d)
    std = np.std(zeroed)
    abs_diff = np.abs(z)
    if std == 0.0:
        effect_size = 0.0
        if abs_diff != 0.0:
            print(
                "Warning: Pooled standard deviation of distributions was 0."
                " Cannot correctly calculate effect size (Cohen's d)."
            )
    else:
        effect_size = np.round(abs_diff / std, 3)

    return effect_size, (np.sum(p >= z) + 1) / (n_perm + 1)


@njit
def permutation_twosample(
    data_a: np.ndarray,
    data_b: np.ndarray,
    n_perm: int = 10000,
    two_tailed: bool = True,
) -> tuple[float, float]:
    """Perform permutation test.

    Parameters
    ----------
    x : array_like
        First distribution
    y : array_like
        Second distribution
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution means
    float
        P-value of permutation test
    """
    if two_tailed:
        zeroed = np.abs(np.mean(data_a) - np.mean(data_b))
        data = np.concatenate((data_a, data_b), axis=0)
        half = int(len(data) / 2)
        p = np.empty(n_perm)
        for i in np.arange(0, n_perm):
            np.random.shuffle(data)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.abs(np.mean(data[:half]) - np.mean(data[half:]))
    else:
        zeroed = np.mean(data_a) - np.mean(data_b)
        data = np.concatenate((data_a, data_b), axis=0)
        half = int(len(data) / 2)
        p = np.empty(n_perm)
        for i in np.arange(0, n_perm):
            np.random.shuffle(data)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.mean(data[:half]) - np.mean(data[half:])

    # Compute effect size (Cohen's d)
    n_a = data_a.size
    n_b = data_b.size
    pooled_std = np.sqrt(
        (
            ((n_a - 1) * np.square(np.std(data_a)))
            + ((n_b - 1) * np.square(np.std(data_b)))
        )
        / (n_a + n_b - 2)
    )
    effect_size = np.abs(np.mean(data_a) - np.mean(data_b)) / pooled_std
    effect_size = np.round(effect_size, 3)
    return effect_size, (np.sum(p >= zeroed) + 1) / (n_perm + 1)
