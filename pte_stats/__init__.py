"""An open-source software package for statistics with time series. """

__version__ = "0.1.0"

from .cluster import (
    cluster_2d,
    clusters_from_pvals,
    clusterwise_pval_numba,
    get_clusters,
)
from .permutation import (
    permutation_2d,
    permutation_onesample,
    permutation_twosample,
)
from .timeseries import correct_pvals, timeseries_pvals
