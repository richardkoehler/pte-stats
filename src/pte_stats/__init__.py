"""An open-source software package for statistics with time series. """

__version__ = "0.2.0"

from .cluster import (
    cluster_analysis_1d,
    cluster_analysis_2d,
    cluster_analysis_from_pvals,
    clusters_from_pvals,
    get_clusters_1d,
    cluster_correct_pvals_2d
)
from .permutation import (
    permutation_onesample,
    permutation_twosample,
    permutation_1d,
    permutation_1d_onesample,
    permutation_1d_twosample,
    permutation_2d,
    permutation_2d_onesample,
    permutation_2d_twosample,
    spearmans_rho_permutation,
)
from .timeseries import (
    timeseries_pvals,
    correct_pvals,
    baseline_correct,
    handle_baseline,
    handle_baseline_bytimes,
)
