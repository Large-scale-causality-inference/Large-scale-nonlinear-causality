"""Numerically compatible implementation of the published lsNGC code.

References below refer to Wismüller et al., Scientific Reports 11, 7817
(2021), https://doi.org/10.1038/s41598-021-87316-6. The implementation
deliberately retains the original code's arithmetic and center indexing;
see ``state_space_transform`` for its difference from Methods Eq. (5).
Matrices use source rows and target columns; inputs use nodes by time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._clustering import cluster_centers

FloatArray = NDArray[np.floating]


def normalize_0_mean_1_std(inp_series: ArrayLike) -> FloatArray:
    """Normalize each row: z = (x - mean(x)) / sqrt(mean((x-mean(x))²)).

    This is the population standard deviation (ddof=0), preprocessing for
    the Methods state-space reconstruction. Operation order matches the
    original helper, including its handling of constant rows (NaN).
    """
    series = np.array(inp_series, copy=True)
    mean = np.array([series.mean(axis=1)]).T
    centered = series - mean * np.ones((1, series.shape[1]))
    squared = np.power(centered, 2)
    scale = np.sqrt(np.array([squared.sum(axis=1)]).T / squared.shape[1])
    return np.divide(centered, scale * np.ones((1, squared.shape[1])))


def multivariate_split(
    X: ArrayLike, ar_order: int, valid_percent: float = 0
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Build delay vectors and next-step targets from node-by-time ``X``.

    Methods, 'Large-scale nonlinear Granger causality': a delay vector is
    [x(t-d+1), ..., x(t)] and its target is x(t+1). Arrays have shapes
    (samples, d, nodes) and (samples, 1, nodes). The last int(T*valid_percent)
    windows form validation data. The original float32 rounding is retained.
    Flattening these arrays places time before node, as in the legacy code.
    """
    series = np.array(X, copy=True)
    if series.ndim != 2:
        raise ValueError("X must have shape (nodes, time)")
    n_nodes, length = series.shape
    if not isinstance(ar_order, (int, np.integer)) or not 1 <= ar_order < length:
        raise ValueError("ar_order must be an integer between 1 and T-1")
    if not 0 <= valid_percent < 1:
        raise ValueError("valid_percent must be in [0, 1)")
    n_valid = int(valid_percent * length)
    n_train = length - ar_order - n_valid
    if n_train < 0:
        raise ValueError("The validation fraction leaves no training windows")
    x_train = np.zeros((n_train, ar_order, n_nodes), dtype=np.float32)
    y_train = np.zeros((n_train, 1, n_nodes), dtype=np.float32)
    x_valid = np.zeros((n_valid, ar_order, n_nodes), dtype=np.float32)
    y_valid = np.zeros((n_valid, 1, n_nodes), dtype=np.float32)
    for i in range(n_train):
        x_train[i] = series.T[i : i + ar_order]
        y_train[i] = series.T[i + ar_order]
    for j in range(n_valid):
        i = n_train + j
        x_valid[j] = series.T[i : i + ar_order]
        y_valid[j] = series.T[i + ar_order]
    return x_train, y_train, x_valid, y_valid


def calc_f_stat(
    RSS_R: ArrayLike, RSS_U: ArrayLike, n: int, pu: int, pr: int
) -> FloatArray:
    """Return F = ((RSS_R-RSS_U)/RSS_U) * ((n-pu-1)/(pu-pr)).

    Methods Eq. (3), with n=T-d+1, pu=k_f+k_g and pr=k_f. The original
    implementation passes residual variances instead of sums of squares;
    the shared divisor cancels. Negative values are intentionally retained.
    """
    restricted, unrestricted = np.asarray(RSS_R), np.asarray(RSS_U)
    return ((restricted - unrestricted) / unrestricted) * ((n - pu - 1) / (pu - pr))


def f_pvalues(
    f_statistic: ArrayLike, n_samples: int, ar_order: int, k_f: int, k_g: int
) -> FloatArray:
    """Upper-tail probabilities for Methods Eq. (3): F(k_g, T-d-k_f-k_g).

    These are unadjusted probabilities derived from the legacy F statistic,
    not an additional return value of the original lsNGC function. A zero
    diagonal F statistic has p=1. No multiple-testing correction is applied.
    """
    denominator_df = n_samples - ar_order - k_f - k_g
    if denominator_df <= 0:
        raise ValueError("T - ar_order - k_f - k_g must be positive")
    return stats.f.sf(f_statistic, k_g, denominator_df)


def _width(centers: FloatArray) -> float:
    """Legacy width: largest pairwise center distance / sqrt(2*k)."""
    maximum = 0.0
    for first in centers:
        for second in centers:
            distance = np.linalg.norm(first - second)
            if distance > maximum:
                maximum = distance
    return maximum / math.sqrt(2 * len(centers))


def state_space_transform(
    restricted_states: FloatArray,
    source_states: FloatArray,
    global_centers: FloatArray,
    source_centers: FloatArray,
    ar_order: int,
) -> tuple[FloatArray, FloatArray]:
    """Return normalized radial activations (f, g) used by the legacy code.

    Methods Eqs. (4)-(5), nonlinear transformation; Supplement section 2.
    The code's kernels are exp(-distance²/(2*sigma)²), with sigma equal to
    the largest center spacing / sqrt(2*k); each row is divided by its sum.

    Compatibility detail: for f column j, the original deletes global-center
    columns j:j+d, then takes the Frobenius norm of the difference between a
    state and *all* remaining centers. This differs from the individual-center
    distance in Eq. (5). It is preserved, not silently corrected. Only g has
    the original sigma=1 fallback when all source centers coincide.
    """
    k_f, k_g = len(global_centers), len(source_centers)
    sigma = _width(global_centers)
    sigma_g = _width(source_centers)
    if sigma_g == 0:
        sigma_g = 1
    f = np.empty((len(restricted_states), k_f), dtype=float)
    g = np.empty((len(source_states), k_g), dtype=float)
    # Moving this invariant deletion outside the sample loop does not change
    # arithmetic. The golden tests cover every resulting output.
    reduced_centers = [
        np.delete(global_centers, np.arange(j, j + ar_order), axis=1)
        for j in range(k_f)
    ]
    for row, state in enumerate(source_states):
        for j in range(k_g):
            distance = np.linalg.norm(state - source_centers[j])
            g[row, j] = math.exp(-math.pow(distance, 2) / math.pow(2 * sigma_g, 2))
    for row, state in enumerate(restricted_states):
        for j in range(k_f):
            distance = np.linalg.norm(state - reduced_centers[j])
            f[row, j] = math.exp(-math.pow(distance, 2) / math.pow(2 * sigma, 2))
    # Python sum intentionally retains the original summation order.
    return (
        np.array([row / sum(row) for row in f]),
        np.array([row / sum(row) for row in g]),
    )


def regression_variance(design: FloatArray, targets: FloatArray) -> FloatArray:
    """Fit Methods Eqs. (1)-(2) and return each target's residual variance.

    W = pinv(H.T @ H) @ H.T @ Y; residual = H @ W - Y. Variance is the
    diagonal of cov(residual.T), with ddof=1. The original normal-equation
    pseudoinverse and multiplication order are kept instead of replacing
    them with a numerically different least-squares solver.
    """
    gram = np.dot(design.T, design)
    inverse = np.linalg.pinv(gram)
    factor = np.dot(inverse, design.T)
    weights = np.dot(factor, targets)
    error = np.dot(design, weights) - np.array(targets)
    return np.diag(np.cov(error.T))


@dataclass(frozen=True)
class CausalityResult:
    """Legacy matrices and derived significance, with source rows/target columns."""

    affinity: FloatArray
    f_statistic: FloatArray
    pvalues: FloatArray
    restricted_variance: FloatArray
    unrestricted_variance: FloatArray
    global_centers: FloatArray
    source_centers: FloatArray


def analyze(
    inp_series: ArrayLike,
    ar_order: int = 1,
    k_f: int = 3,
    k_g: int = 2,
    normalize: bool = True,
    *,
    seed: int = 123,
) -> CausalityResult:
    """Infer directed dependence and expose the intermediate variance matrices.

    Implements the Methods lsNGC pipeline, Eqs. (1)-(5), using the released
    code's conventions documented above. ``inp_series`` is (nodes, time).
    Affinity = max(log(var_restricted / var_unrestricted), 0); its diagonal
    and the F-statistic diagonal are zero. The explicit KMeans seed defaults
    to the original 123. The graph is conditioned on the other input series.
    """
    series = np.asarray(inp_series)
    if series.ndim != 2 or series.shape[0] < 2:
        raise ValueError("inp_series must have shape (nodes >= 2, time)")
    if not np.issubdtype(series.dtype, np.number) or np.iscomplexobj(series):
        raise ValueError("inp_series must contain real numbers")
    if not np.isfinite(series).all():
        raise ValueError("inp_series must contain finite values")
    if not isinstance(ar_order, (int, np.integer)) or ar_order < 1:
        raise ValueError("ar_order must be a positive integer")
    if not isinstance(k_f, (int, np.integer)) or k_f < 2:
        raise ValueError("k_f must be an integer >= 2 for a nonzero global width")
    if not isinstance(k_g, (int, np.integer)) or k_g < 1:
        raise ValueError("k_g must be a positive integer")
    if k_f > (series.shape[0] - 1) * ar_order + 1:
        raise ValueError("k_f exceeds the original center-indexing bounds for these dimensions")
    if series.shape[1] - ar_order - k_f - k_g <= 0:
        raise ValueError("T - ar_order - k_f - k_g must be positive")
    if normalize and np.any(np.ptp(series, axis=1) == 0):
        raise ValueError("Cannot normalize constant time series")
    normalized = normalize_0_mean_1_std(series) if normalize else series.copy()
    states, targets, _, _ = multivariate_split(normalized, ar_order)
    states = states.reshape(len(states), -1)
    targets = targets.reshape(len(targets), -1)
    centers = cluster_centers(states, k_f, seed)
    if _width(centers) == 0:
        raise ValueError("Global cluster centers coincide; the radial width is zero")
    n_nodes = series.shape[0]
    restricted = np.zeros((n_nodes, n_nodes))
    unrestricted = np.zeros((n_nodes, n_nodes))
    all_source_centers = []
    for source in range(n_nodes):
        rest, _, _, _ = multivariate_split(np.delete(normalized, [source], axis=0), ar_order)
        own, _, _, _ = multivariate_split(np.array([normalized[source]]), ar_order)
        rest, own = rest.reshape(len(rest), -1), own.reshape(len(own), -1)
        own_centers = cluster_centers(own, k_g, seed)
        all_source_centers.append(own_centers)
        f, g = state_space_transform(rest, own, centers, own_centers, ar_order)
        unrestricted[source] = regression_variance(np.concatenate((0.5 * f, 0.5 * g), axis=1), targets)
        restricted[source] = regression_variance(f, targets)
    affinity = np.log(np.divide(restricted, unrestricted))
    affinity = (affinity > 0) * affinity
    np.fill_diagonal(affinity, 0)
    f_statistic = calc_f_stat(restricted, unrestricted, len(states) + 1, k_f + k_g, k_f)
    np.fill_diagonal(f_statistic, 0)
    return CausalityResult(
        affinity, f_statistic,
        f_pvalues(f_statistic, series.shape[1], ar_order, k_f, k_g),
        restricted, unrestricted, centers, np.stack(all_source_centers),
    )


def lsngc(
    inp_series: ArrayLike, ar_order: int = 1, k_f: int = 3, k_g: int = 2,
    normalize: bool = True, *, seed: int = 123,
) -> tuple[FloatArray, FloatArray]:
    """Return (affinity, F statistic), preserving the original lsNGC API.

    See :func:`analyze` for equations, orientation and additional diagnostics.
    """
    result = analyze(inp_series, ar_order, k_f, k_g, normalize, seed=seed)
    return result.affinity, result.f_statistic


lsNGC = lsngc
