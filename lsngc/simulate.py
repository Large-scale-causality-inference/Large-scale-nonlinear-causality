"""Bundled paper data and deterministic nonlinear regression-test systems."""

from __future__ import annotations

from importlib import resources
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.io.matlab import MatReadError


def _read_mat(path: str | PathLike[str]) -> dict[str, Any]:
    """Give truncated files the same readable error across SciPy versions."""
    try:
        return loadmat(str(path))
    except (IndexError, MatReadError) as error:
        # Older SciPy indexes a short header without checking its length.
        raise ValueError("MAT file appears truncated or invalid") from error


def generate_nonlinear_system(
    n_nodes: int,
    n_samples: int,
    seed: int = 0,
    burn_in: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a nonlinear VAR and its source-row, target-column graph.

    The recurrence is

    ``x_j(t) = 0.45*x_j(t-1) + 0.25*tanh(x_(j-1)(t-1))``
    ``         + 0.12*I[j even]*tanh(x_(j-3)(t-1)) + epsilon_j(t)``,

    with cyclic indices and independent ``N(0, 0.2**2)`` innovations. The
    first ``burn_in`` samples are discarded. Each invocation starts its own
    NumPy ``RandomState(seed)``. The returned series has shape
    ``(n_nodes, n_samples)`` and the adjacency has shape ``(n_nodes, n_nodes)``.
    Self-dependence is omitted from the adjacency diagonal.

    This system was introduced for modernization regression tests; it is not
    one of the paper's published benchmark generators. Its implementation
    matches the independent pre-refactor golden generator exactly.
    """
    if n_nodes < 4 or n_samples < 1 or burn_in < 0:
        raise ValueError("Require n_nodes >= 4, n_samples >= 1, burn_in >= 0")
    rng = np.random.RandomState(seed)
    series = np.zeros((n_nodes, n_samples + burn_in), dtype=np.float64)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    targets = np.arange(n_nodes)
    ring_sources = (targets - 1) % n_nodes
    skip_sources = (targets - 3) % n_nodes
    even_targets = targets % 2 == 0
    adjacency[ring_sources, targets] = 1.0
    adjacency[skip_sources[even_targets], targets[even_targets]] = 1.0
    for t in range(1, series.shape[1]):
        previous = series[:, t - 1]
        series[:, t] = (
            0.45 * previous
            + 0.25 * np.tanh(previous[ring_sources])
            + 0.12 * even_targets * np.tanh(previous[skip_sources])
            + rng.normal(0.0, 0.2, n_nodes)
        )
    return series[:, burn_in:].copy(), adjacency


def load_logistic3(
    realization: int = 0,
    n_samples: int | None = None,
    path: str | PathLike[str] | None = None,
) -> tuple[NDArray, NDArray]:
    """Load a realization of the paper's three-node logistic benchmark.

    This is the three-node benchmark described in the paper's Results,
    "Simulated data network models" ("Complex system with three nodes").
    The stored ``pt_N``
    array contains 100 realizations of 3 nodes and 500 samples each;
    ``Adj`` is the true directed adjacency.
    The original demo selects ``pt_N[realization, :, -n_samples:]``. This
    loader follows that convention, selecting all samples when ``n_samples``
    is omitted. It returns independent arrays with source rows and target
    columns in the adjacency; time series are nodes by samples.

    ``path`` can select an explicit MAT file with the same ``pt_N``/``Adj``
    schema. Otherwise the packaged data are used, with the original repository
    dataset as a fallback when running from a source checkout.
    """
    if path is not None:
        stored = _read_mat(path)
    else:
        bundled = resources.files("lsngc").joinpath("data").joinpath("logistic_3.mat")
        if bundled.is_file():
            with resources.as_file(bundled) as bundled_path:
                stored = _read_mat(bundled_path)
        else:
            repository_path = (
                Path(__file__).resolve().parents[1]
                / "datasets" / "7SYNTHETICS" / "logistic_3.mat"
            )
            if not repository_path.is_file():
                raise FileNotFoundError("Bundled logistic_3.mat is missing; supply path explicitly")
            stored = _read_mat(repository_path)
    if "pt_N" not in stored or "Adj" not in stored:
        raise ValueError("Logistic MAT files must contain pt_N and Adj arrays")
    realizations = np.asarray(stored["pt_N"])
    adjacency = np.asarray(stored["Adj"])
    if realizations.ndim != 3:
        raise ValueError("pt_N must have shape (realizations, nodes, samples)")
    if adjacency.shape != (realizations.shape[1], realizations.shape[1]):
        raise ValueError("Adj must be square with one row per node")
    if not 0 <= realization < realizations.shape[0]:
        raise ValueError("realization must be between 0 and {}".format(realizations.shape[0] - 1))
    sample_count = realizations.shape[2] if n_samples is None else n_samples
    if not 1 <= sample_count <= realizations.shape[2]:
        raise ValueError("n_samples must be between 1 and {}".format(realizations.shape[2]))
    return realizations[realization, :, -sample_count:].copy(), adjacency.copy()
