#!/usr/bin/env python3
"""Capture the untouched implementation before modernization.

Run this file with the separate legacy Python environment. It deliberately never
imports the modern ``lsngc`` package. The original source files in ``legacy/``
must remain byte-for-byte identical to the source revision listed in metadata.

    .legacy-venv/python.exe tests/golden/make_golden.py
    .legacy-venv/python.exe tests/golden/make_golden.py --verify

``--verify`` recomputes every case and checks exact array bytes, including dtype
and shape. Compressed ZIP timestamps are not part of this reproducibility check.
"""

import argparse
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Avoid thread scheduling affecting the reference BLAS reductions. These must
# be set before importing NumPy, SciPy, scikit-learn or torch.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from scipy.io import loadmat
from scipy.stats import f as f_distribution


ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "legacy"


def generate_nonlinear_system(n_nodes, n_samples, seed=0, burn_in=100):
    """Return a noisy nonlinear VAR and its source-row/target-column graph.

    x_j(t) = 0.45 x_j(t-1) + 0.25 tanh(x_(j-1)(t-1))
             + 0.12 I[j even] tanh(x_(j-3)(t-1)) + epsilon_j(t),

    with cyclic node indices and independent N(0, 0.2**2) innovations. The
    diagonal is excluded from the reported between-node graph. Each call starts
    its own RandomState, making both small systems independently seed-0 inputs.
    This is a test system, not a claimed reproduction of a paper benchmark.
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


def _legacy_modules():
    """Import only the frozen source, including its absolute helper imports."""
    sys.path.insert(0, str(LEGACY))
    names = (
        "normalize_0_mean_1_std", "calc_f_stat", "multivariate_split",
        "recovery_performance", "utils",
    )
    modules = {}
    for name in names:
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != LEGACY.resolve():
            raise RuntimeError("Reference import escaped legacy/: " + name)
        modules[name] = module
    return modules


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_digest(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _capture_legacy(function, data, parameters):
    """Observe returned local variances without editing the original source."""
    captured = {}
    kmeans_centers = []
    previous_profile = sys.getprofile()

    def profile(frame, event, argument):
        if (
            event == "return" and frame.f_code.co_name == "fit"
            and frame.f_back is not None
            and frame.f_back.f_code is function.__code__
        ):
            estimator = frame.f_locals.get("self")
            if estimator is not None and hasattr(estimator, "cluster_centers_"):
                kmeans_centers.append(np.array(estimator.cluster_centers_, copy=True))
        if event == "return" and frame.f_code is function.__code__:
            for name in ("sig", "sig_d"):
                if name in frame.f_locals:
                    captured[name] = np.array(frame.f_locals[name], copy=True)

    sys.setprofile(profile)
    try:
        affinity, f_statistic = function(data, **parameters)
    finally:
        sys.setprofile(previous_profile)
    if set(captured) != {"sig", "sig_d"}:
        raise RuntimeError("The legacy lsNGC return did not expose both variances")
    if len(kmeans_centers) != data.shape[0] + 1:
        raise RuntimeError("Did not observe the global and all source KMeans fits")
    captured["global_centers"] = kmeans_centers[0]
    captured["source_centers"] = np.stack(kmeans_centers[1:])
    return affinity, f_statistic, captured


def _case_arrays(modules, data, adjacency, parameters):
    np.random.seed(0)
    import torch
    torch.manual_seed(0)
    affinity, f_statistic, captured = _capture_legacy(
        modules["utils"].lsNGC, data, parameters
    )
    denominator_df = (
        data.shape[1] - parameters["ar_order"] - parameters["k_f"]
        - parameters["k_g"]
    )
    pvalues = f_distribution.sf(f_statistic, parameters["k_g"], denominator_df)
    recovered_adjacency = (pvalues < 0.05).astype(np.uint8)
    np.fill_diagonal(recovered_adjacency, 0)
    auroc = modules["recovery_performance"].recovery_performance(
        [affinity], [adjacency]
    )
    normalized = modules["normalize_0_mean_1_std"].normalize_0_mean_1_std(data)
    arrays = {
        "input": data,
        "true_adjacency": adjacency,
        "affinity": affinity,
        "f_statistic": f_statistic,
        "pvalues": pvalues,
        "adjacency_p05": recovered_adjacency,
        "auroc": auroc,
        "restricted_variance": captured["sig_d"],
        "unrestricted_variance": captured["sig"],
        "global_centers": captured["global_centers"],
        "source_centers": captured["source_centers"],
        "normalized": normalized,
        "ar_order": np.asarray(parameters["ar_order"], dtype=np.int64),
        "k_f": np.asarray(parameters["k_f"], dtype=np.int64),
        "k_g": np.asarray(parameters["k_g"], dtype=np.int64),
        "normalize": np.asarray(parameters["normalize"], dtype=np.int64),
    }
    split_data = normalized if parameters["normalize"] else data
    for validation_fraction, prefix in ((0.0, "split"), (0.2, "validation_split")):
        split = modules["multivariate_split"].multivariate_split(
            split_data, parameters["ar_order"], validation_fraction
        )
        for name, value in zip(("x_train", "y_train", "x_test", "y_test"), split):
            arrays[prefix + "_" + name] = value.numpy().copy()
    return arrays


def _verify_arrays(path, arrays):
    with np.load(str(path), allow_pickle=False) as expected:
        if set(expected.files) != set(arrays):
            raise AssertionError("Output keys changed: " + str(path))
        for key, actual in arrays.items():
            reference = expected[key]
            if (
                reference.dtype != actual.dtype
                or reference.shape != actual.shape
                or reference.tobytes() != actual.tobytes()
            ):
                raise AssertionError("Nonidentical legacy output: {}:{}".format(path, key))


def _runtime_metadata():
    from importlib import metadata
    versions = {}
    for distribution in (
        "numpy", "scipy", "torch", "scikit-learn", "sklearn", "matplotlib",
        "networkx", "joblib", "threadpoolctl",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "versions": versions,
        "thread_environment": {
            key: os.environ[key]
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "torch_default_dtype": str(importlib.import_module("torch").get_default_dtype()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify", action="store_true", help="Compare all arrays byte-for-byte")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    modules = _legacy_modules()
    import torch
    torch.set_num_threads(1)

    dataset_path = ROOT / "datasets" / "7SYNTHETICS" / "logistic_3.mat"
    bundled = loadmat(str(dataset_path))
    inputs = [
        ("logistic3", np.array(bundled["pt_N"][0, :, -200:], copy=True),
         np.array(bundled["Adj"], copy=True)),
        ("synthetic5",) + generate_nonlinear_system(5, 240),
        ("synthetic20",) + generate_nonlinear_system(20, 300),
    ]
    settings = [
        {"ar_order": order, "k_f": kf, "k_g": kg, "normalize": 1}
        for order, kf, kg in itertools.product((1, 2), (2, 3), (2, 3))
    ]
    settings.append({"ar_order": 1, "k_f": 2, "k_g": 2, "normalize": 0})
    cases = []
    for name, data, adjacency in inputs:
        for parameters in settings:
            filename = "{}_order{}_kf{}_kg{}_norm{}.npz".format(
                name, parameters["ar_order"], parameters["k_f"],
                parameters["k_g"], parameters["normalize"]
            )
            started = time.monotonic()
            arrays = _case_arrays(modules, data, adjacency, parameters)
            path = args.output / filename
            if args.verify:
                _verify_arrays(path, arrays)
            else:
                np.savez_compressed(str(path), **arrays)
            cases.append({
                "file": filename,
                "system": name,
                "shape": list(data.shape),
                "parameters": parameters,
                "arrays": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": _array_digest(value)}
                    for key, value in sorted(arrays.items())
                },
            })
            print("{} {} ({:.2f}s)".format(
                "Verified" if args.verify else "Saved", filename,
                time.monotonic() - started
            ), flush=True)

    metadata_path = args.output / "metadata.json"
    source_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
        for path in sorted(LEGACY.iterdir()) if path.is_file()
    }
    if args.verify:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["source_sha256"] != source_hashes:
            raise AssertionError("Frozen legacy source hashes have changed")
        if previous["cases"] != cases:
            raise AssertionError("Golden case metadata has changed")
        print("Verified all 27 cases and frozen source hashes.", flush=True)
        return
    try:
        source_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = None
    metadata = {
        "format_version": 1,
        "source_revision": source_revision,
        "runtime": _runtime_metadata(),
        "source_sha256": source_hashes,
        "dataset_sha256": _sha256(dataset_path),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "randomness": {"numpy_global_seed": 0, "torch_seed": 0,
                       "synthetic_randomstate_seed": 0, "original_kmeans_random_state": 123},
        "conventions": {
            "orientation": "source rows, target columns",
            "pvalues": "scipy.stats.f.sf(f_statistic, k_g, T-ar_order-k_f-k_g)",
            "diagonal_pvalue": 1,
            "recovered_adjacency": "pvalues < 0.05; diagonal zero; no multiple-testing correction",
            "auroc": "Original recovery_performance; includes diagonal; float32 result",
            "residuals": "Original np.cov(error.T) diagonal, observed with sys.setprofile on return",
            "centers": "Global KMeans centers and stacked per-source KMeans centers, observed on fit return",
            "normalization": "Original population-standard-deviation normalization",
            "splits": "Original float32 torch.zeros; validation fractions 0 and 0.2",
            "legacy_kmeans_defaults": "Inherited from recorded scikit-learn version; max_iter=100, random_state=123",
            "logistic3": "Bundled pt_N realization 0, final 200 samples; Adj as stored",
            "synthetics": "generate_nonlinear_system; independent seed 0; 100-sample burn-in",
        },
        "cases": cases,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Saved metadata for {} reference cases.".format(len(cases)), flush=True)


if __name__ == "__main__":
    main()
