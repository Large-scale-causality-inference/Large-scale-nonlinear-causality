"""Command-line interface for inferring an lsNGC affinity matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.io.matlab import MatReadError

from .simulate import load_logistic3


def _load_input(path: Path, realization: int, n_samples: int | None) -> NDArray:
    """Read node-by-sample NPY data or the bundled logistic MAT schema."""
    if path.suffix.lower() == ".mat":
        return load_logistic3(realization=realization, n_samples=n_samples, path=path)[0]
    if path.suffix.lower() != ".npy":
        raise ValueError("Input must be a .npy array or a logistic-schema .mat file")
    if realization != 0:
        raise ValueError("--realization applies only to MAT files")
    series = np.load(path, allow_pickle=False)
    if series.ndim != 2:
        raise ValueError("Input data must have shape (nodes, samples)")
    if n_samples is not None:
        if not 1 <= n_samples <= series.shape[1]:
            raise ValueError("--samples must be between 1 and the input sample count")
        series = series[:, -n_samples:]
    return series


def main(argv: Sequence[str] | None = None) -> int:
    """Infer affinities and optionally save F statistics, p-values, and diagnostics."""
    parser = argparse.ArgumentParser(prog="lsngc", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Infer directed dependence from node-by-sample data")
    run.add_argument("data", type=Path, help="Input .npy, or logistic-schema .mat")
    run.add_argument("--order", type=int, default=1, help="Autoregressive order (default: 1)")
    run.add_argument("--kf", type=int, default=3, help="Restricted-basis centers (default: 3)")
    run.add_argument("--kg", type=int, default=2, help="Source-basis centers (default: 2)")
    run.add_argument("--seed", type=int, default=123, help="KMeans seed (default: 123)")
    run.add_argument("--no-normalize", action="store_true", help="Use input without normalization")
    run.add_argument("--out", type=Path, required=True, help="Output affinity .npy file")
    run.add_argument("--statistics", type=Path, help="Optional .npz with statistics and diagnostics")
    run.add_argument("--realization", type=int, default=0, help="MAT realization (default: 0)")
    run.add_argument("--samples", type=int, help="Use only this many final samples")
    arguments = parser.parse_args(argv)
    try:
        if arguments.out.suffix.lower() != ".npy":
            raise ValueError("--out must have a .npy extension")
        if arguments.statistics is not None and arguments.statistics.suffix.lower() != ".npz":
            raise ValueError("--statistics must have a .npz extension")
        if arguments.out.resolve() == arguments.data.resolve() or (
            arguments.out.exists() and arguments.data.exists()
            and arguments.out.samefile(arguments.data)
        ):
            raise ValueError("--out must differ from the input data path")
        series = _load_input(arguments.data, arguments.realization, arguments.samples)
        from .core import analyze
        result = analyze(
            series,
            ar_order=arguments.order,
            k_f=arguments.kf,
            k_g=arguments.kg,
            normalize=not arguments.no_normalize,
            seed=arguments.seed,
        )
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(arguments.out, result.affinity, allow_pickle=False)
        if arguments.statistics is not None:
            arguments.statistics.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                arguments.statistics,
                affinity=result.affinity,
                f_statistic=result.f_statistic,
                pvalues=result.pvalues,
                restricted_variance=result.restricted_variance,
                unrestricted_variance=result.unrestricted_variance,
                global_centers=result.global_centers,
                source_centers=result.source_centers,
                ar_order=arguments.order,
                k_f=arguments.kf,
                k_g=arguments.kg,
                normalize=not arguments.no_normalize,
                seed=arguments.seed,
            )
    except (OSError, ValueError, TypeError, MatReadError, NotImplementedError) as error:
        parser.exit(2, "lsngc: error: {}\n".format(error))
    print("Saved {} x {} affinity matrix to {}".format(*result.affinity.shape, arguments.out))
    if arguments.statistics is not None:
        print("Saved F statistics, p-values, and diagnostics to {}".format(arguments.statistics))
    return 0
