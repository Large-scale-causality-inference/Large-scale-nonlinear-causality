"""Compare the package to independently captured, unmodified legacy outputs."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from lsngc import analyze, calc_f_stat, lsNGC, multivariate_split, normalize_0_mean_1_std
from lsngc.evaluate import recovery_performance
from lsngc.simulate import generate_nonlinear_system, load_logistic3

GOLDEN = Path(__file__).parent / "golden"
RTOL = 1e-9
ATOL = 1e-12


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("path", sorted(GOLDEN.glob("*.npz")), ids=lambda p: p.stem)
def test_golden(path):
    with np.load(path, allow_pickle=False) as reference:
        data = reference["input"]
        options = {key: int(reference[key]) for key in ("ar_order", "k_f", "k_g", "normalize")}
        result = analyze(data, **options)
        for name in (
            "global_centers", "source_centers", "restricted_variance", "unrestricted_variance",
            "affinity", "f_statistic", "pvalues",
        ):
            assert_close(getattr(result, name), reference[name])
        np.testing.assert_array_equal((result.pvalues < 0.05).astype(np.uint8), reference["adjacency_p05"])
        scores = recovery_performance([result.affinity], [reference["true_adjacency"]])
        assert scores.dtype == np.float32
        np.testing.assert_array_equal(scores, reference["auroc"])
        normalized = normalize_0_mean_1_std(data)
        assert_close(normalized, reference["normalized"])
        split_input = normalized if options["normalize"] else data
        for fraction, prefix in ((0.0, "split"), (0.2, "validation_split")):
            split = multivariate_split(split_input, options["ar_order"], fraction)
            for name, actual in zip(("x_train", "y_train", "x_test", "y_test"), split):
                assert actual.dtype == np.float32
                np.testing.assert_array_equal(actual, reference[prefix + "_" + name])


def test_frozen_sources_and_case_count():
    metadata = json.loads((GOLDEN / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["cases"]) == len(list(GOLDEN.glob("*.npz"))) == 27
    for filename, expected in metadata["source_sha256"].items():
        actual = (GOLDEN.parents[1] / filename).read_bytes()
        assert hashlib.sha256(actual).hexdigest() == expected


@pytest.mark.parametrize("nodes,samples", [(5, 240), (20, 300)])
def test_synthetic_inputs(nodes, samples):
    data, adjacency = generate_nonlinear_system(nodes, samples, seed=0)
    with np.load(GOLDEN / f"synthetic{nodes}_order1_kf2_kg2_norm1.npz") as reference:
        assert_close(data, reference["input"])
        np.testing.assert_array_equal(adjacency, reference["true_adjacency"])


def test_bundled_loader_and_legacy_api():
    data, truth = load_logistic3(realization=0, n_samples=200)
    with np.load(GOLDEN / "logistic3_order1_kf2_kg2_norm1.npz") as reference:
        np.testing.assert_array_equal(data, reference["input"])
        np.testing.assert_array_equal(truth, reference["true_adjacency"])
        affinity, statistic = lsNGC(data, 1, 2, 2, 1)
        assert_close(affinity, reference["affinity"])
        assert_close(statistic, reference["f_statistic"])
        raw_f = calc_f_stat(reference["restricted_variance"], reference["unrestricted_variance"], 200, 4, 2)
        np.fill_diagonal(raw_f, 0)
        assert_close(raw_f, reference["f_statistic"])
