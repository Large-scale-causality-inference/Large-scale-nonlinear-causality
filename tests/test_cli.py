"""Exercise the installed command from outside the source checkout."""

import importlib
import os
from pathlib import Path
import subprocess
import sysconfig

import numpy as np
import pytest
from scipy.io import savemat

import lsngc


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture
def command():
    executable = Path(sysconfig.get_path("scripts")) / ("lsngc.exe" if os.name == "nt" else "lsngc")
    assert executable.is_file(), "Install the package before testing its CLI"
    return str(executable)


def invoke(command, tmp_path, *arguments):
    return subprocess.run(
        [command, "run", *map(str, arguments)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("extension,normalize", [("npy", True), ("mat", False)])
def test_installed_cli_matches_golden(command, tmp_path, extension, normalize):
    filename = "logistic3_order1_kf2_kg2_norm{}.npz".format(int(normalize))
    with np.load(GOLDEN / filename, allow_pickle=False) as expected:
        if extension == "npy":
            source = tmp_path / "input.npy"
            np.save(source, expected["input"])
        else:
            source = ROOT / "datasets" / "7SYNTHETICS" / "logistic_3.mat"
        original_bytes = source.read_bytes()
        output = tmp_path / "results" / "affinity.npy"
        statistics = tmp_path / "results" / "statistics.npz"
        arguments = [
            source, "--order", 1, "--kf", 2, "--kg", 2, "--samples", 200,
            "--out", output, "--statistics", statistics,
        ]
        if not normalize:
            arguments.append("--no-normalize")
        completed = invoke(command, tmp_path, *arguments)
        assert completed.returncode == 0, completed.stderr
        assert "3 x 3 affinity matrix" in completed.stdout
        assert source.read_bytes() == original_bytes
        np.testing.assert_allclose(
            np.load(output, allow_pickle=False), expected["affinity"], rtol=1e-9, atol=1e-12
        )
        with np.load(statistics, allow_pickle=False) as actual:
            array_keys = {
                "affinity", "f_statistic", "pvalues", "restricted_variance",
                "unrestricted_variance", "global_centers", "source_centers",
            }
            assert set(actual.files) == array_keys | {"ar_order", "k_f", "k_g", "normalize", "seed"}
            for key in array_keys:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-9, atol=1e-12)
            assert int(actual["ar_order"]) == 1
            assert int(actual["k_f"]) == int(actual["k_g"]) == 2
            assert bool(actual["normalize"]) is normalize
            assert int(actual["seed"]) == 123


def test_cli_rejects_overwriting_input(command, tmp_path):
    source = tmp_path / "input.npy"
    np.save(source, np.arange(30).reshape(3, 10))
    original_bytes = source.read_bytes()
    completed = invoke(command, tmp_path, source, "--out", tmp_path / "." / "input.npy")
    assert completed.returncode == 2
    assert "must differ from the input data path" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("kind", ["dimensions", "truncated_mat", "missing_mat_arrays"])
def test_cli_rejects_invalid_input_cleanly(command, tmp_path, kind):
    if kind == "dimensions":
        source = tmp_path / "bad.npy"
        np.save(source, np.ones(10))
        message = "shape (nodes, samples)"
    elif kind == "truncated_mat":
        source = tmp_path / "bad.mat"
        source.write_bytes(b"not a MATLAB file")
        message = "truncated"
    else:
        source = tmp_path / "bad.mat"
        savemat(source, {"wrong": np.ones((3, 20))})
        message = "pt_N and Adj"
    output = tmp_path / "affinity.npy"
    completed = invoke(command, tmp_path, source, "--out", output)
    assert completed.returncode == 2
    assert message in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output.exists()


def test_legacy_imports_share_package_functions():
    assert lsngc.lsNGC is lsngc.lsngc
    for module_name, function_name in (
        ("utils", "lsNGC"),
        ("calc_f_stat", "calc_f_stat"),
        ("multivariate_split", "multivariate_split"),
        ("normalize_0_mean_1_std", "normalize_0_mean_1_std"),
        ("recovery_performance", "recovery_performance"),
    ):
        assert getattr(importlib.import_module(module_name), function_name) is getattr(lsngc, function_name)
