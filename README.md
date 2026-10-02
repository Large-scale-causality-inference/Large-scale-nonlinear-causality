# Large-scale nonlinear Granger causality (lsNGC)

lsNGC infers directed dependence from short multivariate time-series data. It compares predictions with and without a candidate source while accounting for the other observed time series. K-means centers and generalized radial basis functions represent nonlinear interactions in a compact state space. This repository provides the reference method as a NumPy/SciPy package, with scikit-learn used for k-means and fixed legacy outputs for regression testing.

**Code: Ali Vosoughi** · [Showcase](https://ali-vosoughi.github.io/) · [Paper page](https://ali-vosoughi.github.io/publications/lsngc/) · [Publication](https://doi.org/10.1038/s41598-021-87316-6) · [Project page source](docs/index.html)

## Install

Use Python 3.10 or newer. From a checkout of this repository:

```bash
git clone https://github.com/ali-vosoughi/Large-scale-nonlinear-causality.git
cd Large-scale-nonlinear-causality
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e .
```

For the notebook, figure, and tests, install the optional dependencies:

```bash
python -m pip install -e ".[demo,test]"
```

## Example

Arrays have shape `(nodes, samples)`. Matrix entry `[source, target]` describes the directed influence of the row on the column; self-influence is set to zero.

```python
import numpy as np
from lsngc import analyze, lsNGC
from lsngc.evaluate import recovery_performance
from lsngc.simulate import load_logistic3

series, truth = load_logistic3(realization=0, n_samples=200)
result = analyze(series, ar_order=1, k_f=2, k_g=2)
affinity, f_statistic = lsNGC(series, ar_order=1, k_f=2, k_g=2)
pvalues = result.pvalues
adjacency = (pvalues < 0.05).astype(np.uint8)
np.fill_diagonal(adjacency, 0)
auroc = recovery_performance([affinity], [truth])[0]
np.save("affinity.npy", affinity)
print("AUROC:", auroc)
print("Affinity (source rows, target columns):\n", affinity)
```

The affinity is the nonnegative log ratio of restricted to unrestricted residual variances; it is a continuous score, not a binary adjacency matrix. The example's `p < 0.05` threshold is uncorrected for multiple testing. Recovery AUROC follows the original code and includes diagonal entries.

The command line accepts a NumPy array or the bundled MATLAB data:

```bash
lsngc run data.npy --order 1 --kf 3 --kg 2 --out adj.npy
lsngc run datasets/7SYNTHETICS/logistic_3.mat --realization 0 --samples 200 --order 1 --kf 2 --kg 2 --out logistic3_affinity.npy
```

`--out` saves the continuous affinity matrix in NumPy format.

## API

| Function | Purpose and return value |
| --- | --- |
| `lsngc.lsNGC(series, ar_order=1, k_f=3, k_g=2, normalize=1, *, seed=123)` | Legacy-compatible entry point; returns `(affinity, f_statistic)`. |
| `lsngc.lsngc(...)` | Alias of `lsNGC`. |
| `lsngc.analyze(...)` | Returns affinity, F-statistics, p-values, both residual variances, and global/source cluster centers as named fields. |
| `lsngc.f_pvalues(F, n_samples, ar_order, k_f, k_g)` | F-distribution upper-tail probabilities using the reference degrees of freedom. |
| `lsngc.normalize_0_mean_1_std(series)` | Normalizes each row using its population standard deviation. |
| `lsngc.multivariate_split(X, ar_order, valid_percent=0)` | Returns float32 lagged training/validation arrays and their one-step targets. |
| `lsngc.calc_f_stat(RSS_R, RSS_U, n, pu, pr)` | Evaluates the restricted/unrestricted F-statistic. |
| `lsngc.evaluate.recovery_performance(Adjs, label)` | Per-matrix AUROC with the original flattening and float32 output. |
| `lsngc.simulate.load_logistic3(realization=0, n_samples=None, path=None)` | Loads `(series, true_adjacency)`; a sample limit selects the final samples. |
| `lsngc.simulate.generate_nonlinear_system(n_nodes, n_samples, seed=0, burn_in=100)` | Generates the nonlinear test system and its true adjacency. |

The historical helper names remain importable from `lsngc`. The implementation preserves the reference float32 lag embedding and numerical conventions; see [the modernization report](MODERNIZE_REPORT_2026-10-01.md) for compatibility details.

Clustering accepts an explicit `seed` keyword, defaulting to `123`. Golden tests fix `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to `1` for reproducible reductions across dependency versions.

## Reproduce the paper and the reference code

[Demo_lsNGC.ipynb](Demo_lsNGC.ipynb) runs the bundled three-node logistic benchmark. The file contains **100 realizations of 500 samples**; the notebook retains the original experiment on the first **50** realizations, using the final **50, 100, 200, and 500** samples, with `ar_order=1`, `k_f=2`, and `k_g=2`. K-means uses the explicit reference seed, `123`.

This notebook reproduces the repository demonstration. The paper's broader experiments include other simulated networks and fMRI data; those datasets and a complete replication pipeline are not bundled here. The Methods section describes empirically selected `k_f=25`, `k_g=5`, and embedding dimension selected using Cao's method; the notebook's small-system settings are different. Consult the [paper](https://doi.org/10.1038/s41598-021-87316-6) and [supplement](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-021-87316-6/MediaObjects/41598_2021_87316_MOESM1_ESM.pdf) for the experimental protocols.

Run the compatibility tests and regenerate the project-page figure:

```bash
pytest -q
python docs/figures/make_figure.py
```

The 27 committed golden cases cover the bundled logistic data and seeded five- and twenty-node systems, two lag orders, two choices each of `k_f` and `k_g`, and an unnormalized case for each system. Their inputs, outputs, seeds, source hashes, and legacy environment are recorded under [tests/golden/](tests/golden/). The untouched original implementation is archived in [legacy/](legacy/); the current package does not require PyTorch. Golden generation requires the separate recorded legacy environment and is not part of normal installation or testing.

## Citation

Axel Wismüller, Adora M. Dsouza, M. Ali Vosoughi, and Anas Abidin. “Large-scale nonlinear Granger causality for inferring directed dependence from short multivariate time-series data.” **Scientific Reports 11, 7817 (2021)**. [DOI: 10.1038/s41598-021-87316-6](https://doi.org/10.1038/s41598-021-87316-6).

```bibtex
@article{wismuller2021large,
  title={Large-scale nonlinear Granger causality for inferring directed dependence from short multivariate time-series data},
  author={Wism{\"u}ller, Axel and Dsouza, Adora M and Vosoughi, M Ali and Abidin, Anas},
  journal={Scientific reports},
  volume={11},
  number={1},
  pages={7817},
  year={2021},
  publisher={Nature Publishing Group UK London}
}
```

Code is distributed under the [MIT License](LICENSE).


## Patent notice

The large-scale Granger causality methods implemented in this repository are the subject of patent rights held by Axel Wismüller and the University of Rochester. The MIT licence of this code grants no rights under those patents. Academic and research use with citation is welcome; for any commercial use, contact the patent holders.
