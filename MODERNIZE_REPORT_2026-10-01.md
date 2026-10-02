# lsNGC modernization report — 2026-10-01

The reference implementation is now an installable NumPy/SciPy package with a CLI,
44 tests, an executed demonstration notebook, and a static project page. The 27
golden cases pass at the requested `rtol=1e-9, atol=1e-12`. Runtime installation
does not include PyTorch. Scikit-learn is used only for KMeans.

Work directory: `C:\Users\abc\projects\lsngc`.
Local branch: `modernize-2026-10`. Nothing was pushed; GitHub Pages was not enabled.
The reviewer can inspect this branch before pushing after Ali's review.

Session ID: `01a0fa3d-23dd-7401-90c2-575bbb9219de`.

## Provenance and commit order

- Original upstream revision: `6e4ddae37f44990476fc4032f2229a2088be1aad`.
- Golden commit, made before any refactor:
  `e032a499ef0b24e4def6501d79a38b7bb0ba8c6a`
  (`Characterize original implementation and freeze legacy golden outputs`).
- Modernization commit message:
  `Modernize into a NumPy/SciPy package with golden tests; add project page`.

Every original Python file, the notebook, requirements, README, and license is
retained in `legacy/`, byte-for-byte identical to the upstream Git blobs.
`.gitattributes` prevents checkout newline conversion of these files. Their hashes
are recorded in `tests/golden/metadata.json` and checked by pytest. The golden
generator imports exclusively from `legacy/`; it never imports the new package.
Archived sources and the legacy-only generator necessarily retain their original
PyTorch dependency. The installable package and compatibility import modules do not.

## What PyTorch did, and its replacement

PyTorch allocated zero-filled float32 lag/target arrays, assigned NumPy slices
into those arrays, and flattened tensors. Tensor subtraction also participated in
radial-distance calculations. There was no autograd, optimizer, learned neural
network, GPU requirement, or PyTorch regression solver. NumPy already performed
the regressions and scikit-learn performed clustering.

`np.zeros(..., dtype=np.float32)` and array reshaping replace allocation,
assignment, and flattening. The float32 conversion is retained: simply making
everything float64 would change results. Old scikit-learn converted the tensor
input back to float64 for clustering, so the replacement does that explicitly.

The clustering compatibility adapter uses the public modern `KMeans` API. It
preserves scikit-learn 0.23.1's ten child seeds, greedy k-means++ initializer,
100-iteration limit, and previous-center stopping convention. Current KMeans
defaults or merely setting `random_state=123` do not reproduce the original.
The adapter uses no saved centers or fixture lookups. All 279 ordered global and
source center arrays in the golden set match bit-for-bit. Its adapted initializer
has the required BSD attribution in `THIRD_PARTY_NOTICES`, included in the wheel.

## Golden set

`tests/golden/` contains 27 compressed NPZ cases:

| Input | Shape | Definition |
| --- | --- | --- |
| Bundled logistic data | 3 × 200 | `pt_N[0, :, -200:]`; true adjacency from `Adj` |
| Seeded synthetic system | 5 × 240 | Nonlinear autoregression, independent `RandomState(0)`, 100 burn-in samples |
| Seeded synthetic system | 20 × 300 | Same recurrence and seed convention |

For each input, the normalized cases use the full Cartesian grid
`ar_order ∈ {1,2}`, `k_f ∈ {2,3}`, `k_g ∈ {2,3}`. A ninth case uses
`ar_order=1, k_f=2, k_g=2, normalize=0`.

The synthetic recurrence is
`x_j(t)=0.45*x_j(t-1)+0.25*tanh(x_(j-1)(t-1))`
`+0.12*I[j even]*tanh(x_(j-3)(t-1))+epsilon_j(t)`,
with cyclic indices and independent Gaussian noise of standard deviation 0.2.
These are new regression-test systems, not claimed paper benchmark generators.

Each archive includes inputs, true adjacency, affinity, F statistics, original
AUROC, both residual-variance matrices, global/source cluster centers, normalized
inputs, float32 train/validation splits at fractions 0 and 0.2, and all parameters.
It also includes derived p-values and unadjusted `p<0.05` adjacency. Original
`lsNGC` returned only `(affinity, F statistic)`; p-values and thresholded graphs
are explicitly identified as derived outputs. Their degrees of freedom are
`k_g` and `T-ar_order-k_f-k_g`, following Methods Eq. (3).

The original KMeans already fixed its seed at 123. Both implementations retain
that default and the new API exposes `seed`. Golden generation also sets NumPy
and PyTorch seeds to zero and fixes OMP, OpenBLAS, and MKL threads to one. A second
legacy run reproduced all 27 cases' array bytes, dtypes, shapes, and source hashes
exactly. No unresolved random initialization difference remains in this set.

## Numerical behavior and tolerances

All golden comparisons use `numpy.allclose(rtol=1e-9, atol=1e-12)` without
relaxation. Helper lag arrays, thresholded adjacency, AUROC, and cluster centers
are checked exactly where appropriate. Observed maximum errors on Python 3.11
with NumPy 2.4.6, SciPy 1.17.1, and scikit-learn 1.9.1:

| Output | Maximum absolute difference from legacy |
| --- | ---: |
| Affinity | `8.6209e-14` |
| F statistic | `1.3712e-11` |
| p-value | `2.1833e-12` |
| Restricted residual variance | `3.4450e-13` |
| Unrestricted residual variance | `3.3740e-13` |
| Global/source cluster centers | `0` |

The largest combined tolerance fraction is approximately 0.758, for an F value
near zero. Thus the requested tolerance is retained rather than tightened.
The remaining differences are floating-point library arithmetic; no output
rounding, clipping beyond the original rules, or fixture-specific correction was
introduced. Synthetic generation across NumPy versions differs by at most
`1.11e-16` because of elementary-function rounding; the saved golden inputs are
used directly for inference comparisons.

Compatibility decisions that matter:

- Preserve population-standard-deviation normalization and float32 lag rounding.
- Preserve `pinv(H.T @ H) @ H.T @ Y` and multiplication order, then the diagonal
  of `cov(error.T)` with `ddof=1`; do not substitute another least-squares solver.
- Preserve the original radial kernel denominator `(2*sigma)**2` and row sums.
- Preserve an unusual restricted-transform calculation: for feature `j`, delete
  center columns `j:j+ar_order`, then take a Frobenius norm against **all** remaining
  centers. This differs from the individual-center expression in Methods Eq. (5).
  Correcting it would change behavior and is outside this modernization.
- Preserve zero diagonals, unclipped off-diagonal F values, nonnegative affinity,
  and AUROC computed over every matrix entry, including diagonal entries.
- Remove repeated lag construction and hoist invariant center deletion out of
  the sample loop. Golden tests validate the resulting outputs; the sensitive
  distance and summation loops retain their original arithmetic.

Additional diagnostics matched clustering exactly for one cluster, constant
data, offset data, and a run reaching the 100-iteration limit. An extra
`k_g=1` case outside the committed grid produced a near-zero F difference of
`1.2484e-12` (`-9.0165e-13` versus `3.4679e-13`), just beyond the absolute
tolerance. This is disclosed rather than hidden by rounding or a looser test.
Exact equivalence for arbitrary inputs, BLAS implementations, and thread counts
is not claimed. Invalid or degenerate input now receives explicit errors, such
as constant normalized rows, invalid center-indexing bounds, or nonpositive
denominator degrees of freedom.

## Function map

| Original | Current implementation / compatibility |
| --- | --- |
| `utils.lsNGC` | `lsngc.core.lsngc`; exported as `lsngc.lsNGC`; root import shim retained |
| `normalize_0_mean_1_std.normalize_0_mean_1_std` | `lsngc.core.normalize_0_mean_1_std`; exported at package root and old module |
| `multivariate_split.multivariate_split` | `lsngc.core.multivariate_split`; same four arrays, now NumPy arrays |
| `calc_f_stat.calc_f_stat` | `lsngc.core.calc_f_stat`; package-root alias and old module |
| `recovery_performance.recovery_performance` | `lsngc.evaluate.recovery_performance`; SciPy average ranks, float32 AUROC |
| Inline KMeans | `lsngc._clustering.cluster_centers` |
| Inline radial transform / regressions | `lsngc.core.state_space_transform` / `regression_variance` |
| New diagnostics and significance | `lsngc.analyze`, `CausalityResult`, `f_pvalues` |
| Data loading and test systems | `lsngc.simulate.load_logistic3`, `generate_nonlinear_system` |
| Command line | `lsngc.cli.main`; `lsngc run` and `python -m lsngc run` |

## Environments and reproduction

The separate `.legacy-venv` is a Conda environment with Python 3.8.20. All original
requirements were installed: NumPy 1.18.5, PyTorch 1.5.1+cpu, SciPy 1.4.1,
Matplotlib 3.2.1, sklearn 0.0, and NetworkX 2.4. The CPU build is the original
PyTorch release, not a substitute newer version. `sklearn==0.0` is a metapackage
that did not pin scikit-learn; this run explicitly uses scikit-learn 0.23.1.
The complete resolved lock is `tests/golden/legacy-requirements-lock.txt`.

```powershell
conda create --prefix .legacy-venv python=3.8 pip -y
.legacy-venv\python.exe -m pip install -r tests/golden/legacy-requirements-lock.txt -f https://download.pytorch.org/whl/torch_stable.html
.legacy-venv\python.exe tests/golden/make_golden.py --verify
```

Modern validation used separate environments:

| Environment | Python | NumPy / SciPy / scikit-learn |
| --- | --- | --- |
| Development `.venv` | 3.11.13 | 2.4.6 / 1.17.1 / 1.9.1 |
| Fresh installed wheel `.venv-clean` | 3.13.5 | 2.5.3 / 1.18.1 / 1.9.1 |
| Older supported dependencies, outside repository | 3.11.13 | 1.24.4 / 1.10.1 / 1.3.2 |

The wheel includes the original MAT data, compatibility modules, MIT license,
and third-party notice. A fresh environment, tested from outside the checkout,
successfully ran `python -c "import lsngc"`; `find_spec('torch')` returned `None`.
The installed wheel's bundled-data loader also worked outside the checkout.

## Validation and project page

- `pytest -q`: 44 tests pass, including all 27 golden cases and installed CLI
  tests for NPY/MAT inputs, statistics, normalization, errors, and input protection.
- The installed wheel passes the same 44 tests on Python 3.13 without PyTorch.
- The older supported dependency combination passes the suite. A SciPy 1.10
  truncated-MAT `IndexError` discovered during this check is now presented as
  a readable input error.
- Ran the CLI on `datasets/7SYNTHETICS/logistic_3.mat`, realization 0, final 200
  samples, order 1, `k_f=2`, `k_g=2`; saved affinity and statistics match the golden
  case. The CLI also passes a golden comparison on NPY input.
- Executed every updated notebook cell. All 200 fits (50 realizations at each
  of 50, 100, 200, and 500 samples) report AUROC `1.0 + 0.0`, matching both a fresh
  full legacy run and the original stored notebook outputs. The original notebook
  contained no figure cells. The file actually stores 100 realizations of 500
  samples; its old description was corrected, while the 50-realization benchmark
  was retained.
- Generated `docs/figures/lsngc_logistic3.png` using the refactored package and the
  committed `make_figure.py`. It shows true adjacency beside recovered affinity.
- Checked the page in headless Chrome at 390, 768, and 1280 pixels: no horizontal
  overflow, four pipeline steps, loaded figure, exact abstract, and no page errors.
  Inspected screenshots. All files under `docs/` total **81,516 bytes**, below 1 MB.
- The abstract is copied unchanged from the public article, including its original
  wording, with CC BY 4.0 attribution. Authors and BibTeX retain the printed forms.
  No separate empirical performance claim was added to the page.
- Test Actions cover Windows and Ubuntu, Python 3.10 and 3.13. The Pages workflow
  deploys `docs/` after the reviewer enables Pages with GitHub Actions as its source.
  Remote Actions and Linux execution have not been run locally: nothing was pushed,
  and this Windows machine has no installed WSL distribution.

The repository demonstration is reproducible; the full paper's other simulated
networks, fMRI data, and complete replication pipeline are not bundled. README
distinguishes its small-system settings from the paper's empirical `k_f=25`,
`k_g=5` and Cao embedding selection. No additional paper results are claimed.

Paper: [Scientific Reports 11, 7817 (2021)](https://doi.org/10.1038/s41598-021-87316-6).
Abstract/method verification: [Europe PMC full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8035412/fullTextXML).
Project page: [docs/index.html](docs/index.html).
Code: Ali Vosoughi. [Showcase](https://ali-vosoughi.github.io/) ·
[Paper page](https://ali-vosoughi.github.io/publications/lsngc/).
