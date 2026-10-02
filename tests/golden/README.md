# Frozen reference results

These arrays were generated **before the refactor**, using the exact original
source retained in `legacy/`. `make_golden.py` imports only those files; it never
imports the modern package. `metadata.json` records all dependency versions,
source and input hashes, parameters, random seeds, shapes, and array hashes.

The 27 cases cover the bundled logistic dataset (realization 0, final 200
samples), a 5-node nonlinear VAR with 240 samples, and a 20-node nonlinear VAR
with 300 samples. Each system uses all eight combinations of orders 1/2, `k_f`
2/3, and `k_g` 2/3 with normalization, plus one unnormalized case. The synthetic
systems each use NumPy `RandomState(0)` with 100 burn-in samples. Their equations
and true directed graphs are specified in the generator docstring. They are
regression inputs, not claimed reproductions of the paper's synthetic systems.

Each archive contains the input, true graph, affinity, F statistic, p-values,
unadjusted `p < 0.05` adjacency, original AUROC (including the diagonal), both
residual-variance matrices, global and per-source KMeans centers, normalized
input, and the original float32 lagged
train/validation arrays at validation fractions 0 and 0.2. Rows indicate
sources and columns indicate targets. P-values use numerator degrees of freedom
`k_g` and denominator degrees of freedom `T - ar_order - k_f - k_g`; they are
derived from the original F statistic rather than returned by legacy `lsNGC`.

Recreate only in the recorded legacy environment:

```powershell
.legacy-venv\python.exe tests/golden/make_golden.py
.legacy-venv\python.exe tests/golden/make_golden.py --verify
```

The verification run compares array bytes, dtype, and shape exactly and checks
the frozen source hashes. NPZ ZIP timestamps are intentionally excluded. The
original KMeans has `random_state=123`; NumPy and torch are seeded at zero for
every case. BLAS/OpenMP thread counts are fixed at one. The modern regression
test uses the documented numerical tolerance for cross-version differences.
