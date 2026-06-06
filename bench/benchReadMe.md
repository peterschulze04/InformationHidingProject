# FLD Ensemble Trainer — Benchmark & Optimization Report

Baseline measurements for the legacy `FldEnsembleTrainer` (sealwatch), used to set
optimization targets for a faster, more memory-efficient, scikit-learn-compatible
re-implementation.

## Goal

The legacy `FldEnsembleTrainer` is slow, memory-hungry, and uses a non-standard
interface (separate trainer/model classes, `Xc`/`Xs` instead of `(X, y)`,
`train()` instead of `fit`). Targets: (1) reduce training time, (2) reduce peak
memory, (3) provide a scikit-learn-compatible interface — while staying
bit-identical to the reference under a `matlab_compat` mode.

## Method

- Synthetic paired cover/stego features (Gaussian, weak signal in a few dims).
- Wall-clock time = median of 3 runs per config.
- Peak resident memory sampled in a background thread (`psutil`); reported as the
  increase over the pre-training RSS. `data (MB)` = size of the input `Xc`+`Xs`.
- Hotspots via `cProfile` (cumulative + self time) on one representative config.
- Hardware: Intel Core i5-1135G7 (4 physical / 8 logical cores), 16 GB RAM.
  Python 3.12.10, NumPy 2.4.4, SciPy 1.17.1.

## Baseline results (legacy)

| Config                      | time (s) | peak mem (MB) | data (MB) | L   |
|-----------------------------|---------:|--------------:|----------:|----:|
| N=300  D=256   L=50  fixed  |     0.05 |           0.1 |       1.2 |  50 |
| N=600  D=256   L=50  fixed  |     0.09 |           2.5 |       2.5 |  50 |
| N=300  D=1024  L=50  fixed  |     0.20 |           4.9 |       4.9 |  50 |
| N=600  D=1024  L=50  fixed  |     0.34 |          12.4 |       9.8 |  50 |
| N=600  D=1024  L=100 fixed  |     0.63 |          13.0 |       9.8 | 100 |
| N=600  D=1024  auto L       |     1.26 |          13.8 |       9.8 | 209 |
| N=600  D=1024  auto d_sub+L |    33.14 |          53.4 |       9.8 | 268 |

### Hotspots (cProfile, N=600 D=1024 L=100, total 0.569 s self-time)

| Function                          | self time (s) | share |
|-----------------------------------|--------------:|------:|
| `np.linalg.solve` (FLD weights)   |         0.224 |  39 % |
| `_fast_fancy_indexing`            |         0.106 |  19 % |
| `fld.fit` (scatter, means, rest)  |         0.082 |  14 % |
| `_find_threshold` (Python loop)   |         0.071 |  12 % |
| other (train loop, OOB, setops)   |        ~0.086 |  16 % |

`base_learner.fit` accounts for **85 %** of total training time and is independent
per base learner — i.e. embarrassingly parallel.


![alt text](figures/hotspots.png)
 ![alt text](figures/memory_overhead.png) 
 ![alt text](figures/time_per_config.png)


## Interpretation

- **The cost lives in `fit`.** 85 % of time is per-learner FLD fitting, fully
  independent → parallelizable for a near-linear speedup on multi-core machines.
- **`solve` dominates and scales cubically** (`O(d_sub³)`). Harmless at the small
  `d_sub` used here, but at real steganalysis scale (SRM `d_sub` up to ~1000+) it
  becomes the bottleneck. The matrix is SPD (stabilized with `+εI`), so a Cholesky
  solve (`scipy.linalg.solve(..., assume_a="pos")`) is ~1.8× faster and closer to
  the Matlab reference (`mldivide`).
- **`_find_threshold` is a pure-Python loop** over all samples → vectorizable with a
  cumulative-sum sweep over sorted scores.
- **`_fast_fancy_indexing`** materializes an `(N, d_sub)` array twice per learner;
  this is intrinsic (the FLD needs the projected data) but drives the memory peak
  as `d_sub` grows.
- **Memory peak.** Fixed configs sit at ~1.3× the input size; the redundant
  `astype(np.float64)` copy (always copies, even when already float64) is ~1× of
  that. The `auto d_sub+L` run peaks at **5.4× data** because scatter (`d_sub²`) and
  the per-learner transients grow with `d_sub`. At SRM scale the redundant float64
  copy alone is ≈1.4 GB — the headline memory win.
- **The real cost driver is the search.** `auto d_sub+L` (33 s) is ~25–50× a single
  fixed run; this is the realistic baseline for "faster".

## Optimization targets

| Metric                 | Baseline        | Target            | Lever                                            |
|------------------------|-----------------|-------------------|--------------------------------------------------|
| Time (`auto d_sub+L`)  | 33 s            | 6–10 s (3–5×)     | joblib parallel fit + Cholesky                   |
| `solve` share          | 39 %            | ~20 %             | `scipy.linalg.solve(assume_a="pos")`             |
| `_find_threshold`      | 12 %            | <3 %              | vectorize (cumsum)                               |
| Peak memory            | 1.3–5.4× data   | 0.3–0.5× data     | `asarray` instead of `astype`; no double-hold    |
| Interface              | trainer+model   | sklearn `fit/predict/score`, `clone`-able | rewrite as `BaseEstimator`/`ClassifierMixin` |

Caveats: parallelism has fixed overhead and only pays off on large jobs (the sub-second
configs get *slower* in parallel) → `n_jobs=1` stays the default. The `(N, d_sub)`
transient is intrinsic; the clean memory win is removing the float64 copy (scales to
GB at SRM size).

## Results after optimization

Legacy vs. reworked, median of 3, both timed back-to-back on the same machine
(`bench/bench_comparison.py`). `peak` = peak RSS increase during training; `kept` =
RSS still held *after* `fit` returns (the legacy trainer keeps the full dataset on
`self`; the rework releases it). `L l/r` = number of base learners chosen by each —
identical in every config, i.e. the Cholesky numerics never flip a search decision.

| Config                      | legacy (s) | rework (s) | speedup | legacy peak | rework peak | legacy kept | rework kept | L (legacy/rework) |
|-----------------------------|-----------:|-----------:|--------:|------------:|------------:|------------:|------------:|------------------:|
| N=600 D=1024 L=100          |       0.73 |       0.45 |   1.6×  |    14.3 MB  |     4.5 MB  |    9.9 MB   |    0.0 MB   |       100 / 100   |
| N=600 D=1024 auto L         |       1.62 |       0.97 |   1.7×  |    15.5 MB  |     5.2 MB  |   12.7 MB   |    0.2 MB   |       209 / 209   |
| N=600 D=1024 auto d_sub+L   |      51.41 |      30.87 |   1.7×  |    56.6 MB  |    46.9 MB  |   12.0 MB   |    3.6 MB   |       268 / 268   |
| N=600 D=2048 d_sub=512      |       2.25 |       1.72 |   1.3×  |    49.9 MB  |    30.1 MB  |   19.7 MB   |    0.0 MB   |        50 / 50    |

(Timing = median of 3; peak is from a single representative run and is noisy on a
loaded laptop — the controlled A/B below isolates the memory effect.)

**`solve` micro-benchmark** (cProfile, N=600 D=2048 d_sub=512, 50 calls): the FLD
weight solve drops from `np.linalg.solve` **0.936 s** (LU) to
`scipy.linalg.solve(assume_a="pos")` **0.405 s** (Cholesky) — **2.3×** on that op,
which is the lever behind the end-to-end speedup.

**`_find_threshold`** is now vectorized (cumulative-sum sweep over the sorted scores
instead of the per-sample Python loop). The threshold function itself is **~4.4×**
faster (0.91 ms → 0.21 ms at N=600 d_sub=512), and a 500-trial head-to-head against
the original loop on random data is **bit-identical** (0 mismatches in chosen weights
and bias).

**Vectorized inference (≈100× faster `predict`).** `decision_function`/`predict`
previously looped over the `L` base learners in Python, each doing its own column
projection `X[:, subspace] @ w - b`. Since every learner's score equals
`X @ W[:, i] - b[i]` where `W[:, i]` is that learner's weight vector scattered back
into the full feature space (zeros outside its subspace), the whole signed majority
vote is a **single matmul** `X @ W - b`. The dense `(n_features, L)` weight matrix is
built lazily and cached on first predict, so the memory is only paid for inference.
On a real trained ensemble (n_test=2000, L=500) this is **~118× faster** (2819 ms →
24 ms) with **identical** votes / labels / `decision_function`. Only inference is
affected — the training path and the Matlab bit-identity gate are untouched.

**float32 compute path** (non-compat, `dtype=np.float32`): halves the per-fit memory
transient (e.g. N=1500 D=6000: 175.6 MB → 89.4 MB; N=2000 D=8000: 314 MB → 161 MB)
**and** runs the scatter/solve in float32 for **1.67–1.73× faster** training than
float64 across all subspace sizes. This required a fix: the legacy `1e-10` diagonal
ridge is below float32's machine epsilon (~1.2e-7) and silently fails to regularize
the scatter matrix, so the float32 Cholesky was generating denormals and running ~6×
*slower* at large `d_sub` (e.g. d_sub=1024: 30 s vs. 5 s). The ridge is now
dtype-aware — float64 keeps exactly `1e-10` (bit-identical to the Matlab reference),
float32 uses a scale-aware `1e-6 × mean(diag)` that is robust to data scaling
(identical timing at data scale ×1 and ×1000).

> ⚠️ **float32 only with a *fixed* `d_sub`.** A later measurement on real features
> (Matlab tutorial set) showed that with the **automatic `d_sub` search**, float32's
> reduced precision biases the OOB error estimates that drive model selection: the
> compass search then picks a too-small `d_sub` (103 instead of 274) and the detection
> error `P_E` worsens by ~0.02 (2 percentage points). The earlier "accuracy matches
> float64 within noise" only holds for synthetic near-chance data / fixed `d_sub`, not
> for the `auto` search. Use **float64** for the `auto` search to keep detection
> accuracy; see `../sealwatch/ensemble_classifier_rework2/README.md` for the breakdown.

**Zero-copy cover/stego split (peak win).** The earlier rework still duplicated the
whole dataset inside `fit`: `np.ascontiguousarray(X[y == neg])` boolean-indexes a
*copy* of each class and holds it for the entire fit, on top of the caller's `X`.
When each class is a contiguous row block of a C-contiguous `X` with the working
dtype (the common case — the legacy shim hands over cover-then-stego float64 data),
`_split_classes` now returns zero-copy **views** instead. `Xc`/`Xs` are read-only
during training, so this is safe and bit-identical; arbitrary label orders / dtype
changes still fall back to a copy. Controlled A/B on identical float64 data (copy
split vs. view split, peak RSS increase):

| Config            | input   | copy split | view split | peak reduction |
|-------------------|--------:|-----------:|-----------:|---------------:|
| N=600  D=4096     | 39.3 MB |  +82.5 MB  |  +32.0 MB  |  2.6× (−51 MB) |
| N=1500 D=6000     | 144 MB  | +175.1 MB  |  +32.8 MB  |  5.3× (−142 MB)|

The win grows with dataset size: the eliminated copy is ~1× the data, so the training
peak drops from ≈ input + transients + a full copy to just input + transients. At SRM
scale (34 671 dims) that removed copy is on the order of GB.

The other memory win is `kept`: the legacy trainer holds the whole feature matrix
(~10–20 MB here) for the object's lifetime, while the rework releases it after `fit`.

Bit-identity vs. legacy under `matlab_compat=True`: **PASS** — `search_d_sub`,
`search_L`, and `search_oob` match the Matlab reference for both implementations
(`test/test_fld_ensemble_classifier.py`, 4/4 passing).

> Note: parallel training (`n_jobs`) was prototyped but dropped — the per-learner fit
> is cheap enough that joblib's fixed overhead made the sub-second configs *slower*,
> and the search loop is inherently sequential (each `d_sub` depends on the previous
> OOB error). Training stays sequential; `n_jobs` remains a no-op placeholder.

## Reproduce

```bash
python bench/benchmark_fld_ensemble.py     # baseline timing/memory grid + cProfile
python bench/bench_comparison.py           # legacy vs rework table + solve profile + float32
python bench/plot_benchmark.py             # figures into bench/figures/
python -m pytest test/test_fld_ensemble_classifier.py   # Matlab bit-identity gate
```

### Optimization Plan
Angepasste, sicherere Reihenfolge (Parallelität ans Ende, optional):

sklearn-Interface + asarray + bincount — bit-identisch → exakt validieren.
Cholesky-Solve in fld.py (der 39%-Posten) — wirkt überall, reproduzierbar. Ändert Numerik leicht → mit Toleranz validieren.
_find_threshold vektorisieren — berechnet denselben Schwellwert, nur ohne Python-Schleife → kann bit-identisch bleiben. Sauberer 12%-Win.
Optional float32-Pfad (Memory, nur Nicht-Compat).
Optional n_jobs (joblib) — maschinenabhängiger Bonus, Default 1.