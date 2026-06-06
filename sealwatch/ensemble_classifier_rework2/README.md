# `ensemble_classifier_rework2` — paper-faithful, output-equivalent fast trainer

A second re-implementation of the FLD ensemble trainer from

> J. Kodovský, J. Fridrich, V. Holub. *Ensemble Classifiers for Steganalysis of
> Digital Media.* IEEE TIFS 7(2), 2012.

The first rework (`ensemble_classifier_rework`) is **bit-identical** to the Matlab
reference. This second rework relaxes that to **output-equivalence**: it must produce
a detector of the *same quality* (test detection error `P_E` / "security"), but is
free to differ internally. It stays faithful to the paper's algorithm (Algorithm 1:
random subspaces + bagging + majority vote; Algorithm 2: compass search over `d_sub`;
Eq. 6: OOB-based stopping for `L`).

## What is different vs. the bit-exact rework

**Capped-L subspace search (the one new lever).** The bit-exact trainer fully
converges an ensemble (auto-`L`, ~100–300 learners) for *every* `d_sub` candidate the
compass search visits, then keeps only the winner. The paper's Fig. 2 shows two things
that make this wasteful:
- `P_E` **saturates quickly** with `L`, and
- `P_E` has a **flat minimum** around the optimal `d_sub`.

So a moderate, fixed `L` (`search_L_cap`, default 50) is enough to *rank* candidates.
`rework2` ranks all candidates at the capped `L`, then trains **only the winning
`d_sub`** to full OOB convergence. Everything else (Cholesky solve, vectorized
threshold, dtype-aware ridge, zero-copy split, lazy single-matmul inference,
memory release after `fit`) is inherited from the bit-exact rework.

`search_L_cap=None` recovers the full-convergence search.

## Validation (Matlab tutorial data, real features, held-out test set)

`N=997` per class, `D=548`. `P_E` measured on the bundled `TST_cover`/`TST_stego`.

| Trainer | `P_E` (test) | training time | speedup |
|---|--:|--:|--:|
| legacy (float64, full search) | 0.1038 | 16.3 s | 1.0× |
| **rework2 (float64, capped search)** | **0.1036** (mean over 5 seeds) | **7.2 s** | **2.2×** |

`P_E` difference is **−0.0002** — within run-to-run noise, i.e. **equally secure**.
The selected `d_sub` lands in the flat minimum (206–274 across seeds), all giving
equivalent `P_E`.

**The speedup grows with the cost of the search** (more candidates, larger
`d_sub`/`L`) — exactly the high-dimensional regime the paper targets. Timing of the
full `auto d_sub+L` search (synthetic data; timing only, since synthetic features are
near-chance and `P_E` is not meaningful there — the quality proof is the tutorial row
above):

| Config | legacy | rework2 | speedup |
|---|--:|--:|--:|
| N=997  D=548  (tutorial, real) | 16.3 s | 7.2 s | 2.2× |
| N=600  D=1024 (synthetic) | 47.8 s | 23.3 s | 2.1× |
| N=600  D=2048 (synthetic) | 236.7 s | 30.5 s | **7.8×** |

## What was tried and rejected for this rework

- **float32 compute.** Tempting (≈1.7× faster, half the memory) and the raw FLD is
  robust to single precision — but it **biases the OOB error estimates** that drive
  the `d_sub`/`L` model selection. On real features the compass search then picks a
  too-small `d_sub` (e.g. 103 instead of 274) and `P_E` degrades by ~0.02 (2
  percentage points). That violates the output-equivalence goal, so `dtype` defaults
  to **float64**. (float32 remains available for memory-bound experiments.)
- **Threading / joblib, batched BLAS** — no gain on CPU; the heavy ops are already in
  multi-threaded BLAS, and numpy has no faster batched gemm/solve than the loop. See
  `../ensemble_classifier_rework/CHANGES.md` §8.

## Honest summary

On CPU, with the hard constraint of equal detection accuracy, the achievable extra
training speedup over the bit-exact rework is the **capped-L search (~1.5–3×,
data-dependent)**. Larger multiples on CPU require either accepting worse `P_E`
(float32, search pruning) or different hardware (GPU). The capped search is the one
lever that is both paper-justified and `P_E`-preserving.

## Usage

```python
# scikit-learn interface
from sealwatch.ensemble_classifier_rework2 import FldEnsembleClassifier
clf = FldEnsembleClassifier(random_state=12345).fit(X, y)   # auto d_sub + capped search
y_hat = clf.predict(X_test)

# legacy-style shim
from sealwatch.ensemble_classifier_rework2 import FldEnsembleTrainer
ens, records = FldEnsembleTrainer(Xc=Xc, Xs=Xs, seed=12345).train()
```
