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

So a moderate, fixed `L` is enough to *rank* candidates. `rework2` ranks all
candidates at that capped `L`, then trains **only the winning `d_sub`** to full OOB
convergence. Everything else (Cholesky solve, vectorized threshold, dtype-aware ridge,
zero-copy split, `fit_presplit`, lazy single-matmul inference, memory release after
`fit`) is inherited from the bit-exact rework.

### One knob: `search_effort` (speed ↔ safety of the d_sub choice)

`search_effort` ∈ [0, 1] sets how many learners each candidate is ranked with
(`max(10, round(100·effort))`; `1.0` = full convergence). Lower = faster but ranks
d_sub on noisier estimates, so an off-optimal d_sub (and slightly worse `P_E`) becomes
more likely — *how* likely depends on how flat the data's d_sub minimum is. To skip the
search entirely, pass a fixed `d_sub` (fastest; you choose the value).

Measured on the tutorial data (`P_E` vs. legacy 0.1038 at 16.3 s):

| setting | d_sub | `P_E` | time | speedup |
|---|--:|--:|--:|--:|
| `search_effort=1.0` (thorough) | 274 | 0.1018 | 10.6 s | 1.5× |
| `search_effort=0.5` (default) | 240 | 0.1033 | 6.0 s | 2.7× |
| `search_effort=0.25` | 266 | 0.1038 | 4.5 s | 3.6× |
| `search_effort=0.1` (fast) | 257 | 0.1008 | 2.9 s | 5.5× |
| `d_sub=200` (search off) | 200 | 0.1008 | 1.2 s | 13.5× |

Here `P_E` stays equivalent across the whole range because this data's minimum is wide
and flat; on a harder problem the low-effort end carries more risk of a worse d_sub.

### Extra memory win: no concat + re-split (`fit_presplit`)

The legacy data path projects each base learner's data to two `(N_trn, d_sub)` cover
and stego blocks, **concatenates** them into one matrix, and the FLD then boolean-
indexes that matrix **back apart** into cover/stego — re-materializing the projected
data ~2–3 times per learner. `rework2`'s `BaseLearner` instead routes the two blocks
straight into `FisherLinearDiscriminantLearner.fit_presplit(Xc, Xs)`, skipping the
concatenate and the re-split. Result is identical; the per-learner transient roughly
halves. This is also why `rework2` uses **float64** comfortably — the structural memory
savings make the (unsafe-for-search) float32 trick unnecessary.

During the d_sub search `rework2` additionally **discards** each candidate's learners
immediately (only the OOB error is needed), so the search phase holds essentially just
the input plus one learner's working set.

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

### Memory (peak RSS increase during training, float64)

The structural wins (zero-copy split, release after `fit`, `fit_presplit`, discarding
search learners) make `rework2` far leaner than legacy **without** float32:

| Config | input | legacy peak | rework2 peak | reduction |
|---|--:|--:|--:|--:|
| N=600  D=4096 | 39 MB | +81 MB | **+23 MB** | 3.5× |
| N=1500 D=6000 | 144 MB | +175 MB | **+16 MB** | 11× |

The remaining `rework2` peak (~input + one base learner's projection + scatter) is the
*intrinsic* FLD working set; it cannot shrink further without changing the algorithm or
lowering precision (which hurts model selection — see below).

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
