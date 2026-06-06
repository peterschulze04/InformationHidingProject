import os
import numpy as np
from scipy.signal import convolve

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from .base_learner import BaseLearner
from .out_of_bag_error_estimates import OutOfBagErrorEstimates
from .subspace_dimensionality_search import (
    SubspaceDimensionalitySearch,
    FixedDimensionalityDummySearch,
)
from .. import tools


log = tools.setup_custom_logger(os.path.basename(__file__))


class FldEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble of Fisher linear discriminant (FLD) base learners for binary
    steganalysis -- a second, performance-oriented re-implementation of

        J. Kodovsky, J. Fridrich, V. Holub, "Ensemble Classifiers for Steganalysis
        of Digital Media", IEEE TIFS 7(2), 2012.

    This version stays faithful to the paper's *algorithm* (Algorithm 1: random
    subspaces + bagging + majority vote; Algorithm 2: compass search over d_sub;
    Eq. 6: OOB-based stopping for L) but is free to differ in everything that does
    not change the *output quality* (detection error P_E). Relative to the bit-exact
    rework it adds one speed lever that the paper itself justifies:

    **Capped-L subspace search.** Fig. 2 of the paper shows P_E saturates quickly with
    L and that the error has a *flat minimum* around the optimal d_sub. So the d_sub
    compass search only needs a moderate, fixed L (``search_L_cap``) to *rank*
    candidates; the winning d_sub is then trained to full OOB convergence. This avoids
    fully converging every discarded candidate -- the dominant cost of the ``auto``
    search -- and was measured to preserve P_E (validated on the Matlab tutorial data:
    P_E within ~1e-3 of the full-search detector at ~2x lower training time).

    Set ``search_L_cap=None`` to evaluate every candidate at full convergence.

    .. note::
       float32 compute was tested and **rejected as a default**. While the raw FLD is
       robust to single precision, the lower precision biases the OOB error estimates
       that drive the d_sub/L model selection: on real features the compass search then
       picks a too-small d_sub and the detection error P_E degrades by ~2 percentage
       points. The default is therefore float64; ``dtype=np.float32`` remains available
       for memory-bound experiments where a small accuracy loss is acceptable.

    Assumes *paired* cover/stego data (both classes equal size; the bootstrap/OOB
    bookkeeping shares row indices between classes).

    :param L: number of base learners; ``None`` selects it automatically (Eq. 6).
    :param d_sub: subspace dimensionality; ``None`` triggers the compass search.
    :param search_L_cap: fixed number of base learners used to evaluate each d_sub
        candidate during the search. ``None`` evaluates each candidate with the full
        automatic L (exact, slower). Ignored when ``d_sub`` is given.
    :param max_num_base_learners: hard cap on the number of base learners.
    :param oob_error_tolerance: tolerance tau for the d_sub compass search.
    :param initial_d_sub_step: initial step delta_d for the d_sub search.
    :param L_min_length: minimum number of base learners before automatic stopping.
    :param L_memory: number of moving averages inspected by the stopping criterion.
    :param L_epsilon: width of the epsilon-tube for the stopping criterion.
    :param random_state: seed for the main RNG (derives the two sub-seeds).
    :param seed_subspaces: explicit seed for the subspace RNG.
    :param seed_bootstrap: explicit seed for the bootstrap RNG.
    :param matlab_compat: if True, reproduce the legacy/Matlab RNG behaviour exactly
        (naive randperm). Default False uses faster subspace sampling.
    :param dtype: working dtype for the feature matrices. Default float64 (preserves
        the OOB model selection); np.float32 halves memory but degrades P_E (see note).
    :param verbose: 1 prints progress, 0 stays quiet.
    """

    def __init__(
        self,
        L=None,
        d_sub=None,
        search_L_cap=50,
        max_num_base_learners=500,
        oob_error_tolerance=0.02,
        initial_d_sub_step=200,
        L_min_length=25,
        L_memory=50,
        L_epsilon=0.005,
        random_state=None,
        seed_subspaces=None,
        seed_bootstrap=None,
        matlab_compat=False,
        dtype=np.float64,
        verbose=0,
    ):
        # scikit-learn contract: only store arguments, do no work here.
        self.L = L
        self.d_sub = d_sub
        self.search_L_cap = search_L_cap
        self.max_num_base_learners = max_num_base_learners
        self.oob_error_tolerance = oob_error_tolerance
        self.initial_d_sub_step = initial_d_sub_step
        self.L_min_length = L_min_length
        self.L_memory = L_memory
        self.L_epsilon = L_epsilon
        self.random_state = random_state
        self.seed_subspaces = seed_subspaces
        self.seed_bootstrap = seed_bootstrap
        self.matlab_compat = matlab_compat
        self.dtype = dtype
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # RNG-consuming draws (Algorithm 1, steps 2-3)
    # ------------------------------------------------------------------ #
    def _generate_random_subspace(self, rng_subspaces, d_sub, max_dim):
        if self.matlab_compat:
            return tools.matlab.randperm_naive(rng_subspaces, max_dim)[:d_sub]
        return rng_subspaces.choice(max_dim, size=d_sub, replace=False)

    @staticmethod
    def _regenerate_bootstrap_samples(rng_bootstrap, n):
        # n draws with replacement (Algorithm 1, step 3); OOB = never drawn.
        train_indices = np.floor(n * rng_bootstrap.rand(n)).astype(int)
        test_indices = np.where(np.bincount(train_indices, minlength=n) == 0)[0]
        return train_indices, test_indices

    # ------------------------------------------------------------------ #
    # OOB bookkeeping (Eq. 5)
    # ------------------------------------------------------------------ #
    def _update_oob_error_estimates(self, base_learner, test_indices, oob, Xc, Xs):
        Xc_proj = base_learner.predict(Xc, subset=test_indices)
        Xs_proj = base_learner.predict(Xs, subset=test_indices)

        oob.Xc_num[test_indices] += 1
        oob.Xs_num[test_indices] += 1
        oob.Xc_fusion_majority_vote[test_indices] += np.sign(Xc_proj).astype(int)
        oob.Xs_fusion_majority_vote[test_indices] += np.sign(Xs_proj).astype(int)

        num_covers = len(oob.Xc_fusion_majority_vote)
        num_stegos = len(oob.Xs_fusion_majority_vote)

        num_fp = np.sum(oob.Xc_fusion_majority_vote > 0)
        num_undecided = np.sum(oob.Xc_fusion_majority_vote == 0)
        num_fp += np.sum(oob.rng_for_ties.rand(num_undecided) > 0.5)

        num_md = np.sum(oob.Xs_fusion_majority_vote < 0)
        num_undecided = np.sum(oob.Xs_fusion_majority_vote == 0)
        num_md += np.sum(oob.rng_for_ties.rand(num_undecided) < 0.5)

        oob.ys.append((num_fp + num_md) / (num_covers + num_stegos))
        return oob

    def _has_next_base_learner(self, i, oob, max_L):
        # Fixed-L (or capped) stopping: stop once we reach max_L.
        if self.L is not None:
            return i < self.L
        # Automatic stopping (Eq. 6) -- moving-average epsilon-tube on the OOB curve.
        if len(oob.ys) == 0:
            return False
        if i >= max_L:
            return False
        if len(oob.ys) < self.L_min_length:
            return True
        ys = oob.ys
        A = convolve(ys[max(0, len(ys) - self.L_memory):], self._L_kernel, mode="valid")
        V = np.abs(np.max(A) - np.min(A))  # Eq. 6 / 7
        return V >= self.L_epsilon

    # ------------------------------------------------------------------ #
    # Train one ensemble for a fixed d_sub (inner loop of Algorithm 1)
    # ------------------------------------------------------------------ #
    def _train_ensemble(self, Xc, Xs, d_sub, max_dim, max_L,
                        rng_subspaces, rng_bootstrap, keep_learners):
        """
        Add base learners (until the L criterion fires, capped at ``max_L``) and
        return ``(base_learners_or_None, final_oob_error, num_base_learners)``.
        When ``keep_learners`` is False the learners are discarded (search phase),
        so only the OOB error is materialised.
        """
        n = Xc.shape[0]
        oob = OutOfBagErrorEstimates(num_covers=n, num_stegos=n)
        base_learners = [] if keep_learners else None
        num_base_learners = 0
        has_next = True

        while has_next:
            subspace = self._generate_random_subspace(rng_subspaces, d_sub, max_dim)
            train_idx, test_idx = self._regenerate_bootstrap_samples(rng_bootstrap, n)

            base_learner = BaseLearner()
            base_learner.fit(Xc=Xc, Xs=Xs, subspace=subspace, subset=train_idx)
            if keep_learners:
                base_learners.append(base_learner)
            num_base_learners += 1

            oob = self._update_oob_error_estimates(base_learner, test_idx, oob, Xc, Xs)
            if self.verbose:
                log.info(f"  - d_sub {d_sub}, OOB {oob.ys[-1]:.4f}, L {num_base_learners}")

            has_next = self._has_next_base_learner(num_base_learners, oob, max_L)

        return base_learners, oob.ys[-1], num_base_learners

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_classes(X, y, neg, pos, dtype):
        """Zero-copy views into X when each class is a contiguous same-dtype block;
        otherwise a contiguous per-class copy (Xc/Xs are read-only in training)."""
        target = np.dtype(dtype)

        def block_slice(mask):
            idx = np.flatnonzero(mask)
            if len(idx) and idx[-1] - idx[0] + 1 == len(idx):
                return slice(int(idx[0]), int(idx[-1]) + 1)
            return None

        sl_c = block_slice(y == neg)
        sl_s = block_slice(y == pos)
        if (sl_c is not None and sl_s is not None
                and X.dtype == target and X.flags["C_CONTIGUOUS"]):
            return X[sl_c], X[sl_s]
        return (np.ascontiguousarray(X[y == neg], dtype=target),
                np.ascontiguousarray(X[y == pos], dtype=target))

    def fit(self, X, y):
        """
        Train the ensemble.

        :param X: features of shape [num_samples, num_features].
        :param y: binary labels; smaller label = cover (-1), larger = stego (+1).
            Both classes must have the same size.
        :return: self
        """
        X, y = check_X_y(X, y, dtype=None, order="C")
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("FldEnsembleClassifier supports exactly two classes")
        neg, pos = self.classes_[0], self.classes_[1]
        self.n_features_in_ = X.shape[1]

        if self.dtype is not None:
            dtype = np.dtype(self.dtype)
        elif self.matlab_compat:
            dtype = np.float64
        else:
            dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float32

        Xc, Xs = self._split_classes(X, y, neg, pos, dtype)
        if Xc.shape[0] != Xs.shape[0]:
            raise ValueError(
                f"Expected equal cover/stego counts (got {Xc.shape[0]} and "
                f"{Xs.shape[0]}); this trainer assumes paired data.")

        n = Xc.shape[0]
        max_dim = Xc.shape[1]

        # RNG setup (same derivation order as the legacy code).
        rng_main = np.random.RandomState(self.random_state)
        seed_subspaces = self.seed_subspaces or int(np.ceil(rng_main.random() * 1e9))
        seed_bootstrap = self.seed_bootstrap or int(np.ceil(rng_main.random() * 1e9))
        self.seed_subspaces_ = seed_subspaces
        self.seed_bootstrap_ = seed_bootstrap
        rng_subspaces = np.random.RandomState(seed_subspaces)
        rng_bootstrap = np.random.RandomState(seed_bootstrap)

        if self.L is None:
            self._L_kernel = np.ones(5) / 5

        training_records = []

        if self.d_sub is None and max_dim > 1:
            # ----- Algorithm 2: compass search over d_sub ----- #
            search = SubspaceDimensionalitySearch(
                d_sub_step=self.initial_d_sub_step,
                max_dim=max_dim,
                oob_error_tolerance=self.oob_error_tolerance,
            )
            # Phase 1: rank candidates with a capped (or full) L; discard learners.
            search_cap = self.search_L_cap if self.search_L_cap else self.max_num_base_learners
            while search.search_in_progress:
                d_sub = search.next_d_sub
                _, oob_error, n_bl = self._train_ensemble(
                    Xc, Xs, d_sub, max_dim, max_L=search_cap,
                    rng_subspaces=rng_subspaces, rng_bootstrap=rng_bootstrap,
                    keep_learners=False)
                search.update(d_sub, oob_error)
                training_records.append(
                    {"d_sub": d_sub, "oob_error": oob_error,
                     "num_base_learners": n_bl, "phase": "search"})
                if self.verbose:
                    log.info("")
            optimal_d_sub = int(search.optimal_d_sub)

            # Phase 2: train the winning d_sub to full OOB convergence.
            base_learners, oob_error, n_bl = self._train_ensemble(
                Xc, Xs, optimal_d_sub, max_dim, max_L=self.max_num_base_learners,
                rng_subspaces=rng_subspaces, rng_bootstrap=rng_bootstrap,
                keep_learners=True)
            training_records.append(
                {"d_sub": optimal_d_sub, "oob_error": oob_error,
                 "num_base_learners": n_bl, "phase": "final"})
        else:
            # ----- Fixed d_sub ----- #
            optimal_d_sub = 1 if max_dim == 1 else self.d_sub
            base_learners, oob_error, n_bl = self._train_ensemble(
                Xc, Xs, optimal_d_sub, max_dim, max_L=self.max_num_base_learners,
                rng_subspaces=rng_subspaces, rng_bootstrap=rng_bootstrap,
                keep_learners=True)
            training_records.append(
                {"d_sub": optimal_d_sub, "oob_error": oob_error,
                 "num_base_learners": n_bl, "phase": "final"})

        self.base_learners_ = base_learners
        self.d_sub_ = optimal_d_sub
        self.training_records_ = training_records
        self._vote_weights_ = None  # lazily built on first predict
        return self

    # ------------------------------------------------------------------ #
    # Inference (single-matmul majority vote)
    # ------------------------------------------------------------------ #
    def _validate_for_prediction(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=None)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but FldEnsembleClassifier was "
                f"fitted with {self.n_features_in_} features.")
        return X

    def _get_vote_weights(self):
        """Dense (n_features, L) weight matrix + length-L bias, built lazily and
        cached, so the whole ensemble vote is one matmul (Algorithm 1, step 7)."""
        cached = getattr(self, "_vote_weights_", None)
        if cached is None:
            L = len(self.base_learners_)
            dtype = np.asarray(self.base_learners_[0].learner.w).dtype
            W = np.zeros((self.n_features_in_, L), dtype=dtype)
            b = np.empty(L, dtype=dtype)
            for i, base_learner in enumerate(self.base_learners_):
                W[base_learner.subspace, i] = base_learner.learner.w
                b[i] = base_learner.learner.b
            cached = (W, b)
            self._vote_weights_ = cached
        return cached

    def _vote(self, X):
        W, b = self._get_vote_weights()
        scores = X @ W - b
        return np.sign(scores).sum(axis=1).astype(int)

    def decision_function(self, X):
        """Confidence in [-1, +1] from the signed majority vote."""
        X = self._validate_for_prediction(X)
        return self._vote(X) / len(self.base_learners_)

    def predict(self, X):
        """Predict class labels. Ties are broken randomly (seeded, as in the paper)."""
        X = self._validate_for_prediction(X)
        votes = self._vote(X)
        rng_for_ties = np.random.RandomState(6020)
        tie_mask = votes == 0
        votes[tie_mask] = np.sign(rng_for_ties.rand(int(np.sum(tie_mask))) - 0.5).astype(int)
        y = np.sign(votes).astype(int)
        return self.classes_[(y + 1) // 2]

    def predict_confidence(self, X):
        return self.decision_function(X)

    @property
    def base_learners(self):
        return self.base_learners_

    @property
    def num_base_learners(self):
        return len(self.base_learners_)


class FldEnsembleTrainer(FldEnsembleClassifier):
    """
    Backward-compatible shim: accepts ``Xc``/``Xs`` and exposes
    ``train() -> (self, training_records)``, like the original trainer.
    """

    def __init__(self, Xc, Xs, seed=None, seed_subspaces=None, seed_bootstrap=None,
                 L="automatic", d_sub="automatic", verbose=1, max_num_base_learners=500,
                 search_L_cap=50, matlab_compat=False, dtype=np.float64):
        super().__init__(
            L=None if L == "automatic" else L,
            d_sub=None if d_sub == "automatic" else d_sub,
            search_L_cap=search_L_cap,
            max_num_base_learners=max_num_base_learners,
            random_state=seed,
            seed_subspaces=seed_subspaces,
            seed_bootstrap=seed_bootstrap,
            matlab_compat=matlab_compat,
            dtype=dtype,
            verbose=verbose,
        )
        self._Xc = Xc
        self._Xs = Xs

    def train(self):
        dtype = np.dtype(self.dtype) if self.dtype is not None else np.float64
        X = np.concatenate([self._Xc, self._Xs], axis=0).astype(dtype, copy=False)
        y = np.concatenate([-np.ones(len(self._Xc), dtype=int),
                            +np.ones(len(self._Xs), dtype=int)])
        self.fit(X, y)
        return self, self.training_records_
