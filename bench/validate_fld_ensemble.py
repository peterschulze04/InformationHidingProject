"""
Validate the new FldEnsembleClassifier against the legacy FldEnsembleTrainer.

Checks bit-identity (same seed -> identical base learners, OOB trajectory, and
predictions) across fixed/automatic d_sub and L, sequential and parallel.

Prerequisite: keep the original implementation as `_legacy_fld_ensemble_trainer.py`.
"""

from sealwatch.ensemble_classifier_rework.fld_ensemble_trainer import (
    FldEnsembleClassifier,
    FldEnsembleTrainer as NewTrainer,
)
from sealwatch.ensemble_classifier.fld_ensemble_trainer import (
    FldEnsembleTrainer as LegacyTrainer,
)
import time
import numpy as np

import importlib.util
import logging
import pathlib
import sys
import types

_LOAD_ORDER = [
    "fld", "base_learner", "out_of_bag_error_estimates",
    "subspace_dimensionality_search", "ensemble_classifier",
    "fld_ensemble_trainer",
]


def _make_package(full_name, dir_path):
    if full_name in sys.modules:
        return sys.modules[full_name]
    mod = types.ModuleType(full_name)
    mod.__path__ = [str(dir_path)]
    mod.__package__ = full_name
    sys.modules[full_name] = mod
    if "." in full_name:
        parent, simple = full_name.rsplit(".", 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], simple, mod)
    return mod


def _load_module(full_name, path, package):
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    parent, simple = full_name.rsplit(".", 1)
    setattr(sys.modules[parent], simple, mod)
    return mod


def _load_subpackage(pkg_name, dir_path):
    _make_package(pkg_name, dir_path)
    for name in _LOAD_ORDER:
        p = dir_path / f"{name}.py"
        if p.exists():
            _load_module(f"{pkg_name}.{name}", p, pkg_name)


def _bootstrap():
    # Assumes this script lives one level below the repo root
    # (e.g. <repo>/bench/ or <repo>/experiments/). Adjust parents[N] otherwise.
    repo_root = pathlib.Path(__file__).resolve().parents[1]   # ...\project\sealwatch
    pkg_dir = repo_root / "sealwatch"
    tools_dir = pkg_dir / "tools"

    # fake top-level package (no heavy __init__)
    _make_package("sealwatch", pkg_dir)

    # minimal sealwatch.tools: only randperm_naive + setup_custom_logger
    if "sealwatch.tools" not in sys.modules:
        _make_package("sealwatch.tools", tools_dir)
        _load_module("sealwatch.tools.matlab", tools_dir / "matlab.py", "sealwatch.tools")

        def setup_custom_logger(name):
            logger = logging.getLogger(str(name))
            if not logger.handlers:
                h = logging.StreamHandler(sys.stdout)
                h.setFormatter(logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
                logger.addHandler(h)
            logger.setLevel(logging.DEBUG)
            return logger
        sys.modules["sealwatch.tools"].setup_custom_logger = setup_custom_logger

    # legacy first (rework's base_learner may reference it if you didn't switch
    # to a relative import), then the rework package
    _load_subpackage("sealwatch.ensemble_classifier", pkg_dir / "ensemble_classifier")
    _load_subpackage("sealwatch.ensemble_classifier_rework",
                     pkg_dir / "ensemble_classifier_rework")


_bootstrap()


def make_paired_data(n=300, d=64, n_signal=8, shift=0.35, seed=0):
    """Synthetic paired cover/stego features with weak signal in a few dimensions."""
    rng = np.random.RandomState(seed)
    Xc = rng.randn(n, d).astype(np.float64)
    Xs = Xc + rng.randn(n, d).astype(np.float64) * 0.1
    signal_dims = rng.choice(d, size=n_signal, replace=False)
    Xs[:, signal_dims] += shift  # stego differs slightly in a few dims
    return np.ascontiguousarray(Xc), np.ascontiguousarray(Xs)


def compare_base_learners(legacy_bls, new_bls):
    assert len(legacy_bls) == len(new_bls), \
        f"L mismatch: {len(legacy_bls)} vs {len(new_bls)}"
    for k, (a, b) in enumerate(zip(legacy_bls, new_bls)):
        assert np.array_equal(a.subspace, b.subspace), f"subspace mismatch at learner {k}"
        assert np.array_equal(a.learner.w, b.learner.w), f"w mismatch at learner {k}"
        assert a.learner.b == b.learner.b, f"bias mismatch at learner {k}"


def compare_records(rec_a, rec_b):
    assert len(rec_a) == len(rec_b), "number of d_sub records differs"
    for ra, rb in zip(rec_a, rec_b):
        assert ra["d_sub"] == rb["d_sub"], f"d_sub differs: {ra} vs {rb}"
        assert ra["num_base_learners"] == rb["num_base_learners"], \
            f"L differs: {ra} vs {rb}"
        assert ra["oob_error"] == rb["oob_error"], \
            f"OOB differs: {ra['oob_error']} vs {rb['oob_error']}"


def run_case(name, L, d_sub, n_jobs):
    Xc, Xs = make_paired_data()
    Xc_test, Xs_test = make_paired_data(n=150, seed=99)
    X_test = np.concatenate([Xc_test, Xs_test])
    y_test = np.concatenate([-np.ones(len(Xc_test), int), np.ones(len(Xs_test), int)])

    seeds = dict(seed=12345, seed_subspaces=111, seed_bootstrap=222)

    # --- Legacy (always sequential) ---
    t0 = time.perf_counter()
    legacy = LegacyTrainer(Xc=Xc, Xs=Xs, L=L, d_sub=d_sub, verbose=0, **seeds)
    legacy_ens, legacy_rec = legacy.train()
    t_legacy = time.perf_counter() - t0

    # --- New, via the sklearn-style class ---
    t0 = time.perf_counter()
    clf = FldEnsembleClassifier(
        L=None if L == "automatic" else L,
        d_sub=None if d_sub == "automatic" else d_sub,
        random_state=12345, seed_subspaces=111, seed_bootstrap=222,
        n_jobs=n_jobs, matlab_compat=True, verbose=0,
    )
    X_tr = np.concatenate([Xc, Xs])
    y_tr = np.concatenate([-np.ones(len(Xc), int), np.ones(len(Xs), int)])
    clf.fit(X_tr, y_tr)
    t_new = time.perf_counter() - t0

    # --- Bit-identity checks ---
    compare_records(legacy_rec, clf.training_records_)
    compare_base_learners(legacy_ens.base_learners, clf.base_learners_)
    assert clf.d_sub_ == legacy_ens.d_sub, "optimal d_sub differs"

    # Predictions must match exactly (same tie-break seed 6020)
    assert np.array_equal(legacy_ens.predict(X_test), clf.predict(X_test)), \
        "predictions differ"
    assert np.allclose(legacy_ens.predict_confidence(X_test),
                       clf.decision_function(X_test)), "confidence differs"

    # --- sklearn API smoke test + backward-compat shim ---
    acc = clf.score(X_test, y_test)
    new_ens, new_rec = NewTrainer(Xc=Xc, Xs=Xs, L=L, d_sub=d_sub, verbose=0,
                                  **seeds).train()
    assert np.array_equal(new_ens.predict(X_test), clf.predict(X_test)), \
        "shim diverges from class"

    speedup = t_legacy / t_new if t_new > 0 else float("nan")
    print(f"[OK] {name:<34} L={clf.num_base_learners:<4} "
          f"d_sub={clf.d_sub_:<5} acc={acc:.3f}  "
          f"legacy {t_legacy:6.2f}s  new {t_new:6.2f}s  ({speedup:4.1f}x)")


if __name__ == "__main__":
    run_case("fixed d_sub, fixed L, seq",      L=30,          d_sub=20, n_jobs=1)
    run_case("fixed d_sub, auto L, seq",       L="automatic", d_sub=20, n_jobs=1)
    run_case("auto d_sub, auto L, seq",        L="automatic", d_sub="automatic", n_jobs=1)
    run_case("fixed d_sub, fixed L, parallel", L=30,          d_sub=20, n_jobs=4)
    run_case("auto d_sub, auto L, parallel",   L="automatic", d_sub="automatic", n_jobs=4)
    print("\nAll bit-identity checks passed.")
