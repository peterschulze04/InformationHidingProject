"""Compare legacy vs reworked FldEnsembleTrainer: time, memory, solve speedup."""

import cProfile
import gc
import importlib.util
import io
import logging
import pathlib
import pstats
import sys
import threading
import time
import types

import numpy as np

try:
    import psutil
    _PROC = psutil.Process()
except ImportError:
    psutil = None
    _PROC = None

# --------------------------------------------------------------------------- #
# Bootstrap: load both packages without the heavy sealwatch __init__ / Rust
# --------------------------------------------------------------------------- #
_LOAD_ORDER = ["fld", "base_learner", "out_of_bag_error_estimates",
               "subspace_dimensionality_search", "ensemble_classifier",
               "fld_ensemble_trainer"]


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
    if not dir_path.exists():
        return
    _make_package(pkg_name, dir_path)
    for name in _LOAD_ORDER:
        p = dir_path / f"{name}.py"
        if p.exists():
            _load_module(f"{pkg_name}.{name}", p, pkg_name)


def _bootstrap():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    pkg_dir = repo_root / "sealwatch"
    tools_dir = pkg_dir / "tools"
    _make_package("sealwatch", pkg_dir)
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
    _load_subpackage("sealwatch.ensemble_classifier", pkg_dir / "ensemble_classifier")
    _load_subpackage("sealwatch.ensemble_classifier_rework",
                     pkg_dir / "ensemble_classifier_rework")


_bootstrap()
from sealwatch.ensemble_classifier.fld_ensemble_trainer import (  # noqa: E402
    FldEnsembleTrainer as LegacyTrainer,
)
from sealwatch.ensemble_classifier_rework.fld_ensemble_trainer import (  # noqa: E402
    FldEnsembleClassifier as ReworkClf,
)

SEEDS = dict(seed=12345, seed_subspaces=111, seed_bootstrap=222)


def make_paired_data(n, d, n_signal=8, shift=0.35, seed=0):
    rng = np.random.RandomState(seed)
    Xc = rng.randn(n, d).astype(np.float64)
    Xs = Xc + rng.randn(n, d).astype(np.float64) * 0.1
    Xs[:, rng.choice(d, size=min(n_signal, d), replace=False)] += shift
    return np.ascontiguousarray(Xc), np.ascontiguousarray(Xs)


class PeakRSSSampler(threading.Thread):
    def __init__(self, interval=0.01):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()
        self.peak = 0

    def run(self):
        if _PROC is None:
            return
        while not self._stop_event.is_set():
            self.peak = max(self.peak, _PROC.memory_info().rss)
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join()
        return self.peak


def _rss():
    return _PROC.memory_info().rss if _PROC else 0


def time_legacy(Xc, Xs, L, d_sub):
    gc.collect()
    rss0 = _rss()
    sampler = PeakRSSSampler(); sampler.start()
    t0 = time.perf_counter()
    trainer = LegacyTrainer(Xc=Xc, Xs=Xs, L=L, d_sub=d_sub, verbose=0, **SEEDS)
    ens, _ = trainer.train()
    elapsed = time.perf_counter() - t0
    peak = sampler.stop()
    retained = _rss() - rss0           # trainer still holds self.Xc/self.Xs
    n = ens.num_base_learners
    del trainer, ens
    return elapsed, max(0, peak - rss0), max(0, retained), n


def time_rework(Xc, Xs, L, d_sub):
    X = np.concatenate([Xc, Xs], axis=0)
    y = np.concatenate([-np.ones(len(Xc), int), np.ones(len(Xs), int)])
    gc.collect()
    rss0 = _rss()
    sampler = PeakRSSSampler(); sampler.start()
    t0 = time.perf_counter()
    clf = ReworkClf(
        L=None if L == "automatic" else L,
        d_sub=None if d_sub == "automatic" else d_sub,
        random_state=SEEDS["seed"], seed_subspaces=SEEDS["seed_subspaces"],
        seed_bootstrap=SEEDS["seed_bootstrap"], n_jobs=1, matlab_compat=True, verbose=0,
    )
    clf.fit(X, y)
    elapsed = time.perf_counter() - t0
    peak = sampler.stop()
    retained = _rss() - rss0           # clf keeps no training data
    n = clf.num_base_learners
    del clf, X, y
    return elapsed, max(0, peak - rss0), max(0, retained), n


def run_compare(repeats=3):
    configs = [
        ("N=600  D=1024  L=100",        600, 1024, 100, 100),
        ("N=600  D=1024  auto L",       600, 1024, "automatic", 100),
        ("N=600  D=1024  auto d_sub+L", 600, 1024, "automatic", "automatic"),
        ("N=600  D=2048  d_sub=512",    600, 2048, 50, 512),   # solve-heavy
    ]
    mb = 1e6
    print(f"{'config':<30} {'leg(s)':>8} {'rew(s)':>8} {'x':>5} "
          f"{'leg peak':>9} {'rew peak':>9} {'leg kept':>9} {'rew kept':>9} {'L l/r':>9}")
    print("-" * 105)
    for label, n, d, L, d_sub in configs:
        Xc, Xs = make_paired_data(n, d)
        lt = lp = lk = rt = rp = rk = 0.0
        ln = rn = 0
        lts, rts = [], []
        for _ in range(repeats):
            e, p, k, ln = time_legacy(Xc, Xs, L, d_sub); lts.append(e); lp, lk = p, k
            e, p, k, rn = time_rework(Xc, Xs, L, d_sub); rts.append(e); rp, rk = p, k
        lt, rt = np.median(lts), np.median(rts)
        sp = lt / rt if rt > 0 else float("nan")
        print(f"{label:<30} {lt:>8.2f} {rt:>8.2f} {sp:>5.1f} "
              f"{lp/mb:>8.1f}M {rp/mb:>8.1f}M {lk/mb:>8.1f}M {rk/mb:>8.1f}M "
              f"{ln:>4}/{rn:<4}")


def profile_solve(n=600, d=2048, d_sub=512, L=50):
    """Show top self-time functions for both -> see solve before/after."""
    Xc, Xs = make_paired_data(n, d)
    X = np.concatenate([Xc, Xs]); y = np.concatenate([-np.ones(len(Xc), int), np.ones(len(Xs), int)])

    for name, fn in [
        ("LEGACY", lambda: LegacyTrainer(Xc=Xc, Xs=Xs, L=L, d_sub=d_sub, verbose=0, **SEEDS).train()),
        ("REWORK", lambda: ReworkClf(L=L, d_sub=d_sub, random_state=SEEDS["seed"],
                                     seed_subspaces=SEEDS["seed_subspaces"],
                                     seed_bootstrap=SEEDS["seed_bootstrap"], verbose=0).fit(X, y)),
    ]:
        pr = cProfile.Profile(); pr.enable(); fn(); pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(6)
        print(f"\n=== {name} top self-time (N={n} D={d} d_sub={d_sub} L={L}) ===")
        print(s.getvalue())



def time_rework_dtype(X, y, L, d_sub, dtype):
    gc.collect()
    rss0 = _rss()
    sampler = PeakRSSSampler(); sampler.start()
    t0 = time.perf_counter()
    clf = ReworkClf(L=L, d_sub=d_sub, random_state=12345,
                    seed_subspaces=111, seed_bootstrap=222,
                    matlab_compat=True, dtype=dtype, verbose=0)
    clf.fit(X, y)
    elapsed = time.perf_counter() - t0
    peak = sampler.stop()
    del clf
    return elapsed, max(0, peak - rss0)


def mem_compare():
    # large D so the feature matrices dominate over baseline noise
    configs = [
        ("N=1500 D=6000  d_sub=300 L=15", 1500, 6000, 15, 300),
        ("N=2000 D=8000  d_sub=400 L=15", 2000, 8000, 15, 400),
    ]
    print(f"\n=== float64 vs float32 (rework, fit adds = peak - baseline) ===\n")
    print(f"{'config':<32} {'dtype':>8} {'input MB':>9} {'fit adds MB':>12} {'time s':>8}")
    print("-" * 74)
    for label, n, d, L, d_sub in configs:
        Xc, Xs = make_paired_data(n, d)
        for dt in (np.float64, np.float32):
            # caller data already in target dtype -> whole chain stays that dtype
            X = np.concatenate([Xc, Xs]).astype(dt)
            y = np.concatenate([-np.ones(n, int), np.ones(n, int)])
            t, added = time_rework_dtype(X, y, L, d_sub, dt)
            print(f"{label:<32} {np.dtype(dt).name:>8} {X.nbytes/1e6:>9.1f} "
                  f"{added/1e6:>12.1f} {t:>8.2f}")
            del X, y
        del Xc, Xs



if __name__ == "__main__":
    if psutil is None:
        print("Note: psutil not installed -> memory columns will be 0.\n")
    print("=== legacy vs rework (median of 3) ===\n")
    run_compare()
    profile_solve()
    mem_compare()