"""
Baseline benchmark for the legacy FldEnsembleTrainer.

Measures (a) training wall-clock time scaling over N / D / L,
(b) peak resident memory added during training, and
(c) a cProfile hotspot breakdown for one representative config.

Run from the repo root:  python experiments/benchmark_fld_ensemble.py
"""

import cProfile
import io
import pstats
import threading
import time

import numpy as np
import sys, pathlib
import importlib.util
import logging
import pathlib
import sys
import types


def _bootstrap_ensemble():
    """Load sealwatch.ensemble_classifier without triggering the heavy package
    __init__ (which pulls in the Rust backend, torch, jpeglib, ...)."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]   # ...\project\sealwatch
    pkg_dir = repo_root / "sealwatch"
    ec_dir = pkg_dir / "ensemble_classifier"
    tools_dir = pkg_dir / "tools"

    def load(name, path, package=None):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        if package is not None:
            mod.__package__ = package
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # 1) fake top-level package (no __init__ executed)
    if "sealwatch" not in sys.modules:
        sw = types.ModuleType("sealwatch")
        sw.__path__ = [str(pkg_dir)]
        sys.modules["sealwatch"] = sw

    # 2) minimal sealwatch.tools: only randperm_naive + setup_custom_logger
    if "sealwatch.tools" not in sys.modules:
        tools = types.ModuleType("sealwatch.tools")
        tools.__path__ = [str(tools_dir)]
        sys.modules["sealwatch.tools"] = tools

        matlab = load("sealwatch.tools.matlab", tools_dir / "matlab.py",
                      package="sealwatch.tools")
        tools.matlab = matlab

        def setup_custom_logger(name):
            logger = logging.getLogger(str(name))
            if not logger.handlers:
                h = logging.StreamHandler(sys.stdout)
                h.setFormatter(logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
                logger.addHandler(h)
            logger.setLevel(logging.DEBUG)
            return logger
        tools.setup_custom_logger = setup_custom_logger
        sys.modules["sealwatch"].tools = tools

    # 3) ensemble_classifier package shim (no __init__ -> skips helpers/h5py/pandas)
    if "sealwatch.ensemble_classifier" not in sys.modules:
        ec = types.ModuleType("sealwatch.ensemble_classifier")
        ec.__path__ = [str(ec_dir)]
        sys.modules["sealwatch.ensemble_classifier"] = ec

    # 4) load the submodules in dependency order
    base = "sealwatch.ensemble_classifier"
    for mod_name in ("fld", "base_learner", "out_of_bag_error_estimates",
                     "subspace_dimensionality_search", "ensemble_classifier",
                     "fld_ensemble_trainer"):
        load(f"{base}.{mod_name}", ec_dir / f"{mod_name}.py", package=base)


_bootstrap_ensemble()
from sealwatch.ensemble_classifier.fld_ensemble_trainer import FldEnsembleTrainer  # noqa: E402

try:
    import psutil
    _PROC = psutil.Process()
except ImportError:
    psutil = None
    _PROC = None


def make_paired_data(n, d, n_signal=8, shift=0.35, seed=0):
    """Synthetic paired cover/stego features with weak signal in a few dims."""
    rng = np.random.RandomState(seed)
    Xc = rng.randn(n, d).astype(np.float64)
    Xs = Xc + rng.randn(n, d).astype(np.float64) * 0.1
    signal_dims = rng.choice(d, size=min(n_signal, d), replace=False)
    Xs[:, signal_dims] += shift
    return np.ascontiguousarray(Xc), np.ascontiguousarray(Xs)


class PeakRSSSampler(threading.Thread):
    """Background thread that records peak RSS while training runs."""

    def __init__(self, interval=0.01):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()   # not _stop -> avoid shadowing Thread internals
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


def time_once(Xc, Xs, L, d_sub):
    """Train once; return (seconds, added_peak_rss_bytes, num_base_learners)."""
    rss_before = _PROC.memory_info().rss if _PROC else 0
    sampler = PeakRSSSampler()
    sampler.start()

    t0 = time.perf_counter()
    trainer = FldEnsembleTrainer(
        Xc=Xc, Xs=Xs, L=L, d_sub=d_sub, verbose=0,
        seed=12345, seed_subspaces=111, seed_bootstrap=222,
    )
    ensemble, _ = trainer.train()
    elapsed = time.perf_counter() - t0

    peak = sampler.stop()
    added = max(0, peak - rss_before) if _PROC else 0
    return elapsed, added, ensemble.num_base_learners


def run_grid(repeats=3):
    # (label, N per class, D, L, d_sub)
    configs = [
        ("N=300  D=256   L=50  fixed",  300, 256,  50, 50),
        ("N=600  D=256   L=50  fixed",  600, 256,  50, 50),
        ("N=300  D=1024  L=50  fixed",  300, 1024, 50, 100),
        ("N=600  D=1024  L=50  fixed",  600, 1024, 50, 100),
        ("N=600  D=1024  L=100 fixed",  600, 1024, 100, 100),
        ("N=600  D=1024  auto L",       600, 1024, "automatic", 100),
        ("N=600  D=1024  auto d_sub+L", 600, 1024, "automatic", "automatic"),
    ]

    print(f"{'config':<30} {'time (s)':>10} {'mem (MB)':>10} "
          f"{'data (MB)':>10} {'L':>5}")
    print("-" * 70)
    for label, n, d, L, d_sub in configs:
        Xc, Xs = make_paired_data(n, d)
        data_mb = (Xc.nbytes + Xs.nbytes) / 1e6

        times, mems, Ls = [], [], []
        for _ in range(repeats):
            t, mem, nbl = time_once(Xc, Xs, L, d_sub)
            times.append(t)
            mems.append(mem)
            Ls.append(nbl)

        t_med = np.median(times)
        mem_med = np.median(mems) / 1e6 if _PROC else float("nan")
        print(f"{label:<30} {t_med:>10.2f} {mem_med:>10.1f} "
              f"{data_mb:>10.1f} {int(np.median(Ls)):>5}")


def profile_one():
    """cProfile breakdown for one mid-size config -> where does time go?"""
    Xc, Xs = make_paired_data(600, 1024)
    profiler = cProfile.Profile()
    profiler.enable()
    trainer = FldEnsembleTrainer(
        Xc=Xc, Xs=Xs, L=100, d_sub=100, verbose=0,
        seed=12345, seed_subspaces=111, seed_bootstrap=222,
    )
    trainer.train()
    profiler.disable()

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    stats.print_stats(15)
    print("\n=== cProfile (top 15 by cumulative time, N=600 D=1024 L=100) ===")
    print(s.getvalue())


if __name__ == "__main__":
    if psutil is None:
        print("Note: psutil not installed -> memory column will be NaN. "
              "Install with `pip install psutil` for memory numbers.\n")
    print("=== Timing & memory grid (median of 3 runs) ===\n")
    run_grid(repeats=3)
    profile_one()