from sealwatch.ensemble_classifier_rework.fld_ensemble_trainer import FldEnsembleClassifier
import numpy as np
import importlib.util
import logging
import pathlib
import sys
import types

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
    repo_root = pathlib.Path(__file__).resolve().parents[1]   # ...\project\sealwatch
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
    _load_subpackage("sealwatch.ensemble_classifier_rework",
                     pkg_dir / "ensemble_classifier_rework")


_bootstrap()


def make_paired_data(n, d, n_signal=8, shift=0.35, seed=0):
    rng = np.random.RandomState(seed)
    Xc = rng.randn(n, d).astype(np.float64)
    Xs = Xc + rng.randn(n, d) * 0.1
    Xs[:, rng.choice(d, n_signal, replace=False)] += shift
    return Xc, Xs


Xc, Xs = make_paired_data(600, 1024)
X = np.concatenate([Xc, Xs])
y = np.concatenate([-np.ones(len(Xc), int), np.ones(len(Xs), int)])
Xc_t, Xs_t = make_paired_data(300, 1024, seed=99)
Xt = np.concatenate([Xc_t, Xs_t])
yt = np.concatenate([-np.ones(len(Xc_t), int), np.ones(len(Xs_t), int)])


def train(dtype):
    clf = FldEnsembleClassifier(L=100, d_sub=200, random_state=12345,
                                seed_subspaces=111, seed_bootstrap=222,
                                matlab_compat=True, dtype=dtype, verbose=0)
    return clf.fit(X, y)


clf64 = train(np.float64)
clf32 = train(np.float32)

oob64 = np.array([r["oob_error"] for r in clf64.training_records_])
oob32 = np.array([r["oob_error"] for r in clf32.training_records_])
print("max |OOB64 - OOB32| =", np.max(np.abs(oob64 - oob32)))
print("prediction agreement =", (clf64.predict(Xt) == clf32.predict(Xt)).mean())
print("acc64 =", round(clf64.score(Xt, yt), 4), " acc32 =", round(clf32.score(Xt, yt), 4))
