# test/conftest.py
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
    if not dir_path.exists():
        return
    _make_package(pkg_name, dir_path)
    for name in _LOAD_ORDER:
        p = dir_path / f"{name}.py"
        if p.exists():
            _load_module(f"{pkg_name}.{name}", p, pkg_name)
    # bind the public names that the real __init__.py would expose
    ec = sys.modules[pkg_name]
    fet = sys.modules.get(f"{pkg_name}.fld_ensemble_trainer")
    enc = sys.modules.get(f"{pkg_name}.ensemble_classifier")
    if fet is not None:
        ec.FldEnsembleTrainer = fet.FldEnsembleTrainer
        if hasattr(fet, "FldEnsembleClassifier"):
            ec.FldEnsembleClassifier = fet.FldEnsembleClassifier
    if enc is not None:
        ec.EnsembleClassifier = enc.EnsembleClassifier


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
    _load_subpackage("sealwatch.ensemble_classifier_rework", pkg_dir / "ensemble_classifier_rework")


# Only stub if the real (built) package isn't available
try:
    import sealwatch  # noqa: F401
    import sealwatch.ensemble_classifier  # noqa: F401
except Exception:
    _bootstrap()