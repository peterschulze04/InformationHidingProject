"""End-to-end security benchmark: detection error ``P_E`` vs. embedding payload.

This experiment is the *security* counterpart to the speed/memory micro-benchmarks
in :mod:`bench`. It answers the question the project asks -- *does the faster
``rework2`` trainer preserve the steganographic security of the original FLD
ensemble?* -- by reproducing the classic steganalysis evaluation of
Kodovsky et al. (IEEE TIFS 2012), Fig. 5: the ensemble detection error as a
function of the relative embedding payload.

Pipeline (fully self-contained, CPU-only, no external dataset)::

    cover JPEG  --tile-->  128x128 sub-images
                --nsF5(alpha)-->  stego sub-images
                --DCTR-->  8000-dim features
                --FLD ensemble-->  P_E on a source-disjoint test split

For every payload we train **two** ensembles on the *same* features:

* ``legacy``  -- :class:`sealwatch.ensemble_classifier.FldEnsembleTrainer`
  (the reference, full ``d_sub``/``L`` search), and
* ``rework2`` -- :class:`sealwatch.ensemble_classifier_rework2.FldEnsembleClassifier`
  (the capped-``L`` search re-implementation).

The two P_E curves are expected to coincide within run-to-run noise, while the
``rework2`` trainer is markedly faster (reported alongside). The absolute error
level is optimistic compared with the paper -- we only have a handful of source
images, tiled and source-disjoint-split -- so the benchmark is read as a
*relative* comparison (rework2 vs. legacy), not a reproduction of BOSSBase
numbers. See ``experiments/README.md``.

Run::

    python -m experiments.security_benchmark            # full run (caches features)
    python -m experiments.security_benchmark --quick    # 3 payloads, smaller cap

Features are cached under ``experiments/_cache`` so re-runs (e.g. to re-plot) are
instant.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import conseal
import jpeglib
import sealwatch as sw
import sealwatch.ensemble_classifier_rework2  # noqa: F401 - register subpackage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "_cache"
FIGURES = HERE / "figures"
COVER_GLOBS = [
    ROOT / "test/assets/cover/jpeg_75_gray",
    ROOT / "test/assets/cover/jpeg_75_color",
]

QF = 75                 # JPEG quality of the cover tiles
TILE = 128              # tile edge in pixels (16x16 DCT blocks -> stable DCTR stats)
PAYLOADS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)   # nsF5 relative change rate (alpha)
SEED = 12345
TEST_FRACTION = 0.30    # fraction of *source images* held out (source-disjoint)


# --------------------------------------------------------------------------- #
# Feature pipeline                                                            #
# --------------------------------------------------------------------------- #
def _try_rust_backend() -> str:
    """Switch sealwatch + conseal to the Rust backend if available (faster)."""
    try:
        sw.set_backend(sw.BACKEND_RUST)
        conseal.set_backend(conseal.BACKEND_RUST)
        return "rust"
    except Exception:
        return "python"


def _to_gray(spatial: np.ndarray) -> np.ndarray:
    """Collapse an ``H x W x C`` spatial image to a single luminance channel."""
    if spatial.shape[-1] == 1:
        return spatial
    # ITU-R BT.601 luma; jpeglib spatial is uint8 RGB.
    rgb = spatial.astype(np.float64)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.round(luma).astype(np.uint8)[..., None]


def _source_images() -> list[tuple[str, np.ndarray]]:
    """Load every cover JPEG as a grayscale spatial image, keyed by a stable id."""
    out: list[tuple[str, np.ndarray]] = []
    for folder in COVER_GLOBS:
        for path in sorted(folder.glob("*.jpg")):
            spatial = jpeglib.read_spatial(str(path)).spatial
            out.append((f"{folder.name}/{path.stem}", _to_gray(spatial)))
    return out


def _tiles(gray: np.ndarray, tile: int = TILE) -> list[np.ndarray]:
    """Cut a grayscale image into non-overlapping ``tile x tile`` sub-images."""
    h, w = gray.shape[:2]
    return [
        gray[i:i + tile, j:j + tile, :]
        for i in range(0, h - tile + 1, tile)
        for j in range(0, w - tile + 1, tile)
    ]


def _dctr_vector(jpeg_path: str) -> np.ndarray:
    """Flatten the DCTR sub-features extracted from a JPEG file into one vector."""
    feats = sw.dctr.extract_from_file(jpeg_path, qf=QF)
    return np.concatenate([np.ravel(v) for v in feats.values()]).astype(np.float64)


def _build_features(work: Path) -> tuple[np.ndarray, dict[float, np.ndarray], np.ndarray]:
    """Compute (and cache) cover + per-payload stego DCTR features.

    :param work: scratch directory for the transient cover/stego JPEG tiles.
    :return: ``(Xc, Xs_by_alpha, image_ids)`` where ``Xc`` is ``[N, 8000]`` cover
        features, ``Xs_by_alpha[alpha]`` the matching stego features, and
        ``image_ids`` the source-image id of every row (for source-disjoint split).
    """
    cover_npz = CACHE / f"cover_tile{TILE}_qf{QF}.npz"
    if cover_npz.exists():
        data = np.load(cover_npz, allow_pickle=True)
        Xc, ids = data["Xc"], data["ids"]
    else:
        rows, ids = [], []
        for img_id, gray in _source_images():
            for k, tile in enumerate(_tiles(gray)):
                cpath = work / f"{img_id.replace('/', '_')}_{k}_cover.jpg"
                jpeglib.from_spatial(tile).write_spatial(str(cpath), qt=QF)
                rows.append(_dctr_vector(str(cpath)))
                ids.append(img_id)
        Xc, ids = np.asarray(rows), np.asarray(ids)
        np.savez_compressed(cover_npz, Xc=Xc, ids=ids)
        print(f"  cached cover features {Xc.shape} -> {cover_npz.name}")

    Xs_by_alpha: dict[float, np.ndarray] = {}
    for alpha in PAYLOADS:
        stego_npz = CACHE / f"stego_a{alpha:.2f}_tile{TILE}_qf{QF}.npz"
        if stego_npz.exists():
            Xs_by_alpha[alpha] = np.load(stego_npz)["Xs"]
            continue
        rows = []
        for row, img_id in enumerate(ids):
            # Re-derive the exact cover tile, embed nsF5 at this payload, re-extract.
            gray = _id_to_gray(img_id)
            tile = _tiles(gray)[_tile_index(ids, row)]
            cpath = work / "c.jpg"
            jpeglib.from_spatial(tile).write_spatial(str(cpath), qt=QF)
            dct = jpeglib.read_dct(str(cpath))
            dct.Y[...] = conseal.nsF5.simulate_single_channel(
                dct.Y, alpha=float(alpha), seed=SEED + row)
            spath = work / "s.jpg"
            dct.write_dct(str(spath))
            rows.append(_dctr_vector(str(spath)))
        Xs = np.asarray(rows)
        np.savez_compressed(stego_npz, Xs=Xs)
        Xs_by_alpha[alpha] = Xs
        print(f"  cached stego  features {Xs.shape} alpha={alpha} -> {stego_npz.name}")
    return Xc, Xs_by_alpha, ids


# Small helpers so the stego loop can recover the exact cover tile by row index.
_SRC_CACHE: dict[str, np.ndarray] | None = None


def _id_to_gray(img_id: str) -> np.ndarray:
    global _SRC_CACHE
    if _SRC_CACHE is None:
        _SRC_CACHE = {k: v for k, v in _source_images()}
    return _SRC_CACHE[img_id]


def _tile_index(ids: np.ndarray, row: int) -> int:
    """Position of ``row`` within its own source image (0-based tile counter)."""
    img_id = ids[row]
    return int(np.sum(ids[:row] == img_id))


# --------------------------------------------------------------------------- #
# Training + evaluation                                                       #
# --------------------------------------------------------------------------- #
def _split_by_image(ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Source-disjoint train/test masks: all tiles of an image share a split."""
    uniq = np.unique(ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * TEST_FRACTION)))
    test_imgs = set(uniq[:n_test].tolist())
    test_mask = np.array([i in test_imgs for i in ids])
    return ~test_mask, test_mask


def _detection_error(predict, cover: np.ndarray, stego: np.ndarray) -> float:
    """``P_E = 0.5 (P_FA + P_MD)`` under equal priors (Kodovsky 2012, Eq. 1)."""
    p_fa = float(np.mean(predict(cover) == +1))
    p_md = float(np.mean(predict(stego) == -1))
    return 0.5 * (p_fa + p_md)


def _train_legacy(Xc_tr, Xs_tr, seed):
    ensemble, _ = sw.ensemble_classifier.FldEnsembleTrainer(
        Xc=Xc_tr, Xs=Xs_tr, seed=seed, verbose=0).train()
    return ensemble.predict


def _train_rework2(Xc_tr, Xs_tr, seed, search_effort):
    X = np.concatenate([Xc_tr, Xs_tr])
    y = np.concatenate([-np.ones(len(Xc_tr), int), np.ones(len(Xs_tr), int)])
    clf = sw.ensemble_classifier_rework2.FldEnsembleClassifier(
        random_state=seed, verbose=0, search_effort=search_effort).fit(X, y)
    return clf.predict


def run(quick: bool = False, search_effort: float = 0.5) -> list[dict]:
    """Execute the full benchmark and write the figure + CSV. Returns the records."""
    CACHE.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    backend = _try_rust_backend()
    payloads = PAYLOADS[:3] if quick else PAYLOADS
    print(f"backend={backend}  payloads={payloads}  search_effort={search_effort}")

    work = CACHE / "_tiles"
    work.mkdir(exist_ok=True)
    Xc, Xs_by_alpha, ids = _build_features(work)
    train_mask, test_mask = _split_by_image(ids, SEED)
    print(f"dataset: {len(Xc)} tiles, {int(train_mask.sum())} train / "
          f"{int(test_mask.sum())} test (source-disjoint), D={Xc.shape[1]}")

    records: list[dict] = []
    for alpha in payloads:
        Xs = Xs_by_alpha[alpha]
        Xc_tr, Xs_tr = Xc[train_mask], Xs[train_mask]
        Xc_te, Xs_te = Xc[test_mask], Xs[test_mask]

        t0 = time.time()
        legacy_predict = _train_legacy(Xc_tr, Xs_tr, SEED)
        t_legacy = time.time() - t0
        pe_legacy = _detection_error(legacy_predict, Xc_te, Xs_te)

        t0 = time.time()
        rework2_predict = _train_rework2(Xc_tr, Xs_tr, SEED, search_effort)
        t_rework2 = time.time() - t0
        pe_rework2 = _detection_error(rework2_predict, Xc_te, Xs_te)

        rec = dict(payload=alpha, pe_legacy=pe_legacy, pe_rework2=pe_rework2,
                   t_legacy=t_legacy, t_rework2=t_rework2,
                   speedup=t_legacy / max(t_rework2, 1e-9))
        records.append(rec)
        print(f"alpha={alpha:.2f}  P_E legacy={pe_legacy:.4f} rework2={pe_rework2:.4f}"
              f"  (dP_E={pe_rework2 - pe_legacy:+.4f})  "
              f"t {t_legacy:.1f}s -> {t_rework2:.1f}s  ({rec['speedup']:.1f}x)")

    _write_csv(records, search_effort)
    _plot(records, backend, search_effort)
    return records


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def _write_csv(records: list[dict], search_effort: float) -> None:
    path = HERE / "security_results.csv"
    cols = ["payload", "pe_legacy", "pe_rework2", "t_legacy", "t_rework2", "speedup"]
    lines = [",".join(cols)]
    for r in records:
        lines.append(",".join(f"{r[c]:.6g}" for c in cols))
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.name}")


def _plot(records: list[dict], backend: str, search_effort: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(skipping plot: {exc})")
        return
    x = [r["payload"] for r in records]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(x, [r["pe_legacy"] for r in records], "o-", label="legacy (full search)")
    ax.plot(x, [r["pe_rework2"] for r in records], "s--",
            label=f"rework2 (effort={search_effort})")
    ax.set_xlabel("relative payload  $\\alpha$  (nsF5 change rate)")
    ax.set_ylabel("detection error  $P_E$")
    ax.set_title("FLD ensemble security vs. payload (DCTR, QF=75)")
    ax.set_ylim(0, 0.55)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIGURES / "security_pe_vs_payload.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="3 payloads only (smoke test)")
    parser.add_argument("--search-effort", type=float, default=0.5,
                        help="rework2 speed/security knob in [0, 1] (default 0.5)")
    args = parser.parse_args()
    run(quick=args.quick, search_effort=args.search_effort)


if __name__ == "__main__":
    main()
