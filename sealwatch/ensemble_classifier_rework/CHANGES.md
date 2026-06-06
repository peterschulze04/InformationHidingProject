# FLD Ensemble — Änderungen `ensemble_classifier` → `ensemble_classifier_rework`

Dieses Dokument listet **alle** Unterschiede zwischen der ursprünglichen Implementierung
(`sealwatch/ensemble_classifier/`) und der überarbeiteten Version
(`sealwatch/ensemble_classifier_rework/`) auf — Optimierungen, Refactorings und die
neue Schnittstelle.

Leitprinzip: **gleicher Algorithmus, gleiches Ergebnis.** Der Standardmodus
(`matlab_compat=True`) reproduziert die Trainings-Trajektorie der Referenz exakt
(`test/test_fld_ensemble_classifier.py`, 4/4 grün gegen die Matlab-`.mat`-Referenz).
Beide Pakete existieren parallel; das Legacy-Paket bleibt als Baseline unangetastet.

---

## 1 · Datei-Übersicht

| Datei | Status | Kern der Änderung |
|---|---|---|
| `__init__.py` | **unverändert** | — |
| `ensemble_classifier.py` | **unverändert** | Standalone `EnsembleClassifier` (Loop-Vote). Von der neuen Klasse nicht mehr verwendet, nur für Rückwärtskompatibilität erhalten. |
| `helpers.py` | **unverändert** | HDF5-/CSV-Datenladen |
| `out_of_bag_error_estimates.py` | **unverändert** | OOB-Zustandsobjekt |
| `subspace_dimensionality_search.py` | **unverändert** | `d_sub`-Gittersuche |
| `base_learner.py` | **nur Import** | `from sealwatch.ensemble_classifier.fld …` → `from .fld …`. Logik (inkl. `_fast_fancy_indexing`) identisch. |
| `fld.py` | **geändert** (234→229 Z.) | Cholesky-Solve, vektorisiertes `_find_threshold`, dtype-bewusster Ridge |
| `fld_ensemble_trainer.py` | **neu geschrieben** (389→407 Z.) | sklearn-Interface, Backward-Compat-Shim, Zero-Copy-Split, vektorisierte Inferenz, Speicher-Freigabe |

Der gesamte inhaltliche Unterschied steckt also in **zwei Dateien**: `fld.py` und
`fld_ensemble_trainer.py`.

---

## 2 · Schnittstelle (Refactor)

**Vorher:** Trainer und Modell waren getrennte Klassen. Eingabe als zwei Arrays
`Xc`/`Xs`, Training über `train()`, das `(model, records)` zurückgibt:

```python
trainer = FldEnsembleTrainer(Xc=Xc, Xs=Xs, seed=42)
ensemble, records = trainer.train()
confidence = ensemble.predict_confidence(X)
```

**Nachher:** eine scikit-learn-konforme Klasse `FldEnsembleClassifier(BaseEstimator,
ClassifierMixin)` mit `fit(X, y)` / `predict` / `decision_function` / `score`,
`clone`-fähig und in `Pipeline`/`GridSearchCV` einsetzbar:

```python
clf = FldEnsembleClassifier(random_state=42).fit(X, y)
labels     = clf.predict(X)
confidence = clf.decision_function(X)
```

**Backward-Compat-Shim:** `FldEnsembleTrainer` existiert weiterhin — jetzt als
Unterklasse von `FldEnsembleClassifier`. Sie akzeptiert die alten `Xc`/`Xs`-Argumente
und stellt `train() -> (self, training_records)` bereit, sodass bestehender Code
unverändert läuft.

Weitere Refactor-Details:
- **Keine Arbeit im `__init__`** (sklearn-Konvention): Konstruktor speichert nur
  Argumente, die gesamte RNG-/Setup-Logik wandert nach `fit`. Die Reihenfolge der
  RNG-Ableitung (Haupt-RNG → `seed_subspaces` → `seed_bootstrap`) bleibt exakt
  erhalten → identische Seeds.
- Standard-sklearn-Validierung (`check_X_y`, `check_array`, `check_is_fitted`,
  `unique_labels`) und gesetzte Attribute (`classes_`, `n_features_in_`, `d_sub_`,
  `base_learners_`, `training_records_`).

---

## 3 · Trainings-Geschwindigkeit

### 3.1 Cholesky- statt LU-Solve (`fld.py`)
Die Within-Class-Scatter-Matrix `sigma_cs` ist symmetrisch positiv definit
(Summe zweier Kovarianzen + Stabilisierungs-ε).

- **Vorher:** `np.linalg.solve(sigma_cs, mu)` — generische LU-Zerlegung, O(⅔ n³).
- **Nachher:** `scipy.linalg.solve(sigma_cs, mu, assume_a="pos")` — Cholesky, O(⅓ n³).
  Das ist genau das Verfahren, das auch Matlabs `mldivide` für SPD-Systeme nutzt.

**Effekt:** Solve-Mikrobenchmark (d_sub=512, 50 Calls) **0.94 s → 0.41 s ≈ 2.3×**.
Numerik verschiebt sich nur auf ~1e-15-Ebene; die `d_sub`/`L`-Such-Entscheidungen
bleiben identisch.

### 3.2 `_find_threshold` vektorisiert (`fld.py`)
Die Schwellenwertsuche minimiert `(P_FA + P_MD)/2` über alle Sortier-Positionen.

- **Vorher:** Python-`for`-Schleife über alle Samples (zwei Vorzeichen-Fälle), die pro
  Schritt den Fehler inkrementell fortschreibt.
- **Nachher:** vektorisierter `cumsum`-Sweep über die sortierten Scores. Die beiden
  Fehlerkurven werden als `error1 = num_covers − c + s` und `error2 = num_stegos +
  c − s` geschlossen berechnet (`c`/`s` = kumulative Cover-/Stego-Zahl), interleaved
  und per `argmin` ausgewertet (erste Fundstelle = identische Tie-Break-Reihenfolge
  „Fall 1 vor Fall 2, kleinster Index").

**Effekt:** Funktion selbst **~4.4× schneller** (0.91 → 0.21 ms bei N=600 d_sub=512);
**bit-identisch** (0 Abweichungen über 500 Zufallsläufe in Gewichten und Bias).

### 3.3 OOB-Indizes: `bincount` statt `setdiff1d` (`fld_ensemble_trainer.py`)
- **Vorher:** `np.setdiff1d(np.arange(n), train_indices)` — O(n log n) pro Lerner.
- **Nachher:** `np.where(np.bincount(train_indices, minlength=n) == 0)[0]` — O(n),
  **gleiche** (sortierte) Ausgabe. Bit-identisch.

### 3.4 In-place Diagonal-Stabilisator (`fld.py`)
- **Vorher:** `sigma_cs = sigma_cs + 1e-10 * np.eye(d)` — allokiert pro Lerner eine
  volle d×d-Identitätsmatrix.
- **Nachher:** `sigma_cs.flat[::d+1] += ridge` (siehe §6 für `ridge`) — In-place auf
  die Diagonale, ohne d×d-Allokation. In float64 exakt äquivalent → bit-identisch.

**Gemessenes Gesamt-Training** (legacy vs. rework, `bench/bench_comparison.py`):
**1.4–1.9×** schneller. Bewusst moderat — der Großteil der Trainingszeit steckt in
intrinsischem BLAS (Scatter-`gemm`, Cholesky-Solve), das bereits nahe optimal ist.

---

## 4 · Speicher

### 4.1 Keine redundante Daten-Kopie, keine Datenhaltung (`fld_ensemble_trainer.py`)
- **Vorher:** `self.Xc = Xc.astype(np.float64)` kopiert **immer** die volle
  Feature-Matrix (auch wenn schon float64) und hält sie für die gesamte
  Objekt-Lebensdauer auf `self`.
- **Nachher:** `np.ascontiguousarray(..., dtype=…)` kopiert nur bei Bedarf; nach `fit`
  werden die Trainingsdaten **freigegeben** (nicht auf `self` gehalten).

**Effekt:** Nach dem Training hält Legacy den ganzen Datensatz (10–20 MB im Test),
rework ~0 MB. Bei SRM (34 671 Dim.) ≈ **GB-Unterschied**.

### 4.2 Zero-Copy Cover/Stego-Split (`_split_classes`, `fld_ensemble_trainer.py`)
- **Vorher (auch im ersten rework):** `X[y == neg]` per Boolean-Indexing erzeugt eine
  **Kopie** jeder Klasse und hält sie die ganze fit-Dauer — zusätzlich zum `X` des
  Callers → Peak ≈ Input + 1× Daten-Kopie.
- **Nachher:** `_split_classes` gibt zero-copy **Views** zurück, wenn jede Klasse ein
  zusammenhängender Zeilenblock eines C-contiguous `X` mit passendem dtype ist (der
  Normalfall — der Shim liefert Cover-dann-Stego float64). `Xc`/`Xs` werden im Training
  nur gelesen → sicher und bit-identisch. Beliebige Label-Reihenfolge / dtype-Wechsel
  → Fallback auf Kopie.

**Effekt** (A/B, identische float64-Daten, Peak-RSS-Anstieg):

| Config | Input | Copy-Split | View-Split | Reduktion |
|---|--:|--:|--:|--:|
| N=600  D=4096 | 39.3 MB | +82.5 MB | +32.0 MB | **2.6×** (−51 MB) |
| N=1500 D=6000 | 144 MB | +175.1 MB | +32.8 MB | **5.3×** (−142 MB) |

Skaliert mit der Datengröße (eingesparte Kopie ≈ 1× Daten) → bei SRM-Skala GB.

---

## 5 · Inferenz (größter Speed-Win)

**Vorher:** `EnsembleClassifier.predict` / `predict_confidence` summieren in einer
**Python-Schleife über alle L Base-Learner**, jeder mit eigener Spalten-Projektion
`X[:, subspace] @ w − b`.

**Nachher:** `_vote` expandiert alle Gewichte einmal in eine dichte `(n_features, L)`-
Matrix `W` (jede Spalte = Lerner-Gewicht, zurück in den vollen Feature-Raum gestreut,
0 außerhalb seines Subspace) und stimmt mit **einem** Matmul ab:
`np.sign(X @ W − b).sum(axis=1)`. `W` wird **lazy beim ersten `predict` gebaut und
gecached** (Speicher nur bei tatsächlicher Inferenz), dtype-treu zum Modell.

Korrektheit: `X @ W[:, i]` reproduziert `X[:, subspace] @ w` exakt (die Nullen tragen
0 bei), daher **identische Votes / Labels / `decision_function`**.

**Effekt** (`run_inference_compare` im Bench):

| Config | Legacy (Loop) | Rework (Matmul) | Speedup | identisch |
|---|--:|--:|--:|:--:|
| D=1024 L=100 n_test=2000 | 551 ms | 6.9 ms | **79×** | ✓ |
| D=2048 L=200 n_test=4000 | 5001 ms | 39 ms | **127×** | ✓ |

Der Faktor wächst mit `L` und `n_test`. Betrifft **nur** die Inferenz — der
Trainingspfad und das Matlab-Gate bleiben unberührt.

---

## 6 · float32-Rechenpfad (Nicht-Compat)

`dtype=np.float32` (nur bei `matlab_compat=False`) rechnet Scatter und Solve in
float32 → halber Speicher der Feature-Matrizen **und** schnelleres Training.

**Behobener latenter Bug (dtype-bewusster Ridge, `fld.py`):** Der Stabilisator-Wert
`1e-10` liegt **unter** dem float32-Maschinen-Epsilon (~1.2e-7) und stabilisiert die
oft schlecht konditionierte Scatter-Matrix in float32 gar nicht → die float32-Cholesky
erzeugt Denormals und wird bei großem `d_sub` **~6× langsamer** (d_sub=1024: 30 s
statt 5 s). Der Ridge ist jetzt dtype-bewusst:

```python
ridge = 1e-10                                    # float64: exakt wie Legacy → bit-identisch
ridge = 1e-6 * (np.trace(sigma_cs) / d)          # float32: skalierungsrobuster Ridge
```

**Effekt:** float32 **1.67–1.73× schneller** als float64 über *alle* `d_sub` und
halber Speicher; skalierungsrobust (identisches Timing bei Daten-Skala ×1 und ×1000).
float64/Compat bleibt durch das exakte `1e-10` bit-identisch.

> ⚠️ **Wichtige Einschränkung (nachträglich auf echten Daten gemessen):** float32 ist
> **nur bei *festem* `d_sub` unbedenklich.** Bei der **automatischen `d_sub`-Suche**
> verfälscht die geringere Präzision die OOB-Fehlerschätzungen, die die Modellselektion
> steuern → der Compass-Search wählt ein zu kleines `d_sub` (auf den Matlab-Tutorial-
> Daten z. B. 103 statt 274) und die Detektionsfehlerrate `P_E` verschlechtert sich um
> **~0.02 (2 Prozentpunkte)**. Mein früheres „Genauigkeit ~gleich" galt nur für
> synthetische Zufallsdaten (nahe Zufallsniveau) bzw. festes `d_sub` und lässt sich
> **nicht** auf die `auto`-Suche verallgemeinern. Für gleiche Sicherheit bei `auto`
> daher **float64** verwenden (siehe `../ensemble_classifier_rework2/README.md`, wo der
> Effekt zerlegt ist). Der sichere Speed-Hebel bei `auto` ist die capped-L-Suche.

---

## 7 · Validierung

- **Matlab-Referenz-Gate:** `test/test_fld_ensemble_classifier.py` trainiert legacy
  **und** rework gegen `tutorial_seed_12345/98765.mat` und prüft `search_d_sub`
  (exakt), `search_L` (exakt) und `search_oob` (`np.isclose`) für jeden Suchschritt.
  **4/4 grün.**
- **Bit-Identität einzeln geprüft:** `_find_threshold` (0/500 Mismatches),
  `bincount`-OOB (gleiche sortierte Ausgabe), View-Split (gleiche Werte),
  vektorisierte Inferenz (identische Votes für float64 *und* float32).
- **Numerik-Hinweis:** Cholesky verschiebt die Gewichte ggü. LU um ~1e-15 (näher an
  Matlabs `mldivide`), kippt aber keine Such-Entscheidung — die `d_sub`/`L`-Trajektorie
  ist in allen getesteten Configs identisch (z. B. 268/268).

Reproduktion:
```bash
python -m pytest test/test_fld_ensemble_classifier.py   # Matlab-Bit-Identität
python bench/bench_comparison.py                         # Training + Inferenz + float32
```

---

## 8 · Getestet & verworfen (kein Gewinn)

| Idee | Ergebnis |
|---|---|
| **syrk-Scatter** (`dsyrk` statt `@` für `Xᵀ@X`) | netto leicht **langsamer** — `@` nutzt schon multithreaded BLAS-gemm; `dsyrk` kopiert C→Fortran-Order, Copy-Overhead frisst den Halb-Flop-Vorteil. |
| **„Subspace 1× projizieren"** (statt 4× `_fast_fancy_indexing`/Lerner) | **0.72× (langsamer)** — Train- und OOB-Zeilen sind disjunkt (OOB = nie im Bootstrap), die gemeinsame Spalten-Projektion amortisiert sich nicht. |
| **Single-gemm-Scatter** (`[Xc0;Xs0]ᵀ[Xc0;Xs0]`) | nur ~1.06× gesamt und nicht bit-identisch → nicht eingebaut. |
| **joblib `n_jobs`** (Parallelität) | Overhead macht Sub-Sekunden-Configs langsamer; die `d_sub`-Suche ist ohnehin sequentiell. Bleibt No-op-Platzhalter. |

---

## 9 · Zusammenfassung der Effekte

| Bereich | Hebel | Effekt | Bit-identisch (compat)? |
|---|---|---|:--:|
| Schnittstelle | sklearn `fit/predict/score` + Shim | nutzbar in Pipeline/GridSearchCV | ✓ |
| Training-Zeit | Cholesky + Threshold-Vektorisierung + bincount + In-place-Diag | **1.4–1.9×** (Solve 2.3×) | ✓ |
| Training-Peak | Zero-Copy-Split | **2.6–5.3× niedriger** | ✓ |
| Speicher nach fit | Daten freigeben | Legacy hält Datensatz, rework ~0 (GB bei SRM) | ✓ |
| **Inferenz** | Single-Matmul-Vote | **79–127×** | ✓ |
| float32 (nur festes `d_sub`) | dtype-Ridge | **1.67–1.73×** + halber Speicher | n/a (Nicht-Compat) |

> ⚠️ float32 **nicht** für die automatische `d_sub`-Suche: es verfälscht die
> OOB-Modellselektion → schlechteres `P_E` (~2 Prozentpunkte). Details in §6.
