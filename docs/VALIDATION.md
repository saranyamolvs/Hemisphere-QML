# Software validation, version 0.1.1

Verified in the attached Linux environment using Python 3.12.14. No user
experimental images were available. The Windows interactive crop window still
requires a check on the user's computer.

- **112 unittest cases: 111 passed, one optional Qiskit equivalence check skipped**
  because Qiskit was unavailable. The mandatory NumPy quantum map was checked
  against independent explicit gate matrices and an analytic fidelity result.
- The rebuilt TP notebook was executed top-to-bottom in a real Jupyter kernel
  using 150 private synthetic TIFFs. The check included one human-confirmed
  false alarm and one saved coordinate correction, immediate review application,
  all three training-defined TP categories, ten 16-frame reference sheets,
  external-label POD/SVM/QSVM processing, and reconciled per-test-image
  predictions. It also refreshed the same original mode figure from ±0.009 to
  ±0.007 and back without refitting POD, and rejected mismatched model settings.
  These are software fixtures, not experimental results. Source notebooks
  remain clean of fixture outputs.
- New tests cover review accept/correct/defer behavior, coordinate-specific
  approval on retracking, retained automatic flags, keyboard shortcut conflicts,
  correction of already accepted points, and category review provenance.
  Streaming reference-export checks cover partial pages, coordinate mapping,
  original-file preservation, atomic overwrite, and output protections.
  Test-prediction exports are reconciled with saved metrics and reject
  mismatched labels, unpaired samples, and source aliases. GUI callbacks are
  checked headlessly; the actual Windows GUI and experimental TP tracking
  accuracy require local verification.
- A complete CLI run processed **200 native 768×512, 16-bit TIFF software-test
  frames** with a fixed 640×448 crop, without spatial binning. It fitted
  training-only POD, defined candidate states, tuned both classifiers on the
  validation block, evaluated matched test rows, and saved all **42 PDF/SVG/PNG
  exports at 600 dpi**. All exported paths existed. This is a software check,
  not a result from the wind-tunnel experiment.
- A 20,000-row disk-backed core POD test used 384 pixels per row, 12,000 training
  rows and rank 24. Projections were finite; mode orthogonality error was
  3.22×10^-15. Singular values matched an exact 384×384 covariance calculation
  to within 4.75×10^-11 relative error. This checks large row count and bounded
  core memory; it does **not** benchmark 20,000 full-resolution TIFF files.
- Numerical tests cover training-only fitting, difficult POD spectral decay,
  variance normalization, near-exact reconstruction error, constant training
  features, unsigned chronological counters, malformed labels, nonfinite JSON,
  source/output path collisions and configuration preflight.
- Figure-source checks cover correct spectral frequency under stride, original
  camera-frame ACF lags, no ordinary Welch spectrum for irregular timestamps,
  missing-state composites represented as missing values, and traceable
  confusion/kernel sample identities. Reconstruction and confusion layouts were
  inspected at the actual image aspect ratio and corrected.
- A requested classification with only one training class was correctly
  reported as partial, preserving POD outputs. Configuration errors are no
  longer silently treated as successful classification-free runs.
- The image preview command passed. Earlier POD-only reruns also verified that
  obsolete classifier files are removed on explicit overwrite.
- The project built offline as a Python wheel, installed into an isolated
  target folder, and its module CLI launched outside the source tree.
- The expanded Anaconda notebook was validated with nbformat and executed
  top-to-bottom using nbclient and a real Python Jupyter kernel on a private
  150-frame, 32×48, 16-bit synthetic TIFF fixture. All 13 source code cells
  passed with pilot/full processing and the optional validation study enabled.
  Checks covered both classifier parameter paths, grouped candidate images,
  single-file inspection, CSV-list export, original mode-figure updates, and frozen-label
  learning curves. Source notebooks retain no fixture outputs; full processing
  and the validation study remain disabled by default for normal Run All.
- Notebook helper tests check kernel-interpreter reuse, partial results,
  output-folder protection, path resolution and interruption cleanup. The
  actual Anaconda Navigator / Windows notebook interface remains untested here.
- The saved-mode recoloring path was executed in a real Jupyter kernel using a
  software-fixture POD model. It exported a shared ±0.01 scale without opening
  the raw images or image cache. The saved POD file hash was unchanged. Tests
  check shared symmetric limits, pooled-percentile selection, reported clipping
  fractions, invalid options, and preservation of original result files.
- Class-coverage reporting distinguishes retained frames from the capped
  classifier examples in each partition. Regression tests cover complete and
  incomplete class coverage, explicit single-class warnings, and unchanged
  tuning and predictions when only reporting fields are added. Scores retain
  their original definitions; coverage warnings do not alter the experiment.
- Quantum-scale regression checks confirm that changing the encoding factor
  preserves candidate assignments and RBF inputs. The default factor of one
  reproduces the previous kernels, predictions, and test metrics. Changing
  held-out test values or labels cannot affect validation-selected settings.
- Saved-run study checks cover fixed validation rows, matched model samples,
  actual capped counts, poisoned-test invariance, the original scaler's
  near-constant-feature behavior, and output/source collision protection. The
  study exports training/validation results only and neither reads TIFFs nor
  refits POD or discovers new labels.
- Candidate-viewer checks cover native 16-bit values, shared grayscale limits,
  saved crop/bin consistency, exact frame provenance, moved source folders,
  prediction-only test rows, complete per-candidate file lists, and source alias
  protection. Synthetic gallery, confusion, and learning-curve layouts were
  visually inspected; actual experimental TIFFs must be inspected locally.

Temporary software fixtures and their plots are excluded from the source ZIP.
Full experimental validation still requires checking actual TIFF ordering,
camera timing/dropouts, spatial calibration, POD/state convergence, physical
interpretation and repeat-run generalization. No experimental performance or
quantum advantage is claimed by these software checks.

The original mode figure was regenerated in place with fixed limits ±0.008
through the normal notebook figure-display cell. Existing-result checks used
the selected full-run directory, preserved fitted model bytes, and created no
separate adjusted-image folder. The combined figure notes and scale metadata
are refreshed with the applied display limits.

The TP workflow notebook was executed top-to-bottom on 150 synthetic TIFFs,
including initial coordinate CSVs, tracking, position overlays, three-category
inspection, training-only mean/SD thresholds, and external-label POD/SVM
processing. Annotation click/Enter/save/resume callbacks were exercised with
Matplotlib's headless backend. Translation tests recover the known pixel
positions; missing-structure tests flag uncertain frames and block label export.
Pipeline tests reject any attempted K-means fit when TP labels are used.
Neither actual Windows desktop operation nor experimental TP tracking accuracy
was verified here; those require local inspection and independent landmarks.
