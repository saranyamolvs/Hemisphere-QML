# Software validation, version 0.1.1

Verified in the attached Linux environment using Python 3.12.14. No user
experimental images were available. The Windows interactive crop window still
requires a check on the user's computer.

- **48 unittest cases: 47 passed, one optional Qiskit equivalence check skipped**
  because Qiskit was unavailable. The mandatory NumPy quantum map was checked
  against independent explicit gate matrices and an analytic fidelity result.
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

Temporary software fixtures and their plots are excluded from the source ZIP.
Full experimental validation still requires checking actual TIFF ordering,
camera timing/dropouts, spatial calibration, POD/state convergence, physical
interpretation and repeat-run generalization. No experimental performance or
quantum advantage is claimed by these software checks.
