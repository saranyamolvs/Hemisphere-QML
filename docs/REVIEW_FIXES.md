# Code review fixes in 0.1.1

The review found and repaired the following concrete issues.

1. **Output collisions and early failure:** image-cache and manifest paths could
   alias each other or overwrite an unselected source image. Checks now happen
   before truncation and cover all source frames. CLI settings, crops, splits
   and external labels are checked before replacing prior outputs. Preview and
   crop-save commands cannot write over their input image.
2. **POD numerical stability:** power iterations previously formed a poorly
   conditioned squared operator without an intermediate temporal QR. Both
   sides are now reorthogonalized. Variance fractions avoid premature
   underflow when squaring very small singular values. Near-exact reconstruction
   errors are recomputed from residuals to avoid cancellation between norms.
3. **Training-constant features:** a coordinate constant in training could gain
   arbitrary held-out angles under the previous scaler. Such coordinates now
   stay at zero angle; their held-out drift is reported separately. Clipping
   fractions now count actual clipping rather than every out-of-range value.
4. **Temporal index validation:** unsigned counters could wrap under subtraction
   and bypass ordering checks. Adjacent comparisons now enforce chronology in
   splitting, classification and figure input validation.
5. **Classification status:** a catch-all ValueError handler hid invalid model
   settings. Data-limited classification now has its own exception. These runs
   preserve valid POD outputs, write `status: partial`, and exit with code 1;
   deliberate POD-only runs report classification as disabled. The Windows
   launcher preserves the command's exit status.
6. **Figure correctness and provenance:** strided ACF exports distinguish
   selected samples, original frames and seconds; Welch notes report actual
   segment counts and overlap. Unsupported state composites are NaN instead of
   false zero-intensity images. Confusion/kernel source tables identify exact
   sample rows and filenames. Reconstruction and matrix labels were adjusted
   after inspecting full-aspect exports.
7. **GUI and invalid inputs:** importing figure helpers no longer forces a
   noninteractive backend. Crop selection gives a clear error if no GUI backend
   is available. Windows UTF-8 BOM configurations are accepted. Malformed label
   rows/headers, invalid numeric settings, path patterns and unknown keys are
   rejected explicitly. JSON outputs cannot contain nonstandard NaN/Infinity.
8. **Quantum simulator guards:** finite values that overflow gate angles now
   raise errors instead of producing NaN states, and memory guards account for
   gate temporaries. Independent gate-matrix tests verify the implemented map.

Training-only POD/scaling/state fitting and validation-only hyperparameter
selection were also reviewed. Classical and quantum-kernel models continue to
use the same capped training, validation and test rows and feature vectors.
Surrogate labels remain statistical candidate states, and all quantum kernels
remain exact classical simulations. See VALIDATION.md for the checks performed.
