# Conference-paper study workflow

Open **Conference_Paper_Study.ipynb** in the updated project using your Anaconda
kernel. Restart the kernel after replacing code. The notebook uses completed,
reviewed TP/POD outputs from `Triple_Point_Classification.ipynb`; it does not
require marking the initial 20 frames again.

Default sources are `results-TP/tp_measurement/categories_pod_qsvm` and
`results-TP/tp_measurement/categories/tp_states.csv`. The saved configuration
supplies your image directory, acquisition timing, original crop, binning and
chronological split. Conference outputs default to `results-TP/conference_v1`.
Use a new `STUDY_TAG` for a changed completed study. Nonempty stage outputs are
protected from replacement; original measurements/results remain unchanged.

The source categories remain fixed: Candidate 1 is physically lower
(`y > training mean + SD`), Candidate 2 is the inclusive mean ± SD band, and
Candidate 3 is physically higher (`y < training mean - SD`). Sensor y increases
downward. Crop/feature comparisons never redefine these targets.

## 1. TP measurements and their uncertainty

Enable `PREPARE_REPEAT_SAMPLE`, rerun parameters, and run section 2. The default
sample is 60 time-distributed training frames and 30 validation frames. It
excludes test frames and saves identities without TP proposals.

Enable `OPEN_BLIND_ROUND_1` and run section 3. Click the TP and press Enter. S
defers a frame, Backspace revisits the previous one, and closing preserves
confirmed positions. Complete `OPEN_BLIND_ROUND_2` in a separate session without
consulting the first round; the shuffled order changes. Windows opens a separate
TkAgg window. Original tracks/annotations are not displayed or overwritten.

Enable `RUN_MEASUREMENT_REPORT` and run section 4. Incomplete annotation produces
an explicit pending report, not invented uncertainty. A new timestamped report
is written each time. For each partition the complete report provides:

- Horizontal and vertical annotation repeatability and inter-round bias.
- Tracking bias, MAE and RMSE relative to the mean of two manual clicks.
- Category disagreement between the repeated vertical measurements.
- A diagnostic figure and the exact sampled measurement CSV.

The random single-annotation SD estimate is `SD(round2 - round1)/sqrt(2)`, under
independent equal-variance annotation errors. Inter-round bias is separate.
The RMS difference divided by sqrt(2) also includes that inter-round bias.
Neither establishes absolute truth, common systematic bias, calibration error,
or total measurement uncertainty. Repeat observations and sampled frames may
remain correlated. Consider a second annotator and independent landmarks.
Zero observed repeat variability does not establish zero absolute measurement error.

Physical TP oscillation SD is a different quantity. It defines the position
categories and must not be relabeled as measurement error. All measurements and
thresholds remain in original sensor pixels.

The secondary boundary sensitivity width is set using **training repeats only**:
`2 * max(repeatability_SD, RMS_repeat_difference/sqrt(2))`. This is a descriptive
width, not a calibrated 95% confidence interval. Validation repeats audit
tracking agreement. If the repeats expose incorrect tracks, correct/re-export
the original experiment before continuing; source hashes will then require a
new conference study.

`REQUIRE_MEASURED_UNCERTAINTY=True` blocks the notebook's validation stage while
the repeat report is pending. Setting it False permits an explicitly exploratory
assumed pixel width, labeled as assumed throughout. It does not create a
measurement uncertainty result.

## 2. POD representation and fixed-dimension comparisons

Enable `RUN_VALIDATION_STUDY` after measurement checks. It compares:

- Leading coefficients from the current POD.
- Coefficients ranked by absolute TP-y Pearson correlation using only that
  trial's sampled training rows.
- The same two representations from each requested comparison crop.

Every classifier representation has **four features/qubits by default**.
Change `FEATURE_DIMENSION` consistently; the study limits simulation to eight
qubits. New crop PODs fit all original training images only, using a disk-backed
training/validation image matrix; no test TIFF is read during this stage. The
cache is closed and removed, retaining the fitted POD and crop metadata. TP
coverage inside each crop is reported and must be inspected.

The default wider crop `[320,140,750,510]` extends left of your current crop.
This is a candidate, not a validated optimal region. Check that it retains the
TP excursion and relevant interacting shocks. Set `CROP_STUDIES={}` to compare
features without TIFF access.

Training Pearson/Spearman correlations are exported for every available mode.
A fixed-alpha Ridge regression uses the same sampled training rows and selected
columns to predict continuous TP y on validation. Its RMSE/MAE/R² are diagnostics,
not SVM/QSVM accuracy. A training-mean RMSE baseline is included. Nonlinear
representation diagnostics are also available in `TP_POD_Diagnostics.ipynb`.

## 3. Matched RBF-SVM/QSVM learning curves

Defaults are training sizes 64/128/256/512, three seeded nested training
subsamples, and a common capped 512-frame validation block. Both models receive
identical rows, columns, labels, training-fitted angle scaling and constant-feature
handling within each trial. Min/max scaling and supervised mode ranking fit only
that trial's training rows. Held-out clipping is reported. Validation selects
each model's C/kernel/circuit settings.

The plotted bars are SD across training subsets, **not confidence intervals or
independent wind-tunnel experiments**. POD and TP category boundaries still use
all training images/measurements; these are classifier sample-size curves, not
a demonstration of total annotation efficiency.

A common representation is selected by the average of both models' mean
validation balanced accuracies at the prespecified `FINAL_TRAIN_SIZE`. The
first seed is frozen rather than choosing the best seed. Ties use the first
representation. `selection.json`, its SHA256 and a numeric NPZ recipe store
exact training angles/labels/indices, scaling, columns and selected settings.
All training sample identities and validation predictions are exported.

## 4. Boundary-sensitive errors and computational cost

Primary metrics use every evaluated image. Error-versus-distance bins include
frame counts. Distance is the minimum absolute y displacement to either frozen
training mean ± SD boundary. Secondary metrics outside the sensitivity width
always report retained N and coverage; they must not replace primary metrics.

Timing separates validation hyperparameter search from selected-model prediction,
and new-crop preprocessing/POD fitting/projection from classification. Historical
POD processing costs for the reused source are explicitly unmeasured. Final
reports separately time selected-model fitting and prediction.

The QSVM kernel is an exact noiseless NumPy statevector simulation. Timings
include classical state generation and dense kernel construction. Reported dense
array bytes are not peak process memory. This is no measurement of quantum
hardware speed, shots, noise, compilation, queue time or quantum advantage.

## 5. Frozen evaluation and the paper's evidence

Review sections 5–6, then enable `RUN_FROZEN_TEST` and run section 7. It checks
the saved selection/source/model hashes, reconstructs the selected models from
training data without any new search, and evaluates the same prespecified test
sample for both models. A new crop projects only those sampled test TIFFs. No
refit includes validation. Existing test outputs cannot be overwritten.

The current test record has already been inspected, so the default usage label
is **previously_inspected**. Treat it as retrospective/descriptive. Use the
unseen label only when the underlying holdout really was uninspected and its
protocol was frozen beforehand. An independent record is preferable for the
strongest generalization claim.

The final report includes all three class supports, accuracy/balanced accuracy/
macro-F1, a matched majority baseline, exact image-level predictions, boundary
errors/coverage, confusion counts, wall times, clipping and dense-array sizes.
A paired temporal-block bootstrap compares the two models' balanced accuracies;
choose block length using training correlation and assess sensitivity. Too few
blocks or class-complete draws produce an insufficient-data status. The interval
does not establish temporal independence, stationarity or quantum advantage.

Section 8 records an analysis-status checklist. Code execution does not by itself
complete a conference paper: authors must assess tracking accuracy, calibration,
cropping, state support, temporal dependence and independent validation, and
report the outcomes without promising an improved classifier score.

## Figures and inspectable outputs

The study exports PNG/PDF/SVG figures and CSV/JSON numeric sources for:

1. Blind-repeat localization and tracking agreement.
2. Matched classifier learning curves.
3. Fixed-dimension representation comparisons.
4. POD–TP relationships.
5. Continuous-position validation prediction error.
6. Classical/simulator computational cost.
7. Frozen test confusion matrices.
8. Validation/test error versus boundary distance, with support counts.

The source notebook retains no experimental or synthetic outputs. A complete
software-fixture execution and test-numeric poisoning checks are described in
`docs/VALIDATION.md`. Actual Anaconda/Windows desktop operation and the user's
experimental measurements still require local verification.
