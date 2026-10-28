# Triple-point tracking, review, and classification

Use `Triple_Point_Classification.ipynb` for the new TP-based experiment. The
older notebook remains the K-means exploration workflow.

1. Set the TIFF folder, selected sequence, and a separate TP output folder.
2. Enable `OPEN_ANNOTATION_WINDOW`. Click the TP, then press Enter on each of
   the first 20 original images. Backspace revisits the previous frame. Closing
   preserves confirmed positions; reopening resumes missing annotations.
3. Enable `RUN_TRACKING`. The tracker uses fixed normalized-correlation patches
   from the initial manual training frames. It saves positions and quality
   flags for every selected image without modifying TIFFs.
4. Inspect overlays, including accepted matches. Enable
   `OPEN_CORRECTION_WINDOW` to review uncertain positions. **A / Keep TP**
   confirms that the displayed position is already correct, resolving a false
   alarm without changing its coordinates. **Click + Enter / Save correction**
   saves a corrected coordinate. **S / Defer** moves on without approval; a
   flagged point stays unresolved. Backspace revisits the previous image.
   Closing preserves saved decisions; reopening resumes unresolved points.
   `REVIEW_ALL_TRACKS=True` also permits correcting accepted/manual positions.
   The notebook applies saved decisions immediately after closing the window,
   without a full tracking rerun. The next cell can also apply them separately.
   An optional tracking rerun propagates corrected position anchors; previous
   confirmations apply only when the new coordinate matches the reviewed one.
   Corrections are position anchors, not new appearance templates.
   Unresolved rows block final label export.
5. Enable `EXPORT_TP_CATEGORIES`. Boundaries use only the chronological training
   block's mean y and population SD (`ddof=0`), then stay fixed for validation
   and test. Original image y increases downward:

   | Label | Criterion |
   | --- | --- |
   | candidate_1 | Physically lower: y > training mean + SD |
   | candidate_2 | Average band: mean − SD ≤ y ≤ mean + SD |
   | candidate_3 | Physically higher: y < training mean − SD |

6. Inspect original frames in each TP category. Enable
   `RUN_POD_AND_CLASSIFIERS` to run a new experiment with the TP label CSV.
   K-means is bypassed; the category definition is copied into the new results.

The notebook includes your requested settings: TIFFs under
`H:\FSU\Hemisphere\POD_QuantCom\20000Images`, 20,000 frames at 51,032 fps,
POD crop `[400,160,750,510]`, binning by two, and fixed **±0.009** limits in
the original mode figure. Outputs go under `PROJECT_DIR/results-TP/tp_measurement`.
The 350×350 crop becomes 175×175 processed pixels. Restart the kernel after
updating the whole project so the new functions load. The old K-means notebook
remains separate.

## Save marked reference sheets

Enable `EXPORT_REFERENCE_SHEETS` to save every frame, up to **16 per 4-by-4
PNG**. At 20,000 frames this produces 1,250 sheets. Export before review to
inspect flags, then rerun after review to keep resolved positions. Rerun after
category export to include the category labels. `REFERENCE_TILE_WIDTH` controls
thumbnail detail and disk use. Category sheets also show two horizontal lines:
**cyan at training mean − SD**, separating Candidate 3 above from Candidate 2,
and **yellow at training mean + SD**, separating Candidate 2 from Candidate 1
below. The same frozen sensor-coordinate thresholds appear on every frame.
The TP marker is drawn on top of the lines. One TIFF and one page are held at a time.
Reference grayscale conversion does not affect TIFFs or POD intensities.

`reference_sheets/tp_reference_index.csv` saves every original filename,
coordinate, quality/review field, category when available, page, tile, and
thumbnail coordinates. The exact marked points remain in original sensor
pixels. These sheets are reference visualizations, not quantitative replacement
images.

## Check the classifier predictions

The classifiers use POD modal coefficients as inputs and TP categories as
targets. **TP y is not supplied as a classifier input.** Training-only POD and
feature scaling prevent test information from fitting the feature basis. Both
models share samples; validation selects model settings, then test is evaluated.
The quantum kernel is a noiseless NumPy statevector simulation.

Default caps are 512 training, 512 validation, and 256 test images from
chronological blocks with gaps. Tracking and POD projection cover all selected
frames. These caps and the POD rank/feature count are visible parameters. A
larger quantum kernel costs more time and memory.

The final notebook cell exports `tp_test_predictions.csv` into
`categories_pod_qsvm`, with only actually evaluated held-out test images:
filename, TP coordinates, true category, each model's prediction, and each
model's correctness. It also displays accuracy, balanced accuracy, macro-F1,
and confusion matrices. Blank predictions elsewhere in `state_assignments.csv`
mean unsampled images, not failed classifications. Do not tune on final test
outcomes.

Changing `MODE_COLOR_VMAX` refreshes the same original spatial-mode figure
through the final results cell; there is no separate recoloring cell or second
adjusted mode-image file.

## Retained measurements and review history

- `tp_manual.csv`: initial seed clicks and corrected coordinates.
- `tp_review.csv`: explicit confirmation/correction decisions and the proposal
  being reviewed. Keeping a correct proposal does not create a manual anchor.
- `tracking/tp_tracks.csv`: all positions, `automatic_accepted`, final
  `accepted`, and `review_status`.
- `tracking/tp_tracking_manifest.json`: settings, hashes, and unresolved counts.
- `categories/`: labels, complete TP states, frozen mean/SD definition and plot.
- `categories_pod_qsvm/`: POD, classifier results, per-test-image predictions,
  and copied measurement/review provenance.

If initial seeds are missing, use initial annotation mode (without the review
filter). Keep the same `ANNOTATION_CSV` for seeds and corrections. Existing
CSVs are retained; the code does not silently merge similarly named results
folders. The notebook checks saved settings before using old tracking, and
checks category/track hashes before reporting model results. Restore settings
or rerun tracking after parameter changes; re-export categories after changing
tracks. Existing model results require explicit `OVERWRITE_POD_RESULTS=True`
before replacement. `RUN_POD_AND_CLASSIFIERS=True` starts only after its
prerequisites exist and are consistent.

All coordinates are original sensor pixels, independent of POD cropping or
binning. The export retains positions, states, mean/SD metadata, a position-time
figure, and a configuration preserving the measurement sequence and split.

## Measurement validation

Template matching is an initial measurement aid, not a validated TP detector.
Similarity and competing-peak scores are not error probabilities. High
correlation can select another shock feature. The first 20 frames span about
0.372 ms at 51,032 fps; check manual landmarks throughout the record, especially
extremes and low contrast. Adjust patch size, search radius, or a fixed TP ROI
only with measurement validation; do not simply reduce thresholds to remove
review flags. Assess labeling error near the category boundaries.

The desktop uses TkAgg (or QtAgg if installed) in a child process so Jupyter
keeps its inline backend. Its click/save callbacks are tested headlessly; the
actual Anaconda/Windows window and experimental tracking accuracy require
local verification. Original TIFFs and manually marked copies must not be
mixed in the source sequence.

The position bins define an observable-based task, not proof of three discrete
flow regimes. Accuracy is not directly comparable with earlier K-means target
reproduction. Select models using validation and reserve a fresh final holdout.
