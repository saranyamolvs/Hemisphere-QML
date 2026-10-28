# Triple-point position workflow

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
   `OPEN_CORRECTION_WINDOW` to click uncertain positions. Then rerun tracking
   with `OVERWRITE_TRACKS=True`. Additional manual points are position anchors,
   not new appearance templates. Uncertain rows block final label export.
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
