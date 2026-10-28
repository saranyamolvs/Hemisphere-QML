# Run the rebuilt TP notebook in Anaconda

1. Close the running notebook kernel. Extract the updated project ZIP. To keep
   existing measurements, copy its `schlieren_qml` folder and
   `Triple_Point_Classification.ipynb` into your existing project, replacing the
   code. Keep a backup of `results-TP`; do not delete it or move it into the
   TIFF folder. Keep the existing `configs` folder beside the notebook.
2. Open `Triple_Point_Classification.ipynb` in Jupyter and run sections 1 and 2.
   The TIFF path, crop, frame count, mode limits, and results paths are already
   set to the supplied values. Check the displayed crop and missing seed count.
3. If initial seeds are missing, enable `OPEN_ANNOTATION_WINDOW=True`, rerun
   parameters, and run section 3. Click TP and press Enter for each missing
   initial image. Then set that switch back to False.
4. Enable `RUN_TRACKING=True`, rerun parameters, and run section 4. Wait for
   all selected images to finish. Then set that switch back to False.
5. Inspect section 5. Enable `OPEN_CORRECTION_WINDOW=True`, rerun parameters,
   and run its review cell. Use **A** if the marked point is already correct,
   **click + Enter** to correct it, and **S** to leave it for later. Closing
   preserves decisions. The notebook applies them without retracking. Repeat
   until unresolved flags are zero. Use `REVIEW_ALL_TRACKS=True` if you want
   to inspect/correct accepted points too. Turn off the review switch afterward.
6. Enable `EXPORT_REFERENCE_SHEETS=True`, rerun parameters, and run section 6
   to save all points as 16-frame PNG sheets. This can be done before review
   too. An index CSV retains the exact coordinates and page/tile location.
7. Enable `EXPORT_TP_CATEGORIES=True`, rerun parameters, and run section 7.
   Inspect the three groups. The average band is training mean ± one SD;
   Candidate 1 is physically lower, Candidate 3 physically higher. Rerun
   section 6 afterward if you want the categories printed on all sheets.
8. `RUN_POD_AND_CLASSIFIERS=True` is already supplied. Run section 8 after
   categories are ready. Existing results are retained unless you enable
   `OVERWRITE_POD_RESULTS=True`. Run section 9 for metrics, confusion matrices,
   the fixed-limit original POD modes, and `tp_test_predictions.csv`.

The default models use 512 training, 512 validation, and 256 held-out test
samples, with POD projections and TP positions for all 20,000 frames. These
caps are separate parameters. The last CSV identifies which test images were
actually evaluated and whether each model predicted their category correctly.

Full instructions and limitations: `docs/TRIPLE_POINT_WORKFLOW.md`.
