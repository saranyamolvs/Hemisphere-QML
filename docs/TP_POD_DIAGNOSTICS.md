# Does POD capture vertical TP motion?

Open `TP_POD_Diagnostics.ipynb` in your existing Anaconda project after the
TP-category and POD experiments have completed. The default inputs are:

- `results-TP/tp_measurement/categories_pod_qsvm/modal_coefficients.csv`
- `results-TP/tp_measurement/categories/tp_states.csv`
- The saved configuration and category-definition JSON files.

The basic study needs no TIFF reads, TP retracking, POD rerun, quantum package,
or new dependencies beyond the project requirements. Each run saves a new
timestamped folder under `results-TP/pod_tp_diagnostics` without changing the
original results.

## Run in order

1. Set the parameter paths, feature counts and optional nonlinear switch.
2. Run alignment checks; confirm the training/validation counts and saved
   mean/SD agree. Final test numeric values are excluded.
3. Inspect training Pearson/Spearman correlations, time histories, and scatter
   plots. The plots select modes using training correlation only. Signs of POD
   modes are arbitrary; magnitude and relationship shape are more meaningful.
4. Run the feature-count study. Leading modes are compared with modes ranked
   by absolute training Pearson correlation. Ridge and optional RBF-SVR predict
   continuous TP y from modal coefficients. Both use the same deterministic
   capped training rows and all validation rows; standardization fits training
   rows only. No TP coordinate or time is supplied as an input feature.
5. Optionally set `RUN_CROP_STUDY=True` and enter `CROP_STUDIES`, for example
   `{"interaction": [x0,y0,x1,y1]}` with actual original-pixel bounds. Include
   the complete TP excursion and intersecting shocks. This refits training-only
   POD from TIFFs and compares validation performance using the same splits.
   A disk-backed training/validation image matrix bounds memory and is removed
   unless `KEEP_CROP_CACHE=True`. No test image is read by the crop loop.
6. Inspect validation RMSE/MAE/R², predicted-vs-measured scatter and time
   histories, and exported per-image predictions. Hyperparameters and feature
   counts are selected on validation RMSE. Repeated validation selection makes
   the best score optimistic relative to a fresh final test.

## Interpretation

RMSE and MAE use original sensor pixels. Compare them with the training-mean
baseline and the category band width. Later useful modes can indicate that
the first four modes omit TP-relevant information. Better nonlinear regression
can suggest that a linear association misses useful structure. A better tighter
crop can support focusing POD on the interaction. None of these outcomes is
guaranteed; tracking noise, drift, illumination, and training coverage also
affect performance.

The predicted category is obtained by thresholding a regression prediction of
y against the original training boundaries. It is a diagnostic metric, **not
the SVM/QSVM result**. No assumption of independent high-speed frames or naive
correlation p-values is used. These analyses cannot establish physical causality
or prove that information is absent when one model performs poorly.

## Outputs and software verification

Figures include mode correlations, standardized training histories, mode/TP
scatter plots, validation error versus feature count, and measured/predicted
validation positions. Each is saved as PNG, PDF and SVG. CSVs preserve the full
search, validation winners and image-level predictions; a JSON manifest hashes
inputs and records settings.

The notebook was executed in a real Jupyter kernel on 150 synthetic TIFFs,
including both regressors, immediate TP-derived labels and an optional tighter
crop. A second execution with invalid numeric strings in all test coefficient
and TP-y fields produced identical validation search tables. This establishes
software behavior, not accuracy on the experimental sequence. Inspect your
actual measurement accuracy and use a fresh final record after feature selection.
