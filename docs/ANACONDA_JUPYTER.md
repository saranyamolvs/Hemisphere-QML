# Run the schlieren workflow in Anaconda and Jupyter Notebook

You can use your existing Anaconda installation. Open the supplied
`Schlieren_POD_QSVM.ipynb` notebook to enter your TIFF folder, select a fixed crop,
run the 200-image pilot, and view figures and tables in notebook cells. A separate
Python virtual environment is not required. The same notebook can show original
images grouped by candidate and compare classifier settings from saved results.

## Open the project

1. Download and extract the complete project folder. Keep the notebook together
   with `schlieren_qml`, `configs`, and `requirements.txt`; the notebook uses the
   code in these folders.
2. Open **Anaconda Navigator** and launch **Jupyter Notebook** in the environment
   you normally use.
3. In Jupyter's file browser, go to the extracted project folder and open
   `Schlieren_POD_QSVM.ipynb`. If the folder is outside the locations shown by
   Jupyter, move the extracted project to an accessible folder, or launch
   `jupyter notebook` from an Anaconda Prompt opened in the project folder.
4. Select your Python kernel. Execute cells in order with **Shift+Enter**.

The setup cells display the Python executable and check its version.
The project requires Python 3.10 or newer; Python 3.12 is a suitable choice.
Anaconda Navigator and Jupyter can be running while the notebook kernel uses a
different environment, so the executable printed inside the notebook is the
one that matters.

## Install missing packages in the active kernel

Many Anaconda environments already contain NumPy, SciPy, scikit-learn,
Matplotlib, and Pillow. Run the notebook's dependency check first. If it reports
missing or incompatible packages, run its installation cell, or use this in a
new notebook cell while the working directory is the project folder:

```python
%pip install -r requirements.txt
```

If `PROJECT_DIR` differs from the current working directory, use the absolute
path already defined by the setup cell:

```python
%pip install -r "{requirements_path}"
```

Use `%pip` inside Jupyter so the installation targets the active kernel. After
installation, restart the kernel from Jupyter's **Kernel** menu and run the
notebook from the beginning. Installing packages in a separate terminal can
target another Python environment and leave `ModuleNotFoundError` unresolved.

## Set your data and crop

Set `IMAGE_DIRECTORY` to the folder containing one TIFF time sequence, for
example:

```python
IMAGE_DIRECTORY = r"D:\Schlieren\Mach2_run01"
ROI = None
```

The raw string prefix `r` prevents Windows backslashes from becoming escape
characters. Forward slashes also work: `"D:/Schlieren/Mach2_run01"`.

Keep `ROI = None` initially to preview the complete image. Then enter the same
fixed crop for every frame:

```python
ROI = [120, 80, 700, 430]
```

This means columns 120 through 699 and rows 80 through 429, in original camera
pixels. The coordinates are `[x0, y0, x1, y1]`; upper bounds are excluded. The
notebook preview shows the proposed region before the run. After changing the
parameters, rerun the parameter and preview cells before running the analysis.
This notebook approach does not require a separate desktop crop window.

Keep the output folder separate from the image folder, with neither folder
inside the other. Use an output name for each crop or experiment so previous
results are retained. The raw TIFFs are read without modification.

## Run and inspect the 200-frame pilot

Keep the initial frame limit at 200 and the temporal stride at 1. Run the pilot
cell, then the inspection cells. Figures are displayed inline, and publication
exports and numeric files are saved to the output directory printed by the run.
Inspect the ROI, mean/RMS, POD variance spectrum, spatial modes and
state-conditional images before interpreting the classifier scores.

Changing and rerunning the parameters cell clears the active pilot report to
prevent mixing old figures with new paths or timing settings. A completed run
keeps a frozen copy of its settings for the result cells. Rerun the analysis in
a new output folder when the crop or processing parameters change. If you interrupt a
run, wait for the kernel to become idle before retrying. Rerunning a completed
folder requires the notebook's explicit overwrite setting or a new folder.

At 51,032 fps, 200 consecutive frames cover approximately 3.9 ms. This is a
software and crop pilot, not a converged measurement of low-frequency dynamics
or a reliable performance comparison. The state labels are candidate clusters,
not independently validated physical flow regimes. The quantum kernel uses an
exact classical simulator.

## Set classifier sample counts and quantum encoding scale

The parameters cell exposes the settings used by both pilot and full runs:

```python
CLASSIFIER_MAX_TRAIN = 512
CLASSIFIER_MAX_VALIDATION = 512
CLASSIFIER_MAX_TEST = 256
CLASSIFIER_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
QUANTUM_ANGLE_SCALES = [0.1, 0.25, 0.5, 1.0]
QUANTUM_REPETITIONS = [1, 2]
```

These are caps on classifier examples, not the number of TIFFs processed.
`PILOT_FRAMES` controls the pilot image count; the full configuration uses up to
20,000. Both classifiers use the same capped examples. A 200-frame pilot can
provide far fewer than 512 training or validation examples, and actual counts
are reported in its metrics.

`QUANTUM_ANGLE_SCALES` searches factors applied only when constructing quantum
states from the shared angle features. It does not rescale the RBF features or
change the K-means target assignments. Scale, C, and repetitions are chosen using
validation scores. Larger C, deeper circuits, and more training examples do not
guarantee a better held-out score. Inspect training and validation results before
drawing a conclusion about overfitting or sample-size limits.

Rerun the parameters cell after editing these controls. A new image-processing
run needs a fresh output directory or the explicit overwrite setting. To study
an existing run without repeating image processing, use the validation-only
section below.

## Fixed color limits for the original mode figure

Set `MODE_COLOR_VMAX = 0.008` and `MODE_COLOR_PERCENTILE = None` in the
parameters cell. Pilot and full processing save the original
`figures/04_spatial_modes.png` with the shared range -0.008 to +0.008.
There is no separate recoloring cell or adjusted-image folder in this workflow.

To update an existing run, set `REVIEW_RESULTS_DIRECTORY` to its saved folder,
then run setup/imports, parameters, and the normal figure-display cell. That
cell regenerates the original mode figure and its PDF/SVG versions in place
from the saved POD model. It does not read TIFFs or refit POD. Mode values,
coefficients, and variance fractions are unchanged. Color saturation is recorded
in `figures/04_spatial_modes_color_scale.json`.

## Move to the full record

The full-record section is disabled by default through `RUN_FULL = False`.
After reviewing the pilot, check the full configuration, available disk space,
and spatial binning, then set `RUN_FULL = True`, rerun the parameters and preview
cells, and run the full-record cell deliberately. Keep
temporal stride at 1 for spectra.

A full 20,000-image float32 cache at 768×512 requires approximately 31.46 GB.
Spatial binning by two reduces that to about 7.86 GB, before any crop. Binning
can remove fine spatial features, so compare it with the native-resolution
pilot. Cropped width and height must divide exactly by the bin factor.

For a large local run, keep the computer awake and allow the kernel to finish.
Closing a browser tab does not necessarily stop the Jupyter kernel; check the
running sessions before starting a second copy of the same analysis.

## View images belonging to each candidate

Set `REVIEW_RESULTS_DIRECTORY` in the parameters cell to the completed run, for
example:

```python
REVIEW_RESULTS_DIRECTORY = PROJECT_DIR / "results" / "notebook_full"
GALLERY_EXAMPLES_PER_CANDIDATE = 4
GALLERY_SEED = 42
GALLERY_LABEL_COLUMN = "state"
GALLERY_PARTITION = "train"
GALLERY_PROCESSED = True
GALLERY_ONLY_CANDIDATE = None
```

Run setup/imports, parameters, and **Inspect images in each candidate group**.
This does not run POD or classification again. Each image shows its original
filename, time, partition, and candidate assignment. Change the seed to see
other examples, or set `GALLERY_ONLY_CANDIDATE = "candidate_2"` to inspect only
that group. All panels share grayscale limits. The default uses training frames
for developing the physical interpretation before examining a fresh holdout.

The viewer applies the crop and binning recorded with that saved run. Set
`GALLERY_PROCESSED = False` to view the original TIFF pixels. Recorded source
paths identify the images; if TIFFs moved, set `IMAGE_DIRECTORY_OVERRIDE` to
their new folder. The TIFFs are never modified.

`state_assignments.csv` is the complete image-to-candidate lookup. Open it in
Excel and filter its `state` column to list every image in a group. The
`rbf_test_prediction` and `qsvm_test_prediction` columns are populated only for
the evaluated test examples; blank values elsewhere are expected. To inspect a
prediction gallery, choose one of those columns and set
`GALLERY_PARTITION = "test"` (or `None`). Titles retain both assignment and
prediction so the two cannot be confused.

Set `FILENAME_TO_VIEW` to an exact filename and run the single-image cell to
inspect that particular TIFF. Set `EXPORT_CANDIDATE_LISTS = True` to write a CSV
for each candidate to `CANDIDATE_LIST_OUTPUT_DIRECTORY`. Use a fresh export
folder. This creates filename lists without copying thousands of TIFF files.

Compare several frames per candidate with
`figures/11_state_conditional_means.png` and the spatial POD modes. The requested
three clusters are groups of similar snapshots; they are not automatically
three independently established physical flow regimes.

## Compare validation-only learning curves from saved results

The **Validation-only classifier learning curves** section uses the saved
`modal_coefficients.csv`, `state_assignments.csv`, `config_used.json`, and
`classification_metrics.json`. It preserves the existing POD feature definition,
training-fitted scaling, candidate assignments, and chronological partitions.
It does not open TIFFs, refit POD, rediscover clusters, or evaluate the test
block. Both models use identical training examples at each curve point, with a
fixed validation sample across all training sizes.

```python
RUN_VALIDATION_STUDY = True
LEARNING_CURVE_TRAIN_SIZES = [256, 512, 1024]
VALIDATION_STUDY_OUTPUT_DIRECTORY = PROJECT_DIR / "results" / "validation_study"
OVERWRITE_VALIDATION_STUDY = False
```

Run setup/imports, parameters, and the study cell. Choose a fresh study-output
directory, separate from the saved run, to preserve previous studies. Replacing
its generated outputs requires `OVERWRITE_VALIDATION_STUDY = True`. The study
uses the shared validation cap, C grid, scale grid, and repetition settings from
the parameters cell; actual available counts and class coverage are recorded.
It saves editable PDF/SVG figures, PNG previews, and inspectable JSON/CSV
results. Kernel storage and work grow approximately with the square of the
classifier training count, so start with the provided sizes before enlarging
them.

Compare validation balanced accuracy and macro-F1 and inspect training-to-
validation gaps. A learning curve may show that sample count is limiting, or
that the current encoding remains less suitable for these targets. There is no
guaranteed accuracy increase. The calculation remains an exact classical
statevector simulation, with no hardware or speedup demonstrated.

Do not choose settings from the previously viewed test matrix. Once choices are
settled, reserve a fresh chronological block or independent tunnel run for final
reporting, with uncertainty assessed by justified temporal blocks/runs. Changing
the crop, number of clustering features, or cluster count can redefine the
target task; keep existing labels fixed for controlled classifier comparisons.

## Optional dedicated Conda environment

If your current environment is old or has dependency conflicts, open **Anaconda
Prompt** in the project folder and run:

```text
conda env create -f environment.yml
conda activate hemisphere-qml
python -m ipykernel install --user --name hemisphere-qml --display-name "Python (hemisphere-qml)"
jupyter notebook
```

Choose **Python (hemisphere-qml)** in the notebook's kernel menu. Alternatively,
select the `hemisphere-qml` environment in Navigator and launch Jupyter from
there. The environment file supplies Python 3.12, the analysis packages and
Jupyter tools. Optional Qiskit circuit export is not needed for the default
quantum-kernel calculation.

If your environment already exists, do not rerun the creation command with the
same name. Activate it and check its packages, or choose another environment
name with `conda env create -n another-name -f environment.yml`.

Actual TIFF processing, camera ordering/timing, physical state interpretation,
and your Windows notebook interface still need validation with your experiment.
The bundled notebook contains no experimental results.
