# Run the schlieren workflow in Anaconda and Jupyter Notebook

You can use your existing Anaconda installation. Open the supplied
`Schlieren_POD_QSVM.ipynb` notebook to enter your TIFF folder, select a fixed crop,
run the 200-image pilot, and view figures and tables in notebook cells. A separate
Python virtual environment is not required.

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

## Adjust an existing mode-shape colorbar

Use the notebook's **Adjust the colorbar of already saved mode shapes** section.
Run the setup and imports cells, then set `RESULTS_DIRECTORY` in the recoloring
cell to your saved run folder. You do not need to run the image-processing cells
again. The helper needs only `pod_model.npz`, `frames.json`, and
`config_used.json`; it does not open the image matrix or original TIFFs.

```python
from schlieren_qml.notebook import replot_spatial_modes, display_saved_figures

adjusted = replot_spatial_modes(
    RESULTS_DIRECTORY, vmax=0.01, n_modes=4, dpi=600
)
display_saved_figures(adjusted)
```

This uses one shared linear range from -0.01 to +0.01, with zero at the neutral
color. Treat 0.01 as a display trial rather than a universal setting. To use a
pooled absolute-amplitude percentile, use `vmax=None, percentile=99.5` instead.
Set at most one of the two limit options. Leaving both as None uses the full
data range. Never infer additional resolved variance from stronger colors.

Color clipping is recorded per mode in
`04_spatial_modes_color_scale.json` and indicated by the colorbar's end arrows.
The data, modal coefficients and variance fractions are unchanged. Adjusted
PDF/SVG/PNG files go into `figures/mode_color_adjusted`, preserving the original
mode figure. For future complete runs, the parameters cell also exposes
`MODE_COLOR_VMAX` and `MODE_COLOR_PERCENTILE`.

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
