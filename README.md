# Schlieren POD and quantum-kernel SVM feasibility study

A runnable local Python project for a Windows folder of single-frame TIFF images. Configurations record **Mach 2.0, stagnation pressure 20 psia, acquisition 51,032 fps**, and a nominal 768-pixel width × 512-pixel height. Actual dimensions are read from the TIFF files. No experimental image or experimental result is included: the experiment has not yet been processed here.

The first run uses 200 **consecutive** images. The full configuration uses up to 20,000. Default state count, crop, feature count, rank, and temporal gaps are exploratory settings rather than established properties of this flow.

## Start with Anaconda and Jupyter Notebook

Open Anaconda Navigator, launch Jupyter Notebook, and open
[`Schlieren_POD_QSVM.ipynb`](Schlieren_POD_QSVM.ipynb) in the extracted project
folder. Run its cells in order with **Shift+Enter**. The notebook checks your
active Python kernel, lets you set the TIFF folder and fixed crop, runs the
200-frame pilot, and displays saved figures and tables inline. Keep the notebook
beside the `schlieren_qml` and `configs` folders. Your existing Anaconda
environment can be used with Python 3.10 or newer; no separate virtual
environment is required.

If packages are missing, the notebook can install them into its active kernel
with `%pip install -r requirements.txt`; restart the kernel afterward. For a
dedicated Conda environment, use the supplied `environment.yml`. See
[`docs/ANACONDA_JUPYTER.md`](docs/ANACONDA_JUPYTER.md) for step-by-step setup,
Windows path examples, kernel troubleshooting and the optional full-record run.

## Alternative: run from PowerShell on Windows

Open PowerShell in this project folder. Use Python 3.10 or newer (3.11/3.12 recommended). These commands do not require PowerShell activation scripts:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Edit `configs/pilot.json`: replace `image_directory` with your TIFF folder, using forward slashes, for example `D:/Schlieren/Mach2_run01`. Paths without a drive letter resolve from the folder where you launch the command. The `pattern` value `*` accepts `.tif` and `.tiff` with either case. Keep the input folder limited to one camera sequence, without reference or calibration frames mixed into the time series.

```powershell
.\.venv\Scripts\python.exe -m schlieren_qml inspect "D:/Schlieren/Mach2_run01/frame00001.tif"
.\.venv\Scripts\python.exe -m schlieren_qml select-roi "D:/Schlieren/Mach2_run01/frame00001.tif" --save configs/roi.json
```

The second command opens a desktop window. Drag a rectangle and press Enter. Copy the printed `roi` into `configs/pilot.json`. For example, `"roi": [120, 80, 700, 430]` selects columns 120–699 and rows 80–429. Coordinates refer to the original image, x increases rightward, y downward, and upper bounds are excluded. Leave `roi: null` to use the whole image. If the GUI is unavailable, inspect the saved preview and type the coordinates. Include the bow shock, separation/shock interaction and downstream region relevant to your question; remove irrelevant borders, text and reflections. A rectangular crop cannot remove the hemisphere/wall silhouette separately; complex body masks and registration require an explicit follow-up extension after image inspection.

```powershell
.\.venv\Scripts\python.exe -m schlieren_qml run configs/pilot.json
```

Alternatively double-click `run_pilot.bat` after setup. Nonempty output folders are protected from accidental overwrite. Use a new `output_directory` or add `--overwrite` to deliberately rerun it. Keep outputs and source images in separate folders, with neither folder inside the other. Configuration, crop, split and label checks run before previous outputs are replaced. The TIFF source files are read without modification.

## What the run does

1. Natural-sort filenames, preserve the grayscale TIFF numeric scale, apply the same crop and optional square area binning to every frame, and stream a float32 matrix to disk. No per-image contrast normalization is used. Single-frame TIFF is supported; multipage TIFF is rejected with an explicit message.
2. Create ordered train/validation/test partitions with gaps measured in original camera-frame indices. Defaults are nominal 60/20/20% boundaries before exclusions. The pilot's 10-frame gap is a starting value, not a decorrelation guarantee.
3. Fit the image mean and randomized spatial POD **only on training frames**. Project the complete record onto this frozen basis. Retained variance uses the exact total centered training-intensity variance as denominator, including the unresolved tail.
4. Fit a training-only [0, π] angle mapping on the first four modal coefficients. Features constant during training remain at zero angle, and held-out changes in these features are reported separately. With no labels, fit three K-means candidate clusters in that training feature space and assign held-out frames using frozen centers. Both SVMs receive identical features, sampled frames and class weighting.
5. Tune RBF-SVM C/γ and fidelity-kernel SVM C/circuit repetitions on validation, then evaluate once on test. A majority-class baseline is included. Both final models remain fitted on training only. The quantum kernel is an exact NumPy statevector calculation; no quantum account, hardware or Qiskit installation is needed.
6. Save numeric outputs, full settings, software versions, warnings, training autocorrelation, classifier sample indices and figure source tables. Export PDF, SVG and 600-dpi PNG figures. SVG text is editable; PDFs embed TrueType fonts.

**No independent labels are available.** Classifier accuracy currently measures reproduction of a train-defined K-means partition in the same features. Nearest-center assignment defines these labels exactly. High accuracy is not independent evidence of physical flow states, better flow understanding, or quantum advantage. Inspect state-conditional images and state duration, then establish a physical label definition or use independent measurements for stronger classification claims.

## Figure outputs

Generated under `results/pilot/figures/` after you run your images:

| Figure | Content |
|---|---|
| 01 | Original schlieren frame with selected ROI |
| 02 | Selected-record mean and RMS intensity fluctuation maps |
| 03 | Training POD modal and cumulative image-intensity variance |
| 04 | Leading spatial modes with one symmetric color scale |
| 05 | Modal coefficient time histories and classification block ranges |
| 06 | Welch modal spectra with true sample rate and stated frequency-bin spacing |
| 07 | Training-standardized POD portrait and candidate assignments |
| 08 | Held-out image, POD reconstruction and residual |
| 09 | Count-annotated, row-normalized held-out confusion matrices |
| 10 | Training and test-versus-training fidelity kernel matrices |
| 11–12 | Candidate-state composites and temporal occupancy, when classification succeeds |
| 13 | Training modal autocorrelation for assessing temporal separation |
| 14 | Relative RMS reconstruction error versus retained rank in each time partition |

`FIGURE_NOTES.md` documents scales, population and interpretation. These are journal-oriented exports; final panel dimensions, lettering, captions and journal requirements should be reviewed after the actual images are available. No physical length scale is invented. Set `figures.pixel_size_mm` to **mm per original sensor pixel** when calibrated. Image axes are ROI-relative rather than wall/body coordinates. `hemisphere_diameter_mm` and `Uinf` are reserved metadata for a future explicit Strouhal calculation and are not currently used to relabel spectra.

## Key numeric outputs

- `frames.json`: selected filenames, natural-order indices, assumed times and intensity diagnostics.
- `image_matrix.npy`: disk-backed preprocessed image matrix. This is the main disk cost.
- `pod_model.npz`: training mean, unit-normalized modes, singular values and variance fractions.
- `reconstruction_metrics.json`: errors by rank and partition, normalized by fluctuations about the training mean.
- `modal_coefficients.npy/.csv`: coefficients for all selected frames and partition membership.
- `training_autocorrelation.csv`: training-only modal lag correlations; use these to investigate the gap, with sensitivity checks.
- `figures/13_modal_autocorrelation.csv`: distinguishes selected-sample lags, original-camera-frame lags and seconds, including when stride exceeds one.
- `splits.json`, `classification_arrays.npz`: complete partition and capped classifier row indices, labels, predictions and kernels.
- `classification_metrics.json`: balanced accuracy, macro-F1, support, validation search, clipping, timing and kernel diagnostics.
- `state_assignments.csv`: inspectable frame-by-frame states and predictions on the actual evaluated test rows.
- `config_used.json`, `run_manifest.json`: reproducibility settings, versions and important limitations.

Load saved arrays without pickle: `np.load(path, allow_pickle=False)`. Reconstruct with `schlieren_qml.pod.load_pod(path)` and `model.reconstruct(scores)`.

If the requested classification cannot be supported by the data, POD outputs are still saved, but `run_manifest.json` reports `status: partial` and `classification_status: unavailable`; the command exits with code 1. Invalid configuration raises an error rather than silently skipping classification. A deliberately POD-only run (`classify: false`) can finish with `status: complete` and `classification_status: disabled`. Undefined numeric quantities use JSON null; unsupported state-mean images use NaN in NPZ rather than fabricated zero-valued images.

## Scale to 20,000 images

First inspect the pilot outputs and filename ordering. A folder with missing frame numbers is not automatically repaired: `time_s = sorted_sequence_index / fps` assumes no dropped images. If the camera dropped frames or filename counters encode gaps, obtain acquisition timestamps and extend the timing input before interpreting spectra. Keep `stride: 1` for temporal analyses. Temporal striding can alias high-frequency dynamics; this program does not apply an antialias filter.

Edit `configs/full.json` with the same image folder and selected ROI, then run:

```powershell
.\.venv\Scripts\python.exe -m schlieren_qml run configs/full.json
```

The full default uses `bin_factor: 2` to reduce disk cost. This spatial averaging can suppress fine features, so compare it against `bin_factor: 1` on the pilot. Crop width and height must each divide exactly by the bin factor. Full 768×512 images ×20,000 require **31.46 GB** for float32 cache alone; binning by two reduces it to **7.86 GB**. Allow extra space for modes, outputs and filesystem overhead. POD batches and a spatial sketch keep RAM bounded; the record is not loaded as a dense full matrix in RAM, and no 20,000² snapshot matrix is formed. Hardware and BLAS settings affect run time.

Both SVMs default to the same 256 training /128 validation /256 test examples on the full configuration. The complete image record is still used for POD, spectra and diagnostics. Capping the kernel examples controls its quadratic cost; it does not make 20,000-image SVM training free. Four POD features encode four qubits (16 statevector amplitudes). With 256 training samples, the training kernel contains 65,536 entries. The two circuit repetition settings have distinct validation searches. Timings describe this classical simulation, not hardware speed or asymptotic advantage.

## Timing and publication limits

At 51,032 fps, Δt=19.5955 µs and the camera Nyquist frequency is 25,516 Hz. First-to-last span is 3.8995 ms for 200 frames and 0.391891 s for 20,000. A 200-point DFT has 255.16 Hz bin spacing. The default pilot Welch segment of 128 frames has 398.6875 Hz spacing and very few averages. Exposure time, optical response, jitter and any aliasing also limit usable bandwidth; Nyquist alone does not establish measurement fidelity.

The short pilot checks code and figures. It cannot establish converged modes or reliable model superiority. Before paper results, check POD subspace/variance convergence versus record length and spatial resolution; assess ROI sensitivity, state count/stability, illumination and camera-motion artifacts; choose temporal gaps using training evidence and sensitivity analysis; and use repeat runs if possible. Compute comparison uncertainty with justified temporal-block or run-level resampling, not independent-frame bootstrap. No confidence interval or significance claim is fabricated by this pilot.

The pressure is **stagnation absolute pressure**, not static pressure or psig. Schlieren measures an optical intensity response linked to line-of-sight refractive-index gradients. These POD modes are not velocity modes and their variance is not kinetic energy. See `docs/PAPER_GUIDE.md` for equations, writing guidance and the evidence needed for the proposed paper.

## Optional external physical labels and quantum circuits

After defining labels independently, use a UTF-8 CSV with `filename,label` columns, exactly matching every selected frame basename, and set `labels_csv` to its path. Empty/duplicate/missing selected labels are rejected. Keep label definitions independent of test data. To perform POD/figures without classification set `"classify": false`.

Optional Qiskit export: install the project extra (`python -m pip install ".[qiskit]"`) and call `schlieren_qml.quantum.make_qiskit_circuit(angles, reps=2)`. The exported circuit implements the documented custom H/RZ/RZZ map. Hardware fidelity would estimate the all-zero outcome probability of `U(z)†U(x)|0…0⟩`; shots, backend, transpilation, noise and PSD treatment need separate experimental design. This project does not submit hardware jobs.

## Check the implementation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use small mathematical/image fixtures rather than experimental observations. They check train-only centering/scaling, POD reconstruction, total-variance accounting, original-index gaps, image bit depth/crops, fair classifier samples and fidelity-kernel correctness against independent explicit gates. Optional Qiskit equivalence is skipped when Qiskit is absent.
