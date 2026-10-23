# Publication guide: schlieren POD and quantum-kernel classification

This document describes a defensible study design. It contains no experimental
results, performance values, or evidence of quantum advantage. Figures produced
by the software require inspection and scientific interpretation before use in a
paper.

## What the present experiment supports

The supplied acquisition conditions are Mach 2.0, a wall-mounted hemisphere,
stagnation pressure 20 psia, nominal camera rate 51,032 frames/s, and image size
768 by 512 pixels. The pressure is absolute stagnation pressure; do not label it
as static pressure or gauge pressure. Verify the pressure measurement location,
uncertainty, and temporal stability from the tunnel records.

No independent state labels have been supplied. Clusters in POD space are
therefore **candidate statistical states**. Training a classifier to reproduce
those cluster assignments is a surrogate-label experiment. High classification
accuracy does not independently establish the existence, physical meaning, or
predictability of flow regimes. K-means can partition a continuous trajectory
even when the underlying process has no discrete states.

The current quantum kernel is evaluated with exact classical statevector
simulation. This establishes an executable hybrid formulation and a controlled
comparison of feature maps. It does not establish a computational speedup,
scalability on hardware, or quantum advantage. A simulator is a classical
computation of quantum-circuit quantities.

## Acquisition duration and spectral limitations

For consecutive frames with no losses, the sample spacing is 19.596 microseconds
and the Nyquist frequency is 25.516 kHz. Exposure duration and the optical/camera
response may impose a lower useful bandwidth.

| Frames | Nominal record duration N/fs | First-to-last timestamp span | Full-record DFT bin spacing fs/N |
| --- | --- | --- | --- |
| 200 | 3.919 ms | 3.900 ms | 255.16 Hz |
| 20,000 | 391.911 ms | 391.891 ms | 2.5516 Hz |

These full-record bin spacings are not confidence limits or guaranteed ability
to resolve nearby physical frequencies. Welch spectra use shorter segments. At
128 samples per segment the bin spacing is 398.688 Hz; at 2,048 samples it is
24.918 Hz. Report the actual window, segment length, overlap, detrending, segment
count, effective sampling frequency, and PSD scaling. Zero padding adds display
samples rather than new frequency information. The 200-frame sequence is a
software/ROI pilot and may contain too few flow cycles for reliable dynamics.

Confirm file numbering, dropped frames, and time stamps against the acquisition
record. The default natural sorting and index/fs timestamps assume a complete
constant-rate sequence. A temporal stride changes the sample rate and can cause
aliasing without a suitable low-pass filter. Default to stride one for spectra.

## Experimental information to recover before writing results

- Hemisphere diameter, mounting details, wall position, coordinate origin, field
  of view, optical magnification, and spatial calibration in mm/pixel.
- Tunnel type, test-section dimensions, free-stream Mach calibration, reservoir
  temperature and pressure histories, run duration, startup exclusion, and
  repeat-run availability. A Mach number and stagnation pressure alone do not
  determine the free-stream velocity or Reynolds number.
- Relevant Reynolds number and its reference length; fluid properties and the
  assumptions used to obtain them.
- Schlieren layout, knife-edge orientation and cutoff, illumination, exposure,
  camera model, bit depth and actual numeric range, lens, and acquisition timing.
- Boundary-layer thickness and state upstream of the body, if measured; blockage
  and wall/interference considerations appropriate to the installation.
- Instrument uncertainties and calibration procedures. Distinguish measured
  quantities from values derived under ideal-gas or isentropic assumptions.

Do not infer these values from an image or fill them with typical numbers.

## Preprocessing and POD definition

Retain the raw TIFF files. Apply one fixed ROI to every selected frame and record
its original-pixel bounds as [x0,y0,x1,y1), with the upper bounds excluded. Use
the same field of view in every comparison. Record any area binning and use the
corresponding processed-pixel calibration. Binning changes spatial resolution;
check that shocks and separation features of interest remain resolved.

The current pipeline retains the camera numeric intensity scale and performs no
implicit contrast equalization or per-frame normalization. Such operations can
change modal amplitudes and apparent dynamics. If illumination correction,
registration, masking, or background subtraction becomes necessary, justify it,
record the exact operation, and show sensitivity to reasonable choices. Learn
any data-dependent preprocessing using training frames only.

For equal-area pixels, let x_i be the flattened processed intensity image. For
the training set T, define the mean mu = (1/|T|) sum_(i in T) x_i and the centered
matrix A whose rows are (x_i-mu)^T. The spatial POD follows

    A = U Sigma V^T,
    phi_j = column j of V,
    a_j(t_i) = (x_i-mu)^T phi_j,
    x_i^(r) = mu + sum_(j=1)^r a_j(t_i) phi_j.

The implementation uses a randomized spatial-range SVD to control memory use.
Report retained rank, oversampling, power iterations, seed, batch size, and the
software version. Verify the approximation on a manageable subset against a
direct SVD, and assess rank/seed sensitivity where modes are nearly degenerate.
Individual mode signs are arbitrary. Nearly equal eigenvalues can also permit
rotations of a modal subspace, so compare such subspaces rather than relying on
one mode's sign or orientation.

The retained variance fraction is sigma_j^2 / ||A||_F^2. The denominator is the
total centered **training image-intensity variance**, including the unresolved
tail. A truncated cumulative spectrum need not reach 100%. Do not normalize only
by retained singular values and then claim that they explain all variation.

Schlieren intensity is an optical measurement related to line-of-sight refractive
index gradients and the optical transfer setup. Its POD eigenvalues are not
kinetic energy. Use “image-intensity variance fraction” in the paper. Without
additional calibration and inversion, the pipeline does not recover velocity,
pressure, or a quantitative density-gradient field.

Report normalized reconstruction errors with an explicit denominator. For
example, the ratio of summed squared residuals to summed squared fluctuations
about the training mean is suitable; label its square root as a relative RMS
error if taking a square root. Evaluate training and held-out errors separately.

## Splits, leakage, and uncertainty

Create training, validation, and final test partitions in acquisition time before
fitting the mean, POD basis, scalers, or candidate-state definitions. Use
validation data for model settings and keep the final test data out of selection.
Fit any revised preprocessing or POD dimension selection inside this separation.
Fitting POD to the full sequence before splitting is transductive and changes the
scientific question, even when the later SVM fitting uses training frames only.

The example configuration uses chronological blocks with omitted frames around
boundaries. Its gap is a starting setting, not evidence that samples are
independent. Estimate temporal correlations using training data; test several
larger gaps and block lengths. A short-gap, single-run split can still be highly
correlated. Optical drift or a tunnel transient can also confound a chronological
comparison. Plot the split locations alongside modal coefficients and intensity
quality histories.

For the final study, prefer multiple repeat tunnel runs and hold out complete
runs. If only one run exists, describe that limitation and use a predeclared
blocked evaluation design. Do not apply random frame-level train/test splits to
claim generalization to independent flow realizations.

An SVM score from one test block is a point estimate. Nearby image frames are not
independent Bernoulli trials. Ordinary framewise bootstrap intervals or binomial
accuracy intervals can be misleading. With enough data, use paired block
resampling with a justified block length or, preferably, repeat-run uncertainty.
Report class counts, variation across splits/runs, and paired uncertainty for the
model difference. Do not infer model superiority merely from different point
estimates or from overlapping/nonoverlapping marginal intervals.

## Candidate states and fair model comparison

Fit candidate-state centers using training POD features only. Treat state count
as a predeclared exploratory choice or select it using training-only evidence;
show sensitivity to the state count and seed. Silhouette or cluster-separation
scores describe geometry and do not prove physical regimes. Inspect conditional
mean images, representative raw frames, state occupancy, dwell times, and
transitions. Name physical states only when observable flow structures support
those names. A state unseen in a test block must be reported, not hidden by
dropping its row from a confusion matrix.

Both models must receive the same feature dimension, same labels, same selected
training/validation/test frames, and train-fitted scaling. Keep an unmodified
majority-class baseline and report balanced accuracy, macro F1, accuracy, class
counts, and confusion counts. RBF-SVM is already a strong nonlinear baseline;
adding a linear SVM can help interpret whether the quantum map provides anything
beyond simple geometry. Compare across feature dimensions rather than choosing
only the dimension that favors one model.

If large records require capped kernel subsets, report exact selected indices
and sample sizes. The image/POD record length and the number of SVM training
samples are different quantities. Increasing the number of processed images
does not automatically increase the quantum training set. Record validation
search sizes and computational budgets for both methods.

The fitted angle mapping can place validation/test features outside the
training range. Clipping to [0,pi] is an explicit modeling choice, not a neutral
operation. Report the out-of-range fraction and test its impact. Never refit the
angle mapping on held-out data to make clipping disappear.

The first pilot is a cluster-surrogate task. Before claiming physically useful
classification, obtain independent labels, a synchronized physical observable,
or a clearly defined downstream task. Defining and predicting labels from the
same instantaneous POD coefficients can produce a mathematically easy task even
with perfect temporal separation.

## Quantum feature map and kernel

For d retained coefficients, the implemented example uses d qubits initialized
in |0>^d. Each repetition applies H to each qubit, then RZ(2*x_j) to qubit j, and
then RZZ(2*(pi-x_j)*(pi-x_k)) to each declared entangled pair. The default pairs
are nearest neighbors in a linear chain. Features are train-scaled angles.
Report this explicit gate convention, order, repetition count, and connectivity;
the phrase “ZZ feature map” alone does not specify a reproducible circuit.

    |psi(x)> = U_phi(x)|0>^d,
    K_Q(x,z) = |<psi(x)|psi(z)>|^2,
    K_RBF(x,z) = exp(-gamma ||x-z||^2).

The SVM uses the quantum fidelity matrix as a precomputed kernel. This is a
quantum-kernel SVM, not a trained variational quantum neural network. Check kernel
symmetry, unit diagonal, numerical positive semidefiniteness, and nontrivial
off-diagonal structure. Kernel concentration, rank, and scaling sensitivity are
relevant diagnostics; a matrix close to identity can fit training data without
useful generalization.

Exact statevector work grows exponentially with qubit count. Kernel-pair counts
grow quadratically with training sample count; a 20,000-by-20,000 kernel has 400
million entries. For actual hardware, one possible fidelity experiment measures
the all-zero probability after U_phi(z)^dagger U_phi(x). Report backend, circuit
transpilation, depth, shots, noise/error mitigation, execution dates, measurement
uncertainty, and total resource use. Finite-shot/noisy kernels can lose positive
semidefiniteness; any repair must be specified and evaluated without test-driven
tuning. Hardware experiments are a separate extension to the present exact
simulation.

## Figure plan and caption requirements

Use vector PDF/SVG for line plots and sufficient-resolution raster exports for
images. Inspect the figures at the journal's final column width. Keep fonts,
coordinates, units, color scales, and state colors consistent. Include scale bars
only after spatial calibration. Do not use a generated flow image as evidence.

| Figure | Purpose and required caption information |
| --- | --- |
| Experimental setup | Author-supplied measured geometry and optical layout; body diameter, flow direction, coordinate system, and acquisition conditions. A schematic must be labeled schematic. |
| ROI and quality | Actual full frame, fixed ROI, pixel or calibrated coordinates; selected record and any binning. Include mean-intensity drift if it affects interpretation. |
| Mean and RMS images | Record used, native intensity units, RMS about the stated mean, and separate truthful color scales. |
| POD spectrum | Training-frame count, exact total-intensity-variance denominator, approximate SVD settings, retained rank, and unresolved tail. |
| Spatial POD modes | ROI and calibration; arbitrary mode sign; color-bar meaning and whether each panel has an independent scale. |
| Modal time histories | Timestamp origin, modal units, split locations, omitted gaps, and sample stride. |
| Modal PSD | Actual sample rate, window, segment length, overlap, detrending, PSD units, and limited pilot duration. State any statistically justified uncertainty separately. |
| Reduced-space states | Which coefficients, scaling, label origin, train-only state fitting, and which displayed points are held out. |
| Reconstruction | Actual frame identity, rank, shared intensity scale, residual scale, and error definition. |
| Confusion matrices | Exact held-out sample count, class counts, normalized denominator, raw counts, and surrogate-label caveat. |
| Kernel diagnostic | Common selected frame order, kernel definition, sample count, and shared [0,1] color scale for fidelities. |
| Convergence/robustness | Separate study of record length, ROI/binning/rank/state count/gaps and repeat runs. Do not imply the initial pilot alone includes these tests. |

Not every generated figure belongs in the main paper. Use the figures that
support the scientific question and place processing diagnostics in supplementary
material. The software cannot determine the physical flow structures from a
caption alone; inspect the actual images and compare them with established
supersonic hemisphere-flow physics.

## Suggested paper structure and cautious abstract

Possible title: **A POD-based hybrid classical–quantum kernel framework for
experimental supersonic schlieren data: a feasibility study**.

Structure: experimental setup and uncertainty; image preprocessing; train-only
POD and temporal sampling; candidate-state definition; classical and quantum
kernel methods; leakage-controlled evaluation; physical interpretation and
convergence; limitations and hardware outlook.

Use the following only after the described processing has actually been run:

> A hybrid classical–quantum kernel framework is investigated for experimentally
> acquired supersonic schlieren data. Images of flow over a wall-mounted
> hemisphere at Mach 2.0 and a stagnation pressure of 20 psia are recorded at
> 51,032 frames per second. A training-set proper orthogonal decomposition
> provides a reduced representation of image-intensity fluctuations. Candidate
> statistical states are defined within this representation and a radial-basis
> support vector machine is compared with a support vector machine using an
> exact-simulation fidelity kernel. Evaluation uses separated temporal blocks
> and matched feature and sample sets. [Insert measured results with uncertainty
> and describe the validation performed.] The study assesses the feasibility and
> limitations of quantum-kernel integration into an experimental fluid-dynamics
> workflow; it does not claim quantum computational advantage.

Replace the candidate-state formulation if independent physical labels become
available. Do not claim “dominant flow dynamics,” “accurate flow-state
identification,” or “quantum enhancement” solely from a short-record POD spectrum
and surrogate classification accuracy.

## Reproducibility checklist

- Preserve raw TIFFs and acquisition metadata; archive a file manifest and, for
  the final dataset, content hashes or an immutable source-data identifier.
- Record exact ROI, binning, frame ordering, missing-frame handling, chosen
  record, seed, POD settings, split indices and gaps, feature scaling, state
  settings, model grids, and selected settings.
- Save numerical figure sources, trained POD basis, modal coefficients, labels
  with origin, predictions, confusion counts, and kernel diagnostics.
- Archive exact installed package versions, code revision, OS/hardware details,
  wall times and memory measures with the timing boundary clearly defined.
- Distinguish exploratory choices from the predeclared final evaluation. Report
  all final test runs used to support the conclusion, including unfavorable
  model comparisons or failed class coverage.
- Verify bibliographic details from the original papers or publisher records.
  Cite original POD, SVM, quantum-kernel, randomized-SVD, and relevant hemisphere
  flow/experimental schlieren sources actually consulted; no references are
  fabricated in this guide.
