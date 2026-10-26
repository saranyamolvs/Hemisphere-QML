"""Windows-friendly command line; all image processing runs on the user's computer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from pathlib import Path, PureWindowsPath

import numpy as np


def _configure_plot_cache():
    # Keeps transient plotting caches in the working folder, including on
    # managed machines where the default user profile may be read-only.
    cache = Path.cwd() / "work" / ".plot_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")


def _validate_config(cfg):
    """Reject invalid settings before allocating caches or replacing outputs."""
    if not isinstance(cfg, dict):
        raise ValueError("The run configuration must be a JSON object.")
    allowed = {
        "image_directory", "output_directory", "pattern", "fps", "mach", "stagnation_pressure_psia",
        "roi", "bin_factor", "start", "limit", "stride", "rank", "batch_size", "oversampling",
        "power_iterations", "seed", "train_fraction", "validation_fraction", "gap_frames",
        "labels_csv", "classify", "classifier", "figures", "software_test_fixture",
    }
    if set(cfg)-allowed:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(set(cfg)-allowed))}")
    def integer(value, key, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    def positive(value, key):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number.")
    for key in ("image_directory", "output_directory"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty path string.")
    if not isinstance(cfg.get("pattern", "*"), str) or not cfg.get("pattern", "*").strip():
        raise ValueError("pattern must be a nonempty glob string.")
    pattern = cfg.get("pattern", "*")
    if Path(pattern).is_absolute() or PureWindowsPath(pattern).is_absolute() or ".." in PureWindowsPath(pattern).parts:
        raise ValueError("pattern must be relative to image_directory and cannot contain '..'.")
    for key, default in (("bin_factor", 1), ("stride", 1), ("rank", 12), ("batch_size", 32)):
        integer(cfg.get(key, default), key)
    for key, default in (("start", 0), ("oversampling", 8), ("power_iterations", 1), ("seed", 42), ("gap_frames", 10)):
        integer(cfg.get(key, default), key, 0)
    if cfg.get("limit", 200) is not None:
        integer(cfg.get("limit", 200), "limit")
    for key, default in (("fps", 51032), ("mach", 2), ("stagnation_pressure_psia", 20)):
        positive(cfg.get(key, default), key)
    train, validation = cfg.get("train_fraction", .6), cfg.get("validation_fraction", .2)
    positive(train, "train_fraction")
    positive(validation, "validation_fraction")
    if train+validation >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")
    for key in ("classify", "software_test_fixture"):
        if key in cfg and not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be a JSON boolean.")
    if cfg.get("labels_csv") is not None and (not isinstance(cfg["labels_csv"], str) or not cfg["labels_csv"].strip()):
        raise ValueError("labels_csv must be null or a nonempty path string.")
    roi = cfg.get("roi")
    if roi is not None:
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError("roi must be null or [x0,y0,x1,y1].")
        for value in roi:
            integer(value, "ROI coordinates", 0)
        if roi[0] >= roi[2] or roi[1] >= roi[3]:
            raise ValueError("ROI must have positive width and height.")
    figures = cfg.get("figures", {})
    if not isinstance(figures, dict):
        raise ValueError("figures must be a JSON object.")
    figure_keys = {"dpi", "figure_dpi", "n_modes_plot", "psd_nperseg", "acf_max_lag", "pixel_size_mm", "hemisphere_diameter_mm", "Uinf", "mode_color_vmax", "mode_color_percentile"}
    if set(figures)-figure_keys:
        raise ValueError(f"Unknown figures keys: {', '.join(sorted(set(figures)-figure_keys))}")
    for key in ("dpi", "figure_dpi", "n_modes_plot", "acf_max_lag"):
        if key in figures:
            integer(figures[key], f"figures.{key}")
    if "psd_nperseg" in figures:
        integer(figures["psd_nperseg"], "figures.psd_nperseg", 4)
    for key in ("pixel_size_mm", "hemisphere_diameter_mm", "Uinf"):
        if figures.get(key) is not None:
            positive(figures[key], f"figures.{key}")
    for key in ("mode_color_vmax", "mode_color_percentile"):
        if figures.get(key) is not None:
            positive(figures[key], f"figures.{key}")
    if figures.get("mode_color_percentile") is not None and figures["mode_color_percentile"] > 100:
        raise ValueError("figures.mode_color_percentile must be greater than zero and at most 100.")
    if figures.get("mode_color_vmax") is not None and figures.get("mode_color_percentile") is not None:
        raise ValueError("Set only one of figures.mode_color_vmax and figures.mode_color_percentile.")
    classifier = cfg.get("classifier", {})
    if not isinstance(classifier, dict):
        raise ValueError("classifier must be a JSON object.")
    classifier_keys = {"n_features", "n_clusters", "max_train", "max_validation", "max_test", "C_values", "gamma_values", "reps", "entanglement", "seed", "class_weight", "angle_scales"}
    if set(classifier)-classifier_keys:
        raise ValueError(f"Unknown classifier keys: {', '.join(sorted(set(classifier)-classifier_keys))}")
    for key, default, minimum in (("n_features", 4, 1), ("n_clusters", 2, 2), ("seed", 42, 0)):
        integer(classifier.get(key, default), f"classifier.{key}", minimum)
    if cfg.get("classify", True) and classifier.get("n_features", 4) > min(cfg.get("rank", 12), 12):
        raise ValueError("classifier.n_features must not exceed POD rank or 12 simulated qubits.")
    for key in ("max_train", "max_validation", "max_test"):
        if classifier.get(key) is not None:
            integer(classifier[key], f"classifier.{key}", 2 if key == "max_train" else 1)
    if "C_values" in classifier:
        if not isinstance(classifier["C_values"], list) or not classifier["C_values"]:
            raise ValueError("classifier.C_values must be a nonempty list.")
        for value in classifier["C_values"]:
            positive(value, "classifier.C_values")
    if "angle_scales" in classifier:
        if not isinstance(classifier["angle_scales"], list) or not classifier["angle_scales"]:
            raise ValueError("classifier.angle_scales must be a nonempty list.")
        for value in classifier["angle_scales"]:
            positive(value, "classifier.angle_scales")
    if "gamma_values" in classifier:
        if not isinstance(classifier["gamma_values"], list) or not classifier["gamma_values"]:
            raise ValueError("classifier.gamma_values must be a nonempty list.")
        for value in classifier["gamma_values"]:
            if isinstance(value, str) and value in {"scale", "auto"}:
                continue
            positive(value, "classifier.gamma_values")
    if "reps" in classifier:
        reps = classifier["reps"] if isinstance(classifier["reps"], list) else [classifier["reps"]]
        if not reps:
            raise ValueError("classifier.reps must contain at least one repetition count.")
        for value in reps:
            integer(value, "classifier.reps")
    entanglement = classifier.get("entanglement", "linear")
    if not isinstance(entanglement, str) or entanglement not in {"linear", "full"}:
        raise ValueError("classifier.entanglement must be 'linear' or 'full'.")
    if classifier.get("seed", 42) > 2**32-1:
        raise ValueError("classifier.seed must not exceed 2**32-1.")
    weight = classifier.get("class_weight", "balanced")
    if weight is not None and not isinstance(weight, dict) and weight != "balanced":
        raise ValueError("classifier.class_weight must be null, 'balanced', or a label-to-weight object.")
    if isinstance(weight, dict):
        for value in weight.values():
            positive(value, "classifier.class_weight")


def _load_labels(path, filenames):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"filename", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("labels_csv must contain filename,label columns.")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("labels_csv contains duplicate column names.")
        lookup = {}
        for line, row in enumerate(reader, 2):
            if None in row or not isinstance(row.get("filename"), str) or not isinstance(row.get("label"), str):
                raise ValueError(f"Malformed label CSV row {line}: filename and label are required.")
            name, label = row["filename"].strip(), row["label"].strip()
            if not name:
                raise ValueError(f"Empty filename in label CSV row {line}.")
            if name in lookup:
                raise ValueError(f"Duplicate label row: {name}")
            if not label:
                raise ValueError(f"Empty label for {name}")
            lookup[name] = label
    missing = [name for name in filenames if name not in lookup]
    if missing:
        raise ValueError(f"Labels missing for {len(missing)} selected images, e.g. {missing[:3]}")
    return np.asarray([lookup[name] for name in filenames])


def _temporal_diagnostics(scores, fps, stride, train_indices):
    """Training-only autocorrelation; a diagnostic, not an automatic embargo choice."""
    from scipy.signal import fftconvolve

    rows = []
    crossing = []
    for j in range(min(4, scores.shape[1])):
        a = np.asarray(scores[train_indices, j], dtype=float)
        a -= a.mean()
        ac = fftconvolve(a, a[::-1], mode="full")[len(a)-1:]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        hit = np.flatnonzero(np.abs(ac[1:]) <= np.exp(-1))
        lag = int(hit[0] + 1) if len(hit) else None
        crossing.append({"mode": j+1, "first_abs_acf_below_1_over_e_frames": None if lag is None else lag*stride})
        for k in range(min(len(a)//2, 1000)):
            rows.append((j+1, k*stride, k*stride/fps, ac[k]))
    return rows, crossing


def _reconstruction_metrics(X, model, scores, splits, batch_size):
    """Projection-error curves with an explicit fluctuation-norm denominator."""
    result = {}
    for name in ("train", "validation", "test"):
        indices = splits[name]
        centered_ss = 0.0
        for first in range(0, len(indices), batch_size):
            centered = np.asarray(X[indices[first:first+batch_size]], dtype=float)-model.mean
            centered_ss += float(np.einsum("ij,ij->", centered, centered))
        projected_ss = np.sum(np.asarray(scores[indices])**2, axis=0)
        raw_residual_ss = centered_ss-np.cumsum(projected_ss)
        residual_ss = np.maximum(raw_residual_ss, 0.0)
        # The difference of two large norms loses precision near an exact
        # reconstruction. Recompute only those ranks from explicit residuals.
        if centered_ss > 0:
            for rank_index in np.flatnonzero(np.abs(raw_residual_ss) <= centered_ss*1e-8):
                residual_sum = 0.0
                rank = rank_index+1
                for first in range(0, len(indices), batch_size):
                    rows = indices[first:first+batch_size]
                    residual = np.asarray(X[rows], dtype=float)-model.mean-scores[rows, :rank]@model.modes[:rank]
                    residual_sum += float(np.einsum("ij,ij->", residual, residual))
                residual_ss[rank_index] = residual_sum
        error = np.sqrt(residual_ss/centered_ss) if centered_ss > 0 else None
        result[name] = {
            "ranks": list(range(1, scores.shape[1]+1)),
            "relative_rms_fluctuation_error": error.tolist() if error is not None else [None]*scores.shape[1],
            "n_frames": int(len(indices)), "centered_sum_squares": centered_ss,
            "definition": "sqrt(sum||x-mu_train-projection_r||^2 / sum||x-mu_train||^2); all frames in this partition",
        }
    return result


def run(config_path, overwrite=False):
    from .data import build_image_matrix, list_images, read_image, preprocess_image
    from .pod import fit_pod
    from .classify import make_splits, compare_classifiers, ClassificationUnavailableError

    config_path = Path(config_path).resolve()
    config_bytes = config_path.read_bytes()
    cfg = json.loads(config_bytes.decode("utf-8-sig"))
    _validate_config(cfg)
    # Paths in the config resolve against the working directory, matching Windows examples.
    out = Path(cfg["output_directory"]).resolve()
    image_dir = Path(cfg["image_directory"]).resolve()
    if out == image_dir or out in image_dir.parents or image_dir in out.parents:
        raise ValueError("Keep output_directory separate from image_directory; neither may contain the other.")
    if (out / "run_manifest.json").exists() and not overwrite:
        raise ValueError(f"Completed output exists at {out}. Choose a new output_directory or --overwrite.")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is not empty: {out}. Choose a new folder or --overwrite.")
    paths = list_images(image_dir, cfg.get("pattern", "*"))
    selected_paths = paths[cfg.get("start", 0)::cfg.get("stride", 1)]
    if cfg.get("limit", 200) is not None:
        selected_paths = selected_paths[:cfg.get("limit", 200)]
    if not selected_paths:
        raise ValueError("No images remain after start/limit/stride selection.")
    original_indices = cfg.get("start", 0)+np.arange(len(selected_paths))*cfg.get("stride", 1)
    splits = make_splits(original_indices, cfg.get("train_fraction", .6),
                         cfg.get("validation_fraction", .2), cfg.get("gap_frames", 10))
    first_processed = preprocess_image(read_image(selected_paths[0]), cfg.get("roi"), cfg.get("bin_factor", 1))
    if cfg.get("rank", 12) > min(len(splits["train"])-1, first_processed.size):
        raise ValueError("rank exceeds available centered training frames or processed pixels.")
    if cfg.get("classify", True) and not cfg.get("labels_csv") and cfg.get("classifier", {}).get("n_clusters", 2) > len(splits["train"]):
        raise ValueError("classifier.n_clusters exceeds the number of training frames.")
    labels = None
    if cfg.get("classify", True) and cfg.get("labels_csv"):
        labels_path = Path(cfg["labels_csv"]).resolve()
        if labels_path == out or out in labels_path.parents:
            raise ValueError("Keep labels_csv outside output_directory to protect source labels.")
        names = [p.name for p in selected_paths]
        if len(set(names)) != len(names):
            raise ValueError("Selected image basenames are not unique; filename-based labels would be ambiguous.")
        labels = _load_labels(labels_path, names)
    _configure_plot_cache()
    import matplotlib
    matplotlib.use("Agg")
    from .figures import make_figures, plot_roi
    out.mkdir(parents=True, exist_ok=True)
    if overwrite:
        # Remove only this program's conditional outputs, so a rerun without
        # successful classification cannot leave old classifier figures/metrics.
        for name in ("classification_metrics.json", "classification_arrays.npz", "state_assignments.csv", "run_manifest.json"):
            (out / name).unlink(missing_ok=True)
        if (out / "figures").is_dir():
            for file in (out / "figures").iterdir():
                if file.is_file() and (file.name == "FIGURE_NOTES.md" or
                    (re.match(r"^\d{2}_[a-z_]+\.", file.name) and file.suffix in {".png", ".svg", ".pdf", ".csv", ".npz", ".json", ".md"})):
                    file.unlink()
    write_json(out / "config_used.json", cfg)
    started = time.perf_counter()
    print(f"Found {len(paths):,} image files. Reading selected consecutive sequence...", flush=True)
    X, meta = build_image_matrix(
        paths, out / "image_matrix.npy", fps=cfg.get("fps", 51032),
        roi=cfg.get("roi"), bin_factor=cfg.get("bin_factor", 1),
        start=cfg.get("start", 0), limit=cfg.get("limit", 200), stride=cfg.get("stride", 1),
        manifest_path=out / "frames.json",
    )
    n, p = X.shape
    # Sequence assumes all images are consecutive and none were dropped; audit saved filenames.
    original_indices = np.asarray([record["frame_index"] for record in meta["frame_records"]])
    fps = float(cfg.get("fps", 51032))
    times = original_indices/fps
    write_json(out / "splits.json", {k: v for k, v in splits.items()})
    print(f"Matrix {n:,} x {p:,}; train={len(splits['train'])}, validation={len(splits['validation'])}, test={len(splits['test'])}.", flush=True)
    print("Fitting training-only randomized POD...", flush=True)
    model = fit_pod(X, splits["train"], rank=cfg.get("rank", 12),
                    batch_size=cfg.get("batch_size", 32), oversampling=cfg.get("oversampling", 8),
                    power_iterations=cfg.get("power_iterations", 1), seed=cfg.get("seed", 42))
    model.save(out / "pod_model.npz")
    scores = model.transform(X, batch_size=cfg.get("batch_size", 32))
    np.save(out / "modal_coefficients.npy", scores)
    meta["reconstruction_metrics"] = _reconstruction_metrics(X, model, scores, splits, cfg.get("batch_size", 32))
    write_json(out / "reconstruction_metrics.json", meta["reconstruction_metrics"])
    with (out / "modal_coefficients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "sequence_frame_index", "time_s", "partition"] + [f"a_{j+1}" for j in range(scores.shape[1])])
        membership = np.full(n, "excluded", dtype=object)
        for key in ("train", "validation", "test"):
            membership[splits[key]] = key
        for i in range(n):
            writer.writerow([selected_paths[i].name, original_indices[i], times[i], membership[i], *scores[i]])
    acf_rows, acf_crossings = _temporal_diagnostics(scores, fps, cfg.get("stride", 1), splits["train"])
    with (out / "training_autocorrelation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "lag_original_frames", "lag_s", "normalized_acf_biased"])
        writer.writerows(acf_rows)
    result = None
    classification_status = "disabled" if not cfg.get("classify", True) else "pending"
    warnings = [
        "Camera sequence timing assumes no missing/dropped frames and uniform acquisition; verify frames.json.",
        "POD variance is schlieren image-intensity variance, not kinetic energy or calibrated density variance.",
        "Default state count, rank, crop, and temporal gap are exploratory; test robustness and independent runs.",
        "Test results from one run do not establish generalization to other tunnel runs or operating conditions.",
    ]
    if n < 1000:
        warnings.append("Short pilot record: do not claim converged POD, low-frequency statistics, or reliable classifier superiority.")
    if cfg.get("stride", 1) != 1:
        warnings.append("Temporal striding reduces Nyquist frequency and can alias dynamics; no temporal antialias filter is applied.")
    if cfg.get("limit", 200) is not None and n < cfg.get("limit", 200):
        warnings.append(f"Requested {cfg.get('limit', 200)} frames but only {n} were available after start/stride selection.")
    if cfg.get("classify", True):
        print("Tuning RBF-SVM and exact simulated quantum-kernel SVM on validation block...", flush=True)
        try:
            result = compare_classifiers(scores, labels, splits, cfg.get("classifier", {}))
        except ClassificationUnavailableError as exc:
            # Still preserve the valid POD analysis if a short block cannot support classification.
            warnings.append(f"Classification was not produced: {exc}")
            classification_status = "unavailable"
            print(warnings[-1], flush=True)
        if result is not None:
            classification_status = "complete"
            write_json(out / "classification_metrics.json", result["metrics"])
            np.savez_compressed(out / "classification_arrays.npz",
                               labels=result["labels"], test_indices=result["test_indices"],
                               train_indices=result["train_indices"], validation_indices=result["validation_indices"],
                               **{f"prediction_{k}": v for k, v in result["predictions"].items()},
                               **{f"kernel_{k}": v for k, v in result["kernels"].items()})
            prediction_by_row = {
                int(row): {name: prediction[j] for name, prediction in result["predictions"].items()}
                for j, row in enumerate(result["test_indices"])
            }
            with (out / "state_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "sequence_frame_index", "time_s", "partition", "state", "rbf_test_prediction", "qsvm_test_prediction", "majority_test_prediction"])
                for i in range(n):
                    pred = prediction_by_row.get(i, {})
                    writer.writerow([selected_paths[i].name, original_indices[i], times[i], membership[i], result["labels"][i],
                                     pred.get("rbf_svm", ""), pred.get("qsvm", ""), pred.get("majority", "")])
            if labels is None:
                warnings.append("Cluster labels are training-defined surrogate targets; classification measures assignment reproduction, not independent physical validation.")
    fig_cfg = dict(cfg.get("figures", {}))
    fig_cfg.update(fps=fps, stride=cfg.get("stride", 1), bin_factor=cfg.get("bin_factor", 1), batch_size=cfg.get("batch_size", 32))
    print(f"Exporting PDF, SVG and {fig_cfg.get('dpi', 600)}-dpi PNG figures...", flush=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    original = read_image(selected_paths[0])
    figures = plot_roi(original, cfg.get("roi"), fig_dir, config=fig_cfg)
    figures += make_figures(X, tuple(meta["image_shape"]), model, scores, times,
                           splits["train"], fig_dir, meta, classification=result, config=fig_cfg)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "scikit-learn", "matplotlib", "Pillow")}
    manifest = {
        "status": "partial" if classification_status == "unavailable" else "complete",
        "classification_status": classification_status,
        "source": "software-test fixture, not experimental data" if cfg.get("software_test_fixture", False) else "user-supplied image files",
        "synthetic": bool(cfg.get("software_test_fixture", False)),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "python": sys.version, "platform": platform.platform(), "versions": versions,
        "frames": n, "pixels_per_frame": p, "matrix_disk_GB": X.nbytes/1e9,
        "sample_interval_s": cfg.get("stride", 1)/fps,
        "first_to_last_span_s": float(times[-1]-times[0]), "nyquist_Hz": fps/(2*cfg.get("stride", 1)),
        "pod_rank": len(model.modes), "pod_retained_training_variance_fraction": float(np.sum(model.energy_fraction)),
        "training_acf_diagnostics": acf_crossings, "warnings": warnings,
        "elapsed_s": time.perf_counter()-started, "figures": figures,
        "quantum_backend": "exact NumPy statevector simulation; no quantum hardware execution",
    }
    write_json(out / "run_manifest.json", manifest)
    for warning in warnings:
        print("NOTE:", warning)
    print(f"Finished in {manifest['elapsed_s']:.1f} s. Results: {out}", flush=True)
    return manifest


def inspect_image(image_path, output):
    if Path(image_path).resolve() == Path(output).resolve():
        raise ValueError("Preview output must not overwrite the input image.")
    _configure_plot_cache()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .data import read_image
    image = read_image(image_path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image, cmap="gray", interpolation="none", extent=(0, image.shape[1], image.shape[0], 0))
    ax.set(xlabel="x (original pixel)", ylabel="y (original pixel)", title=Path(image_path).name)
    ax.grid(alpha=.25)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Image width={image.shape[1]}, height={image.shape[0]}; preview saved to {output}")


def select_roi(image_path, save_path):
    if Path(image_path).resolve() == Path(save_path).resolve():
        raise ValueError("ROI output must not overwrite the input image.")
    _configure_plot_cache()
    import matplotlib
    backend = str(matplotlib.get_backend()).lower()
    if backend in {"agg", "pdf", "svg", "ps", "pgf", "cairo", "template"} or "inline" in backend:
        raise ValueError("An interactive Matplotlib backend is required for select-roi. Use inspect and enter the ROI coordinates manually if no desktop backend is available.")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    from .data import read_image
    image = read_image(image_path)
    fig, ax = plt.subplots()
    h, w = image.shape
    ax.imshow(image, cmap="gray", extent=(0, w, h, 0))
    ax.set_title("Drag ROI; press Enter to save, close to cancel")
    selected = {}
    def on_select(click, release):
        if None in (click.xdata, click.ydata, release.xdata, release.ydata):
            return
        roi = [max(0, int(np.floor(min(click.xdata, release.xdata)))),
               max(0, int(np.floor(min(click.ydata, release.ydata)))),
               min(w, int(np.ceil(max(click.xdata, release.xdata)))),
               min(h, int(np.ceil(max(click.ydata, release.ydata))))]
        if roi[0] < roi[2] and roi[1] < roi[3]:
            selected["roi"] = roi
    selector = RectangleSelector(ax, on_select, useblit=True, button=[1], interactive=True)
    def on_key(event):
        if event.key == "enter" and "roi" in selected:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            write_json(save_path, selected)
            print(f"Saved {save_path}. Copy this roi into your run config: {selected['roi']}")
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return selector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run configured image/POD/classification/figure pipeline")
    run_parser.add_argument("config")
    run_parser.add_argument("--overwrite", action="store_true")
    inspect_parser = sub.add_parser("inspect", help="Write an image preview with original pixel coordinates")
    inspect_parser.add_argument("image")
    inspect_parser.add_argument("--output", default="results/preview.png")
    crop_parser = sub.add_parser("select-roi", help="Interactively draw an ROI on your local Windows desktop")
    crop_parser.add_argument("image")
    crop_parser.add_argument("--save", default="configs/roi.json")
    args = parser.parse_args()
    try:
        if args.command == "run":
            manifest = run(args.config, overwrite=args.overwrite)
            if manifest["status"] == "partial":
                parser.exit(1, "POD outputs were saved, but requested classification was unavailable; see run_manifest.json.\n")
        elif args.command == "inspect":
            inspect_image(args.image, args.output)
        else:
            select_roi(args.image, args.save)
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
