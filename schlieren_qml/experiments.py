"""Training-size and quantum-encoding studies on saved train/validation data.

The saved POD basis, preprocessing, candidate assignments, and training-fitted
angle scaler stay fixed. Test rows are deliberately not converted to numeric
features, scored, or used to define model-selection classes.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from numbers import Real
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


_INPUT_NAMES = ("config_used.json", "classification_metrics.json",
                "modal_coefficients.csv", "state_assignments.csv")
_OUTPUT_NAMES = ("validation_learning_curve.json", "validation_learning_curve.csv",
                 "validation_search.csv", "validation_learning_curve.pdf",
                 "validation_learning_curve.svg", "validation_learning_curve.png")


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)


def _scales(values):
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
        raise ValueError("angle_scales must be a nonempty sequence of positive finite numbers.")
    scales = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError("angle_scales must contain positive finite numbers.")
        if float(value) not in scales:
            scales.append(float(value))
    return scales


def _selected_csv(path, required):
    selected = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"{path.name} has missing or duplicate required columns.")
        for row in reader:
            # No test/excluded labels or numeric coefficient values are consumed.
            if row.get("partition") not in {"train", "validation"}:
                continue
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError(f"Malformed train/validation row in {path.name}.")
            try:
                index = int(row["sequence_frame_index"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid train/validation frame index in {path.name}.") from exc
            if index < 0 or index in selected or not row["filename"].strip():
                raise ValueError(f"Duplicate/invalid train/validation identity in {path.name}.")
            selected[index] = row
    return selected


def _load_frozen_training_validation(results):
    config = json.loads((results/"config_used.json").read_text(encoding="utf-8-sig"))
    metrics = json.loads((results/"classification_metrics.json").read_text(encoding="utf-8-sig"))
    classifier = dict(config.get("classifier", {}))
    features = _integer(classifier.get("n_features", metrics.get("n_features", 4)), "saved n_features")
    if features > 12 or metrics.get("n_features", features) != features:
        raise ValueError("Saved feature count is inconsistent or exceeds 12 simulated qubits.")
    coefficients = _selected_csv(results/"modal_coefficients.csv",
        {"filename", "sequence_frame_index", "partition"} | {f"a_{j+1}" for j in range(features)})
    assignments = _selected_csv(results/"state_assignments.csv",
        {"filename", "sequence_frame_index", "partition", "state"})
    if not coefficients or coefficients.keys() != assignments.keys():
        raise ValueError("Saved train/validation coefficients and assignments refer to different frames.")
    positions, values, labels = {}, {}, {}
    for partition in ("train", "validation"):
        indices = sorted(index for index, row in coefficients.items() if row["partition"] == partition)
        if not indices:
            raise ValueError(f"No saved {partition} rows are available.")
        block, states = [], []
        for index in indices:
            row, assignment = coefficients[index], assignments[index]
            if row["filename"] != assignment["filename"] or row["partition"] != assignment["partition"]:
                raise ValueError("Saved coefficient and assignment filename/partition alignment failed.")
            state = assignment["state"].strip()
            if not state or state.lower() in {"nan", "none"}:
                raise ValueError("Saved train/validation candidate assignments contain blank labels.")
            try:
                block.append([float(row[f"a_{j+1}"]) for j in range(features)])
            except (ValueError, TypeError) as exc:
                raise ValueError("Saved train/validation POD coefficients must be numeric.") from exc
            states.append(state)
        positions[partition] = np.asarray(indices, dtype=int)
        values[partition] = np.asarray(block, dtype=float)
        labels[partition] = np.asarray(states)
        if not np.all(np.isfinite(values[partition])):
            raise ValueError("Saved train/validation POD coefficients must be finite.")
    if positions["train"][-1] >= positions["validation"][0]:
        raise ValueError("Saved training and validation frames must be disjoint and chronological.")
    scaler = metrics.get("angle_scaler", {})
    if scaler.get("fit_partition", "train") != "train":
        raise ValueError("The saved angle scaler must have been fitted on training only.")
    low = np.asarray(scaler.get("data_min", []), dtype=float)
    high = np.asarray(scaler.get("data_max", []), dtype=float)
    if low.shape != (features,) or high.shape != (features,) or not np.all(np.isfinite([low, high])) or np.any(high < low):
        raise ValueError("Saved training min/max scaler metadata are missing or invalid.")
    if not (np.allclose(low, values["train"].min(axis=0), rtol=5e-12, atol=1e-10)
            and np.allclose(high, values["train"].max(axis=0), rtol=5e-12, atol=1e-10)):
        raise ValueError("Saved angle scaler does not match these training coefficients.")
    if not np.allclose(scaler.get("range", [0., np.pi]), [0., np.pi]):
        raise ValueError("The saved angle scaler must use the shared [0, pi] base feature range.")
    from sklearn.preprocessing import MinMaxScaler

    constant = high == low
    # Reconstruct the original affine mapping solely from its saved extrema,
    # including sklearn's near-constant-range policy. This restores frozen
    # metadata; it does not fit new training/validation-dependent parameters.
    reconstructed_scaler = MinMaxScaler(feature_range=(0., np.pi), clip=False).fit(np.vstack([low, high]))
    for partition in values:
        transformed = reconstructed_scaler.transform(values[partition])
        transformed[:, constant] = 0.0
        values[partition] = np.clip(transformed, 0., np.pi)
    return config, metrics, classifier, positions, values, labels


def _preflight_destination(results, output, overwrite):
    if output == results or results in output.parents or output in results.parents:
        raise ValueError("Keep validation-study output separate from the saved source-results folder.")
    if output.exists() and not output.is_dir():
        raise ValueError("Validation-study output must be a directory.")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("Validation-study output already contains files. Choose a new folder or set overwrite=True.")
    for name in _OUTPUT_NAMES:
        destination = output/name
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError("Generated outputs must not replace symlinks or directories.")
        if destination.exists() and destination.stat().st_nlink > 1:
            raise ValueError("Generated outputs must not replace hard-linked source files.")
        for source_name in _INPUT_NAMES:
            source = results/source_name
            if destination.exists() and destination.samefile(source):
                raise ValueError("Generated outputs must not overwrite saved source inputs.")


def _counts(values, classes):
    counts = Counter(values.tolist())
    return {label: int(counts.get(label, 0)) for label in classes}


def _save_learning_plot(runs, directory, dpi):
    import matplotlib.pyplot as plt
    from .figures import STYLE

    paths = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained", sharey=True)
        for ax, name, title, color in zip(axes, ("rbf_svm", "qsvm"),
                ("RBF-SVM", "Fidelity-kernel SVM"), ("#0072B2", "#D55E00")):
            sizes = [run["n_train"] for run in runs]
            for partition, style, marker in (("training", "--", "o"), ("validation", "-", "s")):
                scores = [run[partition][name]["balanced_accuracy"] for run in runs]
                ax.plot(sizes, scores, color=color, linestyle=style, marker=marker, markersize=4,
                        label="Training" if partition == "training" else "Validation")
            ax.set(title=title, xlabel="Classifier training examples", ylim=(0., 1.03), xticks=sizes)
            ax.grid(axis="y", color="0.9", linewidth=0.5)
            ax.legend(frameon=False)
        axes[0].set_ylabel("Balanced accuracy")
        fig.suptitle("Frozen-label learning curves — training and validation only", fontsize=10)
        try:
            for extension in ("pdf", "svg", "png"):
                path = directory/f"validation_learning_curve.{extension}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=.04)
                paths.append(path.name)
        finally:
            plt.close(fig)
    return paths


def validation_learning_curve(results_directory, output_directory, *,
        train_sizes=(256, 512, 1024), max_validation=512,
        angle_scales=(0.1, 0.25, 0.5, 1.0), C_values=None, reps=None,
        dpi=300, overwrite=False) -> dict[str, Any]:
    """Compare training caps/quantum scales without evaluating held-out test rows.

    Each learning-curve point uses the same saved labels and scaler and the same
    capped validation rows. Both classifiers receive identical rows at that
    point. Larger caps are sampled evenly over the fixed chronological training
    partition; smaller subsets are not claimed to be nested. No improvement,
    temporal independence, or physical state validity is assumed.
    """
    from .classify import fit_validate_classifiers

    if not isinstance(train_sizes, (list, tuple, np.ndarray)) or len(train_sizes) == 0:
        raise ValueError("train_sizes must be a nonempty sequence.")
    sizes = [_integer(value, "train_sizes", 2) for value in train_sizes]
    validation_cap = _integer(max_validation, "max_validation")
    dpi = _integer(dpi, "dpi")
    scales = _scales(angle_scales)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be True or False.")
    results, output = (Path(path).expanduser().resolve() for path in (results_directory, output_directory))
    for name in _INPUT_NAMES:
        if not (results/name).is_file():
            raise FileNotFoundError(f"Saved-run study needs {name} in {results}.")
    _preflight_destination(results, output, overwrite)
    saved_config, source_metrics, config, positions, angles, labels = _load_frozen_training_validation(results)
    if saved_config.get("image_directory"):
        image_directory = Path(saved_config["image_directory"]).expanduser().resolve()
        if output == image_directory or image_directory in output.parents or output in image_directory.parents:
            raise ValueError("Keep validation-study output separate from the original TIFF source folder.")
    config["angle_scales"] = scales
    if C_values is not None:
        config["C_values"] = C_values
    if reps is not None:
        config["reps"] = reps
    intended = np.unique(np.concatenate([labels["train"], labels["validation"]])).tolist()
    validation_rows = np.linspace(0, len(angles["validation"])-1,
                                 min(validation_cap, len(angles["validation"])), dtype=int)
    validation_labels = labels["validation"][validation_rows]
    global_warnings = [
        "Candidate assignments, POD features, and the original training angle scaler are frozen; no clustering or POD is refitted.",
        "This study uses training and validation only. Reserve a fresh chronological block or independent run for final evaluation; a previously inspected test set is no longer a fresh final comparison.",
        "Temporal dependence remains: use justified temporal gaps and independent runs or temporal-block uncertainty before generalization claims.",
        "Quantum kernels are exact noiseless classical simulations; no hardware execution or speedup is demonstrated.",
    ]
    if source_metrics.get("label_mode") == "cluster-surrogate":
        global_warnings.append("Saved targets are K-means surrogate assignments, not independently validated physical flow regimes.")
    missing_validation = sorted(set(intended)-set(validation_labels))
    if missing_validation:
        global_warnings.append("The capped validation set omits intended classes: " + ", ".join(missing_validation) + ".")
    if len(validation_rows) < 20:
        global_warnings.append("Fewer than 20 validation examples: treat the curve as a software/analysis pilot.")
    actual_sizes = sorted({min(size, len(angles["train"])) for size in sizes})
    runs = []
    for count in actual_sizes:
        train_rows = np.linspace(0, len(angles["train"])-1, count, dtype=int)
        training_labels = labels["train"][train_rows]
        missing = sorted(set(intended)-set(training_labels))
        messages = []
        if missing:
            messages.append("Capped training rows omit intended classes: " + ", ".join(missing) + ".")
        fitted = fit_validate_classifiers(angles["train"][train_rows], training_labels,
            angles["validation"][validation_rows], validation_labels, config)
        runs.append({
            "n_train": int(count), "n_validation": int(len(validation_rows)),
            "requested_train_sizes": [size for size in sizes if min(size, len(angles["train"])) == count],
            "train_sequence_frame_indices": positions["train"][train_rows].tolist(),
            "sampled_class_counts": {"train":_counts(training_labels, intended),
                                      "validation":_counts(validation_labels, intended)},
            "training": fitted["training"], "validation": fitted["validation"],
            "selected_hyperparameters": fitted["selected_hyperparameters"],
            "search": fitted["search"], "warnings": messages,
        })
    report = {
        "status":"complete", "study":"frozen_saved_run_validation_learning_curve",
        "source_results_directory":str(results), "output_directory":str(output),
        "source_label_mode":source_metrics.get("label_mode", "unspecified"),
        "n_features":int(angles["train"].shape[1]), "intended_classes":intended,
        "retained_counts":{name:int(len(rows)) for name, rows in positions.items()},
        "validation_sequence_frame_indices":positions["validation"][validation_rows].tolist(),
        "angle_scaler":source_metrics["angle_scaler"],
        "quantum_angle_scales":scales, "runs":runs, "warnings":global_warnings,
        "selection":"At each training size, select hyperparameters by validation balanced accuracy, then macro-F1; test rows are not evaluated.",
        "figures":[str(output/f"validation_learning_curve.{extension}") for extension in ("pdf", "svg", "png")],
    }
    # Validate serialization before creating/replacing any output.
    serialized = json.dumps(report, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="validation_study_", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging/"validation_learning_curve.json").write_text(serialized, encoding="utf-8")
        with (staging/"validation_learning_curve.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["n_train", "n_validation", "model", "partition", "accuracy", "balanced_accuracy", "macro_f1"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for run in runs:
                for partition in ("training", "validation"):
                    for name, performance in run[partition].items():
                        writer.writerow({"n_train":run["n_train"], "n_validation":run["n_validation"],
                            "model":name, "partition":partition,
                            **{key:performance[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")}})
        with (staging/"validation_search.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["n_train", "n_validation", "model", "C", "gamma", "reps", "angle_scale",
                      "accuracy", "balanced_accuracy", "macro_f1"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for run in runs:
                for name, search in run["search"].items():
                    for trial in search:
                        writer.writerow({"n_train":run["n_train"], "n_validation":run["n_validation"],
                            "model":name, **{key:trial.get(key, "") for key in fields[3:]}})
        _save_learning_plot(runs, staging, dpi)
        # Recheck immediately before committing staged outputs.
        _preflight_destination(results, output, overwrite)
        output.mkdir(parents=True, exist_ok=True)
        for name in _OUTPUT_NAMES:
            (staging/name).replace(output/name)
    return report
