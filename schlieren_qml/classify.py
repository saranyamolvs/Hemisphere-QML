"""Chronologically held-out comparison of classical and fidelity-kernel SVMs.

The classifier is a classical SVM in both cases. ``qsvm`` uses a kernel from an
exact, classically simulated quantum feature map; it does not demonstrate a
hardware execution or a computational quantum advantage. All learned transforms
and candidate-state centers use the training partition only.
"""

from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Any
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class ClassificationUnavailableError(ValueError):
    """Valid settings cannot be evaluated because the selected data lack states."""


def make_splits(
    frame_indices: np.ndarray,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    gap_frames: int = 10,
) -> dict[str, np.ndarray]:
    """Split in time, removing an embargo after train and validation.

    ``frame_indices`` contains strictly increasing original sequence indices,
    before any stride. Returned indices are row positions in the selected data.
    A gap of g means g original frame positions are withheld between partitions:
    ``first_next - last_previous > g``. Choose g from training-only temporal
    correlation and check its sensitivity; the default is only a pilot setting.
    Fractions define boundaries before the embargo, so retained fractions differ.
    """
    raw = np.asarray(frame_indices)
    if raw.ndim != 1 or len(raw) < 4:
        raise ValueError("At least four ordered original frame indices are required.")
    if raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw)):
        raise ValueError("Frame indices must be finite numbers.")
    if np.any(raw != np.floor(raw)) or np.any(raw < 0) or np.any(raw[1:] <= raw[:-1]):
        raise ValueError("Frame indices must be nonnegative, strictly increasing integers.")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Train and validation fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fractions must leave a test partition.")
    if not isinstance(gap_frames, (int, np.integer)) or gap_frames < 0:
        raise ValueError("gap_frames must be a nonnegative integer.")
    n = len(raw)
    n_train = int(np.floor(n * train_fraction))
    validation_end = int(np.floor(n * (train_fraction + validation_fraction)))
    train = np.arange(n_train)
    validation = np.arange(n_train, validation_end)
    test = np.arange(validation_end, n)
    if len(train) < 2 or not len(validation) or not len(test):
        raise ValueError("Split fractions leave insufficient training, validation, or test frames.")
    validation = validation[raw[validation] - raw[train[-1]] > gap_frames]
    if not len(validation):
        raise ValueError("The embargo removes all validation frames; use more frames or a smaller justified gap.")
    test = test[raw[test] - raw[validation[-1]] > gap_frames]
    if not len(test):
        raise ValueError("The embargo removes all test frames; use more frames or a smaller justified gap.")
    included = np.concatenate([train, validation, test])
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "excluded": np.setdiff1d(np.arange(n), included),
    }


def _subsample(indices: np.ndarray, maximum: int | None, name: str) -> np.ndarray:
    """Cap only kernel classification rows; preserve ordered endpoints."""
    indices = np.asarray(indices, dtype=int)
    if maximum is None:
        return indices.copy()
    if not isinstance(maximum, (int, np.integer)) or maximum < (2 if name == "train" else 1):
        raise ValueError(f"max_{name} is too small or is not an integer.")
    if len(indices) <= maximum:
        return indices.copy()
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]


def _validate_splits(splits: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    result = {}
    for name in ("train", "validation", "test"):
        raw = np.asarray(splits.get(name, []))
        if raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"{name} indices must be a nonempty one-dimensional integer array.")
        if np.any(raw < 0) or np.any(raw >= n) or np.any(raw[1:] <= raw[:-1]):
            raise ValueError(f"{name} indices must be sorted, unique, and in range.")
        result[name] = raw.astype(int)
    if result["train"][-1] >= result["validation"][0] or result["validation"][-1] >= result["test"][0]:
        raise ValueError("Training, validation, and test partitions must be disjoint and chronologically ordered.")
    return result


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    support = cm.sum(axis=1)
    observed = support > 0
    recall = np.divide(np.diag(cm), support, out=np.zeros(len(classes), dtype=float), where=observed)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recall[observed])),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "balanced_accuracy_definition": "Mean recall over classes present in this evaluated partition.",
        "macro_f1_definition": "Unweighted F1 over the declared union of classes; absent classes contribute zero.",
        "n": int(len(y_true)),
        "class_support": {str(label): int(count) for label, count in zip(classes, support)},
        "class_recall": {str(label): (float(value) if count else None) for label, value, count in zip(classes, recall, support)},
    }


def _label_counts(values: np.ndarray) -> dict[str, int]:
    return dict(sorted((str(k), int(v)) for k, v in Counter(values.tolist()).items()))


def _as_labels(labels: Any, n: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError("External labels must contain exactly one value for each selected frame.")
    if any(value is None or str(value).strip().lower() in {"", "nan", "none"} for value in labels):
        raise ValueError("External labels contain missing or blank values.")
    return labels.astype(str)


def compare_classifiers(
    scores: np.ndarray,
    labels: np.ndarray | None,
    splits: dict[str, np.ndarray],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune on validation, evaluate once on test, and return inspectable arrays.

    ``scores`` must come from a POD model fitted on ``splits['train']`` only;
    this function cannot verify how upstream scores were obtained. The scaler
    and optional K-means state discovery use the entire training partition.
    Both SVMs use identical capped training/validation/test rows and identical
    angle features. They remain fitted on training only after tuning; validation
    is deliberately not merged back before the final held-out comparison.

    Configuration: n_features=4, n_clusters=2, max_train=256,
    max_validation=128, max_test=256, C_values=[.1,1,10],
    gamma_values=[.1,1,'scale'], reps=2 (or a validation-selected list),
    entanglement='linear', seed=42, class_weight='balanced'.
    """
    from .quantum import feature_states, fidelity_kernel, kernel_diagnostics, feature_map_metadata

    config = dict(config or {})
    X = np.asarray(scores, dtype=float)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("POD scores must be a finite two-dimensional array.")
    partitions = _validate_splits(splits, len(X))
    d = config.get("n_features", 4)
    if not isinstance(d, (int, np.integer)) or not 1 <= d <= X.shape[1]:
        raise ValueError(f"n_features must be between one and the {X.shape[1]} available POD modes.")
    if d > 12:
        raise ValueError("This exact statevector pilot supports at most 12 feature qubits; choose fewer POD features.")
    X = X[:, :d]
    scaler = MinMaxScaler(feature_range=(0, np.pi), clip=False).fit(X[partitions["train"]])
    # A training-constant coordinate has no learned scale. sklearn otherwise
    # assigns an arbitrary unit range, allowing held-out drift to create new
    # angles despite the absence of any training variation in that coordinate.
    constant_features = scaler.data_range_ == 0
    raw_angles = scaler.transform(X)
    raw_angles[:, constant_features] = 0.0
    angles = np.clip(raw_angles, 0.0, np.pi)
    clip_tolerance = 8 * np.finfo(float).eps
    clipped = (raw_angles < -clip_tolerance) | (raw_angles > np.pi + clip_tolerance)
    constant_drift = (X != scaler.data_min_) & constant_features
    sampled = {
        name: _subsample(indices, config.get(f"max_{name}", default), name)
        for (name, indices), default in zip(partitions.items(), (256, 128, 256))
    }
    train, validation, test = (sampled[name] for name in ("train", "validation", "test"))
    messages: list[str] = []
    if np.any(constant_features):
        messages.append("Features constant during training are frozen at angle zero for every partition; held-out drift in those coordinates is reported separately and is not treated as learned variation.")
    state_centers = None
    state_discovery = None
    if labels is None:
        k = config.get("n_clusters", 2)
        if not isinstance(k, (int, np.integer)) or not 2 <= k <= len(partitions["train"]):
            raise ValueError("n_clusters must be at least two and no greater than the training count.")
        if len(np.unique(angles[partitions["train"]], axis=0)) < k:
            raise ClassificationUnavailableError("Training data have fewer distinct feature vectors than n_clusters.")
        state_model = KMeans(n_clusters=k, n_init=20, random_state=config.get("seed", 42))
        state_model.fit(angles[partitions["train"]])
        # Sort centers by feature 1 to make labels reproducible across label permutations.
        order = np.lexsort(tuple(state_model.cluster_centers_[:, j] for j in range(d - 1, -1, -1)))
        remap = np.empty(k, dtype=int)
        remap[order] = np.arange(k)
        assigned = remap[state_model.predict(angles)]
        y = np.asarray([f"candidate_{value + 1}" for value in assigned])
        state_centers = state_model.cluster_centers_[order]
        label_mode = "cluster-surrogate"
        messages.append("Labels are training-defined K-means surrogates in the same POD feature space, not independent physical flow-state ground truth.")
        messages.append("Direct assignment to the nearest training K-means center reproduces these targets with 100% accuracy by construction. SVM scores measure approximation of that assignment, not independent validation of physical flow states.")
        diagnostic_rows = _subsample(partitions["train"], 1000, "cluster_diagnostics")
        diagnostic_ids = assigned[diagnostic_rows]
        quality_notes = []
        if 1 < len(np.unique(diagnostic_ids)) < len(diagnostic_ids):
            silhouette = float(silhouette_score(angles[diagnostic_rows], diagnostic_ids, metric="euclidean"))
        else:
            silhouette = None
            quality_notes.append("Silhouette is undefined because the deterministic training subsample has insufficient class coverage.")
        training_ids = assigned[partitions["train"]]
        if 1 < len(np.unique(training_ids)) < len(training_ids):
            davies_bouldin = float(davies_bouldin_score(angles[partitions["train"]], training_ids))
        else:
            davies_bouldin = None
            quality_notes.append("Davies-Bouldin is undefined for this training partition.")
        state_discovery = {
            "algorithm": "KMeans",
            "fit_partition": "train",
            "n_clusters_requested": int(k),
            "training_full_class_counts": _label_counts(y[partitions["train"]]),
            "centers_in_angle_features": state_centers.tolist(),
            "silhouette_training_only": silhouette,
            "silhouette_sample_count": int(len(diagnostic_rows)),
            "silhouette_sampling": "At most 1000 deterministic evenly spaced training rows; Euclidean distances in the shared angle feature space.",
            "davies_bouldin_training_only": davies_bouldin,
            "davies_bouldin_sample_count": int(len(partitions["train"])),
            "diagnostic_notes": quality_notes,
            "nearest_center_assignment_baseline": {
                "accuracy_against_own_targets": 1.0,
                "interpretation": "Exact by construction because nearest training-center assignment defines the targets. This is not predictive validation or physical ground truth.",
            },
        }
    else:
        y = _as_labels(labels, len(X))
        label_mode = "external"
    train_classes = np.unique(y[train])
    if len(train_classes) < 2:
        raise ClassificationUnavailableError("The capped training partition contains fewer than two states. Increase max_train, revise the chronological partition, or collect more data; do not shuffle neighboring frames.")
    classes = np.unique(y[np.concatenate(list(partitions.values()))])
    # Test labels do not define even the class denominator used in tuning.
    selection_classes = np.unique(y[np.concatenate([train, validation])])
    for name, indices in sampled.items():
        missing = sorted(set(classes) - set(y[indices]))
        if missing:
            messages.append(f"Classes absent from sampled {name}: {', '.join(missing)}. Interpret aggregate metrics with class support.")
    unseen = sorted(set(y[np.concatenate([validation, test])]) - set(train_classes))
    if unseen:
        messages.append(f"Held-out classes absent from training cannot be learned: {', '.join(unseen)}.")
    if len(validation) < 20 or len(test) < 20:
        messages.append("Fewer than 20 evaluated validation or test frames: results are a pipeline pilot, not a publication-grade performance estimate.")

    C_values = config.get("C_values", [0.1, 1.0, 10.0])
    if not C_values or any(not np.isfinite(c) or float(c) <= 0 for c in C_values):
        raise ValueError("C_values must be a nonempty list of positive finite numbers.")
    gamma_values = config.get("gamma_values", [0.1, 1.0, "scale"])
    if not gamma_values or any((g not in {"scale", "auto"} if isinstance(g, str) else not np.isfinite(g) or float(g) <= 0) for g in gamma_values):
        raise ValueError("gamma_values must be positive finite numbers, 'scale', or 'auto'.")
    class_weight = config.get("class_weight", "balanced")
    started = perf_counter()
    best_rbf = None
    best_key = (-np.inf, -np.inf)
    rbf_search = []
    for C in C_values:
        for gamma in gamma_values:
            model = SVC(C=float(C), gamma=gamma, kernel="rbf", class_weight=class_weight)
            model.fit(angles[train], y[train])
            performance = _metrics(y[validation], model.predict(angles[validation]), selection_classes)
            rbf_search.append({"C": float(C), "gamma": gamma, **performance})
            key = (performance["balanced_accuracy"], performance["macro_f1"])
            if key > best_key:
                best_key, best_rbf = key, model
                best_rbf_parameters = {"C": float(C), "gamma": gamma}
    rbf_fit_seconds = perf_counter() - started

    repetitions = config.get("reps", 2)
    repetitions = [repetitions] if isinstance(repetitions, (int, np.integer)) else list(repetitions)
    if not repetitions or any(not isinstance(r, (int, np.integer)) or r < 1 for r in repetitions):
        raise ValueError("reps must be a positive integer or a nonempty list of positive integers.")
    entanglement = config.get("entanglement", "linear")
    quantum_search = []
    best_q_key = (-np.inf, -np.inf)
    best_q = None
    quantum_started = perf_counter()
    for reps in repetitions:
        train_states = feature_states(angles[train], reps=int(reps), entanglement=entanglement)
        validation_states = feature_states(angles[validation], reps=int(reps), entanglement=entanglement)
        K_train = fidelity_kernel(train_states)
        K_validation = fidelity_kernel(validation_states, train_states)
        for C in C_values:
            model = SVC(C=float(C), kernel="precomputed", class_weight=class_weight)
            model.fit(K_train, y[train])
            performance = _metrics(y[validation], model.predict(K_validation), selection_classes)
            quantum_search.append({"C": float(C), "reps": int(reps), **performance})
            key = (performance["balanced_accuracy"], performance["macro_f1"])
            if key > best_q_key:
                best_q_key, best_q = key, model
                best_q_parameters = {"C": float(C), "reps": int(reps), "entanglement": entanglement}
                best_train_states = train_states
                best_kernel = K_train
    quantum_fit_seconds = perf_counter() - quantum_started
    test_states = feature_states(angles[test], reps=best_q_parameters["reps"], entanglement=entanglement)
    K_test = fidelity_kernel(test_states, best_train_states)
    predictions = {
        "rbf_svm": best_rbf.predict(angles[test]),
        "qsvm": best_q.predict(K_test),
        "majority": np.full(len(test), Counter(y[train].tolist()).most_common(1)[0][0]),
    }
    test_metrics = {name: _metrics(y[test], prediction, classes) for name, prediction in predictions.items()}
    out_of_range = (X < scaler.data_min_) | (X > scaler.data_max_)
    clip_stats = {
        name: {
            "fraction_feature_values_clipped": float(np.mean(clipped[indices])),
            "fraction_rows_with_any_clipping": float(np.mean(np.any(clipped[indices], axis=1))),
            "fraction_feature_values_outside_training_range": float(np.mean(out_of_range[indices])),
            "fraction_rows_with_ignored_constant_feature_drift": float(np.mean(np.any(constant_drift[indices], axis=1))),
        }
        for name, indices in partitions.items()
    }
    if clip_stats["test"]["fraction_rows_with_any_clipping"] > 0.2:
        messages.append("More than 20% of test rows exceed training feature ranges and are clipped for both classifiers; investigate drift and sensitivity to the angle map.")
    messages.append("Temporal dependence is not removed by the split alone. Select an embargo from training-only correlation, use independent runs when possible, and assess uncertainty by temporal blocks.")
    messages.append("These kernels are noiseless exact classical statevector simulations; hardware noise, shots, compilation, and queue time are not included.")
    for message in messages:
        warnings.warn(message, UserWarning, stacklevel=2)
    metrics = {
        "label_mode": label_mode,
        "n_features": int(d),
        "n_qubits": int(d),
        "classes": classes.tolist(),
        "state_discovery": state_discovery,
        "test": test_metrics,
        "selected_hyperparameters": {"rbf_svm": best_rbf_parameters, "qsvm": best_q_parameters},
        "selection": "Highest validation balanced accuracy; macro-F1 breaks ties; first grid entry resolves remaining ties. No refit on validation.",
        "search": {"rbf_svm": rbf_search, "qsvm": quantum_search},
        "partition_counts": {name: int(len(indices)) for name, indices in partitions.items()},
        "sampled_counts": {name: int(len(indices)) for name, indices in sampled.items()},
        "sampled_class_counts": {name: _label_counts(y[indices]) for name, indices in sampled.items()},
        "sampling": "Deterministic evenly spaced row selection per partition for kernel cost only. POD and temporal analyses use all selected frames.",
        "angle_scaler": {"fit_partition": "train", "range": [0.0, float(np.pi)], "clip_heldout": True, "data_min": scaler.data_min_.tolist(), "data_max": scaler.data_max_.tolist(),
                         "training_constant_feature_indices": np.flatnonzero(constant_features).tolist(),
                         "constant_feature_policy": "Frozen at angle zero, including held-out rows; drift reported separately."},
        "clipping": clip_stats,
        "kernel_diagnostics": kernel_diagnostics(best_kernel),
        "feature_map": feature_map_metadata(d, reps=best_q_parameters["reps"], entanglement=entanglement),
        "timing_seconds": {"rbf_validation_search": rbf_fit_seconds, "quantum_states_kernels_validation_search": quantum_fit_seconds},
        "kernel_work": {"train_matrix_entries": int(len(train) ** 2), "validation_matrix_entries": int(len(validation) * len(train)), "test_matrix_entries": int(len(test) * len(train)), "statevector_dimension": int(2**d)},
        "warnings": messages,
    }
    return {
        "metrics": metrics,
        "labels": y,
        "label_mode": label_mode,
        "classes": classes,
        "test_indices": test,
        "train_indices": train,
        "validation_indices": validation,
        "predictions": predictions,
        "confusion": {name: confusion_matrix(y[test], prediction, labels=classes) for name, prediction in predictions.items()},
        "kernels": {"train": best_kernel, "test": K_test},
        "angles": angles,
        "state_centers": state_centers,
    }
