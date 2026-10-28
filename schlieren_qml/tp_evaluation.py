"""Inspect the saved held-out predictions against TP-derived labels."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .classify import _metrics
from .triple_point import _write_csv


def _rows(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required).issubset(fields):
            raise ValueError(f"Missing or duplicate columns in {Path(path).name}.")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise ValueError(f"Invalid or empty rows in {Path(path).name}.")
    names = [row["filename"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate filenames in {Path(path).name}.")
    return rows


def export_tp_test_predictions(results_directory, states_csv):
    """Export only genuinely evaluated test rows, with TP coordinates and errors.

    The source pipeline caps kernel evaluation cost. Blank predictions on other
    frames stay unevaluated; this helper does not train, predict or extrapolate.
    Existing CSV output can be refreshed, but file aliases are never replaced.
    """
    results = Path(results_directory).expanduser().resolve()
    states_csv = Path(states_csv).expanduser().resolve()
    assignments_csv = results / "state_assignments.csv"
    metrics_path = results / "classification_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("label_mode") != "external" or metrics.get("state_discovery") is not None:
        raise ValueError("Expected an external TP-label experiment without clustering.")

    assignments = _rows(assignments_csv, ["filename", "sequence_frame_index", "partition",
                                         "state", "rbf_test_prediction", "qsvm_test_prediction"])
    states = _rows(states_csv, ["filename", "sequence_frame_index", "x_px", "y_px", "label"])
    lookup = {row["filename"]: row for row in states}
    if set(lookup) != {row["filename"] for row in assignments}:
        raise ValueError("TP states and classifier assignments have different filenames.")
    classes = np.asarray(metrics.get("classes", []), dtype=str)
    if len(classes) < 2 or len(set(classes.tolist())) != len(classes):
        raise ValueError("Classification metrics contain invalid classes.")
    allowed = set(classes.tolist())
    exported = []
    for row in assignments:
        tp = lookup[row["filename"]]
        if (row["sequence_frame_index"] != tp["sequence_frame_index"]
                or row["state"] != tp["label"] or row["state"] not in allowed):
            raise ValueError("TP states and classifier labels/index alignment differ.")
        xy = np.asarray([float(tp["x_px"]), float(tp["y_px"])])
        if not np.isfinite(xy).all():
            raise ValueError("TP coordinates must be finite.")
        rbf, qsvm = row["rbf_test_prediction"], row["qsvm_test_prediction"]
        if bool(rbf) != bool(qsvm):
            raise ValueError("Both classifiers must evaluate the same test rows.")
        if not rbf:
            continue
        if row["partition"] != "test":
            raise ValueError("Saved test predictions occur outside the test partition.")
        if rbf not in allowed or qsvm not in allowed:
            raise ValueError("Saved prediction is outside the declared classes.")
        exported.append({
            "filename": row["filename"],
            "sequence_frame_index": row["sequence_frame_index"],
            "time_s": row.get("time_s", tp.get("time_s", "")),
            "partition": "test", "tp_x_px": tp["x_px"], "tp_y_px": tp["y_px"],
            "actual_category": tp["label"], "rbf_prediction": rbf,
            "rbf_correct": rbf == tp["label"], "qsvm_prediction": qsvm,
            "qsvm_correct": qsvm == tp["label"],
        })
    if not exported:
        raise ValueError("No evaluated held-out test predictions are available.")

    actual = np.asarray([row["actual_category"] for row in exported])
    summaries = {}
    for model, column in [("rbf_svm", "rbf_prediction"), ("qsvm", "qsvm_prediction")]:
        measured = _metrics(actual, np.asarray([row[column] for row in exported]), classes)
        saved = metrics.get("test", {}).get(model, {})
        if saved.get("n") != measured["n"] or any(
            name not in saved or not np.isclose(saved[name], measured[name], rtol=1e-10, atol=1e-12)
            for name in ("accuracy", "balanced_accuracy", "macro_f1")
        ):
            raise ValueError("Saved metrics do not match the exported test predictions.")
        summaries[model] = {name: saved[name] for name in ("accuracy", "balanced_accuracy", "macro_f1")}
        summaries[model]["n_test"] = saved["n"]

    destination = results / "tp_test_predictions.csv"
    for source in (states_csv, assignments_csv, metrics_path):
        if destination.exists() and destination.samefile(source):
            raise ValueError("Prediction export must not overwrite a source file alias.")
        if destination.resolve() == source.resolve():
            raise ValueError("Prediction export must not overwrite a source file.")
    _write_csv(destination, list(exported[0]), exported)
    return {"predictions_csv": str(destination), "n_test": len(exported),
            "n_total_frames": len(assignments), "models": summaries}
