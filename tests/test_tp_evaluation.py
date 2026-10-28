"""Only genuine held-out classifier predictions are presented as TP tests."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from schlieren_qml.classify import _metrics
from schlieren_qml.tp_evaluation import export_tp_test_predictions


class TPTestPredictionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.result = self.root / "results"
        self.result.mkdir()
        self.states = self.root / "tp_states.csv"
        self.assignments = self.result / "state_assignments.csv"
        self.labels = ["candidate_1", "candidate_2", "candidate_3", "candidate_1", "candidate_2", "candidate_3"]
        self.rows = [dict(filename=f"f{i}.tif", sequence_frame_index=i, time_s=i / 51032,
                          partition="train" if i < 2 else "test", state=label,
                          rbf_test_prediction="", qsvm_test_prediction="")
                     for i, label in enumerate(self.labels)]
        for i in (2, 4, 5):
            self.rows[i]["rbf_test_prediction"] = self.labels[i]
            self.rows[i]["qsvm_test_prediction"] = "candidate_2"
        self.tp_rows = [dict(filename=f"f{i}.tif", sequence_frame_index=i,
                             x_px=30, y_px=20+i, label=label) for i, label in enumerate(self.labels)]
        classes = np.asarray(sorted(set(self.labels)))
        actual = np.asarray([self.labels[i] for i in (2, 4, 5)])
        self.metrics = {"label_mode": "external", "state_discovery": None,
                        "classes": classes.tolist(), "test": {
                            "rbf_svm": _metrics(actual, actual, classes),
                            "qsvm": _metrics(actual, np.asarray(["candidate_2"]*3), classes)}}
        self.save()

    def save(self):
        for path, rows in ((self.states, self.tp_rows), (self.assignments, self.rows)):
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        (self.result / "classification_metrics.json").write_text(json.dumps(self.metrics))

    def export(self):
        return export_tp_test_predictions(self.result, self.states)

    def test_exports_only_evaluated_test_rows_and_correct_flags(self):
        before = self.assignments.read_bytes()
        report = self.export()
        self.assertEqual(report["n_total_frames"], 6)
        self.assertEqual(report["n_test"], 3)
        self.assertEqual(report["models"]["rbf_svm"]["accuracy"], 1.)
        self.assertAlmostEqual(report["models"]["qsvm"]["accuracy"], 1/3)
        with Path(report["predictions_csv"]).open() as handle:
            exported = list(csv.DictReader(handle))
        self.assertEqual([row["filename"] for row in exported], ["f2.tif", "f4.tif", "f5.tif"])
        self.assertEqual([row["qsvm_correct"] for row in exported], ["False", "True", "False"])
        self.assertEqual([row["tp_y_px"] for row in exported], ["22", "24", "25"])
        self.assertEqual(self.assignments.read_bytes(), before)

    def test_label_mismatch_fails_before_replacing_export(self):
        report = self.export(); destination = Path(report["predictions_csv"])
        before = destination.read_bytes()
        self.tp_rows[0]["label"] = "candidate_2"; self.save()
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.export()
        self.assertEqual(destination.read_bytes(), before)

    def test_train_predictions_and_disjoint_sampled_rows_are_rejected(self):
        self.rows[0]["rbf_test_prediction"] = "candidate_1"
        self.rows[0]["qsvm_test_prediction"] = "candidate_1"
        self.save()
        with self.assertRaisesRegex(ValueError, "outside the test"):
            self.export()
        self.rows[0]["rbf_test_prediction"] = ""
        self.save()
        with self.assertRaisesRegex(ValueError, "same test rows"):
            self.export()

    def test_stale_metrics_and_cluster_labels_are_rejected(self):
        self.metrics["test"]["qsvm"]["accuracy"] = .95
        self.save()
        with self.assertRaisesRegex(ValueError, "metrics do not match"):
            self.export()
        self.metrics["label_mode"] = "cluster-surrogate"; self.save()
        with self.assertRaisesRegex(ValueError, "external TP-label"):
            self.export()

    def test_alias_output_does_not_change_source(self):
        target = self.result / "tp_test_predictions.csv"
        os.link(self.states, target)
        before = self.states.read_bytes()
        with self.assertRaisesRegex(ValueError, "alias"):
            self.export()
        self.assertEqual(self.states.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
