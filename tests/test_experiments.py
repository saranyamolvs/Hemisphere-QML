"""Saved-run studies preserve targets and leave the held-out test data unused."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
from schlieren_qml.experiments import validation_learning_curve


class SavedValidationStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root/"saved_run"
        self.source.mkdir()
        phase = np.arange(80)
        self.features = np.column_stack([np.sin(.7*phase), np.cos(.4*phase)])
        self.labels = np.where(self.features[:, 0] >= 0, "candidate_1", "candidate_2")
        classifier = {"n_features":2, "C_values":[1.], "gamma_values":[1.], "reps":[1]}
        (self.source/"config_used.json").write_text(json.dumps({"classifier":classifier}))
        (self.source/"classification_metrics.json").write_text(json.dumps({
            "label_mode":"cluster-surrogate", "n_features":2,
            "angle_scaler":{"fit_partition":"train", "range":[0.,np.pi],
                "data_min":self.features[:48].min(axis=0).tolist(),
                "data_max":self.features[:48].max(axis=0).tolist()},
            "test":{"sentinel":"this must not be consumed"}}))
        self._write_rows()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_rows(self, poison_test=False, wrong_assignment=False):
        with (self.source/"modal_coefficients.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "sequence_frame_index", "time_s", "partition", "a_1", "a_2"])
            for i, values in enumerate(self.features):
                partition = "train" if i < 48 else "validation" if i < 64 else "test"
                if poison_test and partition == "test":
                    values = ["not-a-number", "NaN"]
                writer.writerow([f"frame{i:03d}.tif", i, i/51032, partition, *values])
        with (self.source/"state_assignments.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "sequence_frame_index", "time_s", "partition", "state"])
            for i, label in enumerate(self.labels):
                partition = "train" if i < 48 else "validation" if i < 64 else "test"
                name = "wrong.tif" if wrong_assignment and i == 2 else f"frame{i:03d}.tif"
                if poison_test and partition == "test":
                    label = ""
                writer.writerow([name, i, i/51032, partition, label])

    def run_study(self, output, **kwargs):
        defaults = {"train_sizes":[8,16], "max_validation":10,
                    "angle_scales":[.25,1.], "dpi":30}
        defaults.update(kwargs)
        return validation_learning_curve(self.source, self.root/output, **defaults)

    def test_uses_frozen_assignments_shared_rows_and_actual_capped_counts(self):
        from schlieren_qml.classify import fit_validate_classifiers

        with patch("schlieren_qml.classify.fit_validate_classifiers", wraps=fit_validate_classifiers) as fit, \
             patch("schlieren_qml.classify.KMeans.fit", side_effect=AssertionError("Clustering was refitted")), \
             patch("schlieren_qml.pod.fit_pod", side_effect=AssertionError("POD was refitted")), \
             patch("schlieren_qml.data.read_image", side_effect=AssertionError("TIFF was read")):
            report = self.run_study("study", train_sizes=[8,16,100,1000])
        self.assertEqual([run["n_train"] for run in report["runs"]], [8,16,48])
        self.assertEqual(report["runs"][-1]["requested_train_sizes"], [100,1000])
        self.assertEqual(fit.call_count, 3)
        for call, run in zip(fit.call_args_list, report["runs"]):
            selected = run["train_sequence_frame_indices"]
            np.testing.assert_array_equal(call.args[1], self.labels[selected])
            self.assertEqual(len(call.args[0]), run["n_train"])
            self.assertEqual(len(call.args[2]), 10)
            self.assertEqual(call.args[4]["angle_scales"], [.25,1.])
        self.assertEqual(report["retained_counts"], {"train":48, "validation":16})
        self.assertNotIn("test", report)
        self.assertNotIn("test", report["runs"][0])
        self.assertEqual(len(report["figures"]), 3)
        self.assertTrue(all(Path(path).is_file() for path in report["figures"]))
        json.dumps(report, allow_nan=False)

    def test_poisoned_test_features_and_labels_cannot_change_selection(self):
        first = self.run_study("first", train_sizes=[12])
        self._write_rows(poison_test=True)
        second = self.run_study("second", train_sizes=[12])
        self.assertEqual(first["intended_classes"], second["intended_classes"])
        self.assertEqual(first["runs"], second["runs"])

    def test_saved_near_constant_scaler_matches_original_sklearn_transform(self):
        from sklearn.preprocessing import MinMaxScaler
        from schlieren_qml.classify import fit_validate_classifiers

        self.features[:, 0] *= 1e-16
        self._write_rows()
        metrics_path = self.source/"classification_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["angle_scaler"]["data_min"] = self.features[:48].min(axis=0).tolist()
        metrics["angle_scaler"]["data_max"] = self.features[:48].max(axis=0).tolist()
        metrics_path.write_text(json.dumps(metrics))
        scaler = MinMaxScaler(feature_range=(0.,np.pi)).fit(self.features[:48])
        with patch("schlieren_qml.classify.fit_validate_classifiers", wraps=fit_validate_classifiers) as fit:
            report = self.run_study("near_constant", train_sizes=[12])
        selected = report["runs"][0]["train_sequence_frame_indices"]
        expected = np.clip(scaler.transform(self.features[selected]), 0., np.pi)
        np.testing.assert_allclose(fit.call_args.args[0], expected, rtol=1e-14, atol=1e-20)

    def test_misalignment_and_scaler_mismatch_fail_before_output_replacement(self):
        output = self.root/"existing"
        output.mkdir()
        marker = output/"validation_learning_curve.json"
        marker.write_text("previous valid report")
        self._write_rows(wrong_assignment=True)
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.run_study("existing", overwrite=True)
        self.assertEqual(marker.read_text(), "previous valid report")
        self._write_rows()
        metrics_path = self.source/"classification_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["angle_scaler"]["data_min"][0] -= 1
        metrics_path.write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.run_study("existing", overwrite=True)
        self.assertEqual(marker.read_text(), "previous valid report")

    def test_source_output_aliases_and_existing_results_are_protected(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            validation_learning_curve(self.source, self.source)
        with self.assertRaisesRegex(ValueError, "separate"):
            validation_learning_curve(self.source, self.source/"study")
        output = self.root/"existing"
        output.mkdir()
        marker = output/"keep.txt"
        marker.write_text("unrelated")
        with self.assertRaises(FileExistsError):
            self.run_study("existing")
        self.run_study("existing", train_sizes=[8], overwrite=True)
        self.assertEqual(marker.read_text(), "unrelated")
        hardlink_folder = self.root/"hardlinks"
        hardlink_folder.mkdir()
        original = self.source/"config_used.json"
        original_bytes = original.read_bytes()
        os.link(original, hardlink_folder/"validation_learning_curve.json")
        with self.assertRaisesRegex(ValueError, "hard-linked"):
            self.run_study("hardlinks", overwrite=True)
        self.assertEqual(original.read_bytes(), original_bytes)

    def test_study_output_cannot_pollute_original_tiff_source_folder(self):
        image_folder = self.root/"tiff_source"
        image_folder.mkdir()
        config_path = self.source/"config_used.json"
        config = json.loads(config_path.read_text())
        config["image_directory"] = str(image_folder)
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "TIFF source"):
            self.run_study("tiff_source/study")
        self.assertFalse((image_folder/"study").exists())

    def test_invalid_study_settings_do_not_create_outputs(self):
        cases = [{"train_sizes":[]}, {"train_sizes":[True]}, {"max_validation":False},
                 {"angle_scales":[0.]}, {"angle_scales":[float("nan")]}, {"overwrite":1}]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.run_study("invalid", **settings)
            self.assertFalse((self.root/"invalid").exists())
