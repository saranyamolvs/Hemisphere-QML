import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from PIL import Image

from schlieren_qml.cli import _load_labels, _reconstruction_metrics, _validate_config, inspect_image, select_roi, run, write_json
from schlieren_qml.pod import fit_pod


class IntegrationContractTests(unittest.TestCase):
    def test_reported_reconstruction_error_matches_explicit_residual(self):
        rng = np.random.default_rng(31)
        X = rng.normal(size=(30, 12))
        partitions = {"train": np.arange(18), "validation": np.arange(18, 24), "test": np.arange(24, 30)}
        model = fit_pod(X, partitions["train"], rank=4, oversampling=8)
        coefficients = model.transform(X)
        report = _reconstruction_metrics(X, model, coefficients, partitions, 5)
        for name, rows in partitions.items():
            denominator = np.sum((X[rows]-model.mean)**2)
            for rank in (1, 3, 4):
                residual = X[rows]-model.reconstruct(coefficients[rows], rank=rank)
                expected = np.sqrt(np.sum(residual**2)/denominator)
                self.assertAlmostEqual(report[name]["relative_rms_fluctuation_error"][rank-1], expected, places=12)

    def test_labels_match_selected_filenames_and_reject_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"labels.csv"
            with path.open("w", newline="") as handle:
                csv.writer(handle).writerows([["filename", "label"], ["f2.tif", "b"], ["f1.tif", "a"]])
            np.testing.assert_array_equal(_load_labels(path, ["f1.tif", "f2.tif"]), ["a", "b"])
            with self.assertRaisesRegex(ValueError, "missing"):
                _load_labels(path, ["f3.tif"])
            with path.open("a", newline="") as handle:
                csv.writer(handle).writerow(["f1.tif", "a"])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                _load_labels(path, ["f1.tif"])

    def test_near_exact_reconstruction_is_not_a_difference_of_large_norms(self):
        rng = np.random.default_rng(62)
        X = rng.normal(size=(30, 2))@rng.normal(size=(2, 20))+10000
        partitions = {"train": np.arange(18), "validation": np.arange(18, 24), "test": np.arange(24, 30)}
        model = fit_pod(X, partitions["train"], rank=2)
        result = _reconstruction_metrics(X, model, model.transform(X), partitions, 5)
        for value in result.values():
            self.assertLess(value["relative_rms_fluctuation_error"][-1], 1e-10)

    def test_malformed_label_rows_fail_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"labels.csv"
            path.write_text("filename,label\nf1.tif\n")
            with self.assertRaisesRegex(ValueError, "row 2"):
                _load_labels(path, ["f1.tif"])

    def test_duplicate_headers_and_extra_label_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"labels.csv"
            path.write_text("filename,label,label\nf1.tif,a,b\n")
            with self.assertRaisesRegex(ValueError, "duplicate column"):
                _load_labels(path, ["f1.tif"])
            path.write_text("filename,label\nf1.tif,a,b\n")
            with self.assertRaisesRegex(ValueError, "row 2"):
                _load_labels(path, ["f1.tif"])

    def test_invalid_settings_are_rejected_before_outputs_are_replaced(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base/"results"
            output.mkdir()
            previous = output/"run_manifest.json"
            previous.write_text('{"status":"complete"}')
            cfg = {"image_directory": str(base/"images"), "output_directory": str(output), "classifier": {"C_values": [-1]}}
            config_path = base/"config.json"
            config_path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "C_values"):
                run(config_path, overwrite=True)
            self.assertEqual(previous.read_text(), '{"status":"complete"}')
            self.assertFalse((output/"image_matrix.npy").exists())

    def test_nested_input_and_output_paths_are_rejected_without_mutating_images(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            image = base/"f1.tif"
            Image.fromarray(np.arange(64, dtype=np.uint16).reshape(8, 8)).save(image)
            original = image.read_bytes()
            cfg = {"image_directory": str(base), "output_directory": str(base/"results")}
            config_path = base/"config.json"
            config_path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "separate"):
                run(config_path, overwrite=True)
            self.assertEqual(image.read_bytes(), original)

    def test_preview_and_roi_outputs_cannot_overwrite_the_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"f1.tif"
            path.write_bytes(b"source image bytes")
            for function in (inspect_image, select_roi):
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    function(path, path)
                self.assertEqual(path.read_bytes(), b"source image bytes")

    def test_configuration_types_and_unknown_keys_are_rejected(self):
        basic = {"image_directory": "images", "output_directory": "results"}
        for change in ({"stride": 1.0}, {"fps": True}, {"classifier": {"entanglement": []}}, {"fbs": 51032}, {"pattern": "../*.tif"}, {"pattern": "C:/images/*.tif"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                _validate_config({**basic, **change})

    def test_nonfinite_json_is_not_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"metrics.json"
            with self.assertRaises(ValueError):
                write_json(path, {"error": np.nan})
            self.assertFalse(path.exists())

    def test_quantum_angle_scales_are_validated_before_processing(self):
        basic = {"image_directory": "images", "output_directory": "results"}
        _validate_config({**basic, "classifier": {"angle_scales": [0.1, 0.25, 0.5, 1.0]}})
        for value in ([], 0.5, [True], [0.0], [-1.0], [float("nan")], [float("inf")]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "angle_scales"):
                _validate_config({**basic, "classifier": {"angle_scales": value}})

    def test_unavailable_classification_is_recorded_as_partial_not_success(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            images = base/"images"
            images.mkdir()
            rng = np.random.default_rng(9)
            names = []
            for index in range(30):
                name = f"frame{index:03d}.tif"
                Image.fromarray(rng.integers(0, 1000, size=(8, 8), dtype=np.uint16)).save(images/name)
                names.append(name)
            labels = base/"labels.csv"
            with labels.open("w", newline="") as handle:
                csv.writer(handle).writerows([["filename", "label"], *[[name, "one_state"] for name in names]])
            cfg = {"image_directory": str(images), "output_directory": str(base/"results"), "rank": 2,
                   "gap_frames": 0, "labels_csv": str(labels), "classifier": {"n_features": 2}}
            config_path = base/"config.json"
            config_path.write_text(json.dumps(cfg))
            with patch("schlieren_qml.figures.make_figures", return_value=[]), patch("schlieren_qml.figures.plot_roi", return_value=[]), patch("builtins.print"):
                report = run(config_path)
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["classification_status"], "unavailable")
            self.assertTrue((base/"results/pod_model.npz").exists())
            self.assertFalse((base/"results/classification_metrics.json").exists())
