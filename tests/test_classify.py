"""Meaningful safeguards for temporal partitions and classifier preprocessing."""
import json
import unittest
import warnings

import numpy as np

from schlieren_qml.classify import (ClassificationUnavailableError, make_splits,
                                     compare_classifiers, fit_validate_classifiers)


class TemporalSplitTests(unittest.TestCase):
    def test_embargo_uses_original_frame_indices(self):
        original = np.arange(0, 200, 5)
        splits = make_splits(original, train_fraction=0.5, validation_fraction=0.25, gap_frames=10)
        self.assertEqual(splits["validation"][0], 22)
        self.assertEqual(splits["test"][0], 32)
        self.assertGreater(original[splits["validation"][0]] - original[splits["train"][-1]], 10)
        self.assertGreater(original[splits["test"][0]] - original[splits["validation"][-1]], 10)
        all_rows = np.concatenate(list(splits.values()))
        np.testing.assert_array_equal(np.sort(all_rows), np.arange(len(original)))
        self.assertEqual(len(np.unique(all_rows)), len(original))

    def test_too_large_gap_rejected(self):
        with self.assertRaisesRegex(ValueError, "embargo"):
            make_splits(np.arange(20), gap_frames=10)

    def test_unsorted_indices_rejected(self):
        with self.assertRaisesRegex(ValueError, "increasing"):
            make_splits(np.array([0, 1, 3, 2, 4]))
        # np.diff on unsigned arrays wraps a negative difference to a large
        # positive value, so order must be checked without subtraction.
        with self.assertRaisesRegex(ValueError, "increasing"):
            make_splits(np.array([0, 1, 3, 2, 4], dtype=np.uint32))
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            make_splits(np.array([0, 1, 2, 3], dtype=complex))


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        phase = np.arange(120)
        self.scores = np.column_stack([np.sin(phase * 0.7), np.cos(phase * 0.4)])
        self.labels = np.where(self.scores[:, 0] >= 0, "upper", "lower")
        self.splits = make_splits(phase, gap_frames=2)
        self.config = {"n_features": 2, "C_values": [1.0], "gamma_values": [1.0], "reps": 1, "max_train": 32, "max_validation": 16, "max_test": 16}

    def _compare(self, scores=None, labels=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return compare_classifiers(self.scores if scores is None else scores, self.labels if labels is None else labels, self.splits, self.config)

    def test_heldout_extremes_do_not_fit_scaler(self):
        scores = self.scores.copy()
        scores[self.splits["test"], 0] = 1e6
        result = self._compare(scores=scores)
        expected = np.max(scores[self.splits["train"]], axis=0)
        np.testing.assert_allclose(result["metrics"]["angle_scaler"]["data_max"], expected)
        np.testing.assert_allclose(result["angles"][self.splits["test"], 0], np.pi)
        self.assertEqual(result["metrics"]["clipping"]["test"]["fraction_rows_with_any_clipping"], 1.0)
        self.assertEqual(result["kernels"]["test"].shape, (16, 32))
        json.dumps(result["metrics"], allow_nan=False)

    def test_both_models_evaluate_identical_test_rows(self):
        result = self._compare()
        np.testing.assert_array_equal(result["test_indices"], self.splits["test"][[0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 18, 19, 21]])
        for model in ("rbf_svm", "qsvm", "majority"):
            self.assertEqual(len(result["predictions"][model]), len(result["test_indices"]))
            self.assertEqual(result["confusion"][model].sum(), len(result["test_indices"]))

    def test_missing_external_label_rejected(self):
        labels = self.labels.astype(object)
        labels[0] = None
        with self.assertRaisesRegex(ValueError, "missing"):
            self._compare(labels=labels)

    def test_train_defined_candidate_states(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = compare_classifiers(self.scores, None, self.splits, self.config)
        self.assertEqual(result["label_mode"], "cluster-surrogate")
        self.assertEqual(result["state_centers"].shape, (2, 2))
        self.assertIn("not independent", result["metrics"]["warnings"][0])
        quality = result["metrics"]["state_discovery"]
        self.assertEqual(sum(quality["training_full_class_counts"].values()), len(self.splits["train"]))
        self.assertLessEqual(quality["silhouette_sample_count"], 1000)
        self.assertTrue(-1 <= quality["silhouette_training_only"] <= 1)
        self.assertGreaterEqual(quality["davies_bouldin_training_only"], 0)
        self.assertEqual(quality["nearest_center_assignment_baseline"]["accuracy_against_own_targets"], 1.0)
        changed = self.scores.copy()
        changed[self.splits["test"]] = 1e6
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            changed_result = compare_classifiers(changed, None, self.splits, self.config)
        np.testing.assert_allclose(changed_result["state_centers"], result["state_centers"])
        self.assertEqual(changed_result["metrics"]["state_discovery"], result["metrics"]["state_discovery"])

    def test_test_labels_do_not_select_hyperparameters(self):
        first = self._compare()
        labels = self.labels.copy()
        labels[self.splits["test"]] = "new_state"
        second = self._compare(labels=labels)
        self.assertEqual(first["metrics"]["selected_hyperparameters"], second["metrics"]["selected_hyperparameters"])
        self.assertEqual(first["metrics"]["search"], second["metrics"]["search"])
        for model in first["predictions"]:
            np.testing.assert_array_equal(first["predictions"][model], second["predictions"][model])

    def test_single_class_test_is_flagged_without_redefining_metrics(self):
        labels = self.labels.copy()
        labels[self.splits["test"]] = "upper"
        result = self._compare(labels=labels)
        coverage = result["metrics"]["evaluation_coverage"]
        test = coverage["partitions"]["test"]
        self.assertFalse(coverage["full_class_coverage"])
        self.assertFalse(test["retained"]["full_class_coverage"])
        self.assertFalse(test["evaluated"]["full_class_coverage"])
        self.assertEqual(test["evaluated"]["observed_classes"], ["upper"])
        self.assertEqual(test["evaluated"]["missing_classes"], ["lower"])
        self.assertEqual(test["evaluated"]["n_frames"], len(result["test_indices"]))
        self.assertTrue(any("only one reference state" in message and "test" in message for message in result["metrics"]["warnings"]))
        for model, predictions in result["predictions"].items():
            reported = result["metrics"]["test"][model]
            expected_accuracy = float(np.mean(predictions == "upper"))
            self.assertEqual(reported["accuracy"], expected_accuracy)
            self.assertEqual(reported["balanced_accuracy"], expected_accuracy)
            # With only one true class, the other declared class has zero F1.
            present_f1 = 2*expected_accuracy/(1+expected_accuracy)
            self.assertAlmostEqual(reported["macro_f1"], present_f1/2)
        json.dumps(coverage, allow_nan=False)

    def test_complete_class_coverage_is_reported_for_each_population(self):
        result = self._compare()
        coverage = result["metrics"]["evaluation_coverage"]
        self.assertTrue(coverage["full_class_coverage"])
        self.assertEqual(coverage["intended_classes"], ["lower", "upper"])
        for name in ("train", "validation", "test"):
            report = coverage["partitions"][name]
            self.assertEqual(report["retained"]["n_frames"], len(self.splits[name]))
            self.assertEqual(report["evaluated"]["n_frames"], len(result[f"{name}_indices"]))
            for population in report.values():
                self.assertTrue(population["full_class_coverage"])
                self.assertEqual(population["missing_classes"], [])
                self.assertEqual(population["n_observed_classes"], 2)
                self.assertEqual(sum(population["class_counts"].values()), population["n_frames"])

    def test_sampling_loss_of_coverage_is_distinguished_from_retained_support(self):
        config = dict(self.config, max_validation=1, max_test=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = compare_classifiers(self.scores, self.labels, self.splits, config)
        for name in ("validation", "test"):
            report = result["metrics"]["evaluation_coverage"]["partitions"][name]
            self.assertTrue(report["retained"]["full_class_coverage"])
            self.assertFalse(report["evaluated"]["full_class_coverage"])
            self.assertEqual(report["evaluated"]["n_frames"], 1)
            self.assertTrue(any("only one reference state" in message and name in message
                                for message in result["metrics"]["warnings"]))

    def test_constant_training_feature_does_not_encode_heldout_drift(self):
        scores = self.scores.copy()
        scores[:, 1] = 7.0
        scores[self.splits["test"], 0] = 0.0
        unchanged = self._compare(scores=scores)
        scores[self.splits["test"], 1] = 7.25
        changed = self._compare(scores=scores)
        np.testing.assert_array_equal(changed["angles"][:, 1], 0.0)
        np.testing.assert_allclose(changed["kernels"]["test"], unchanged["kernels"]["test"])
        self.assertEqual(changed["metrics"]["angle_scaler"]["training_constant_feature_indices"], [1])
        stats = changed["metrics"]["clipping"]["test"]
        self.assertEqual(stats["fraction_rows_with_any_clipping"], 0.0)
        self.assertEqual(stats["fraction_rows_with_ignored_constant_feature_drift"], 1.0)
        self.assertEqual(stats["fraction_feature_values_outside_training_range"], 0.5)

    def test_default_quantum_scale_matches_explicit_one(self):
        first = self._compare()
        self.config = dict(self.config, angle_scales=[1.0])
        explicit = self._compare()
        self.assertEqual(first["metrics"]["search"], explicit["metrics"]["search"])
        self.assertEqual(first["metrics"]["test"], explicit["metrics"]["test"])
        np.testing.assert_array_equal(first["kernels"]["train"], explicit["kernels"]["train"])
        np.testing.assert_array_equal(first["kernels"]["test"], explicit["kernels"]["test"])
        self.assertEqual(explicit["metrics"]["selected_hyperparameters"]["qsvm"]["angle_scale"], 1.0)

    def test_quantum_scale_preserves_candidates_rbf_and_sample_rows(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            original = compare_classifiers(self.scores, None, self.splits, self.config)
            scaled = compare_classifiers(self.scores, None, self.splits,
                                         dict(self.config, angle_scales=[0.25]))
        np.testing.assert_array_equal(original["labels"], scaled["labels"])
        np.testing.assert_array_equal(original["angles"], scaled["angles"])
        np.testing.assert_array_equal(original["state_centers"], scaled["state_centers"])
        for partition in ("train_indices", "validation_indices", "test_indices"):
            np.testing.assert_array_equal(original[partition], scaled[partition])
        self.assertEqual(original["metrics"]["search"]["rbf_svm"], scaled["metrics"]["search"]["rbf_svm"])
        np.testing.assert_array_equal(original["predictions"]["rbf_svm"], scaled["predictions"]["rbf_svm"])
        self.assertGreater(np.max(np.abs(original["kernels"]["train"]-scaled["kernels"]["train"])), 0.1)
        self.assertEqual(scaled["metrics"]["feature_map"]["angle_scale"], 0.25)
        np.testing.assert_allclose(scaled["metrics"]["feature_map"]["encoded_input_range"], [0, np.pi/4])

    def test_multiscale_tuning_is_unchanged_by_test_features_or_labels(self):
        self.config = dict(self.config, angle_scales=[0.1, 0.5, 1.0], C_values=[0.1, 1.0])
        first = self._compare()
        changed_scores, changed_labels = self.scores.copy(), self.labels.copy()
        changed_scores[self.splits["test"]] = 1e5
        changed_labels[self.splits["test"]] = "never_in_training"
        second = self._compare(scores=changed_scores, labels=changed_labels)
        self.assertEqual(first["metrics"]["search"], second["metrics"]["search"])
        self.assertEqual(first["metrics"]["selected_hyperparameters"], second["metrics"]["selected_hyperparameters"])
        self.assertEqual(first["metrics"]["training"], second["metrics"]["training"])
        self.assertEqual(first["metrics"]["validation"], second["metrics"]["validation"])

    def test_validation_only_helper_reports_training_and_selects_from_grid(self):
        # These rows already have training-fitted features; no held-out rows are supplied.
        angles = (self.scores + 1.0) * np.pi/2
        train, validation = self.splits["train"], self.splits["validation"]
        result = fit_validate_classifiers(angles[train], self.labels[train],
                                         angles[validation], self.labels[validation],
                                         dict(self.config, angle_scales=[0.25, 1.0], reps=[1, 2]))
        self.assertEqual(set(result["training"]), {"rbf_svm", "qsvm", "majority"})
        self.assertEqual(set(result["validation"]), {"rbf_svm", "qsvm", "majority"})
        self.assertEqual(result["kernel_train"].shape, (len(train), len(train)))
        self.assertNotIn("test", result)
        self.assertNotIn("test_indices", result)
        rows = result["search"]["qsvm"]
        best = max(rows, key=lambda row: (row["balanced_accuracy"], row["macro_f1"]))
        selected = result["selected_hyperparameters"]["qsvm"]
        self.assertEqual((selected["C"], selected["reps"], selected["angle_scale"]),
                         (best["C"], best["reps"], best["angle_scale"]))
        for metrics in result["validation"].values():
            self.assertEqual(metrics["n"], len(validation))
        for metrics in result["training"].values():
            self.assertEqual(metrics["n"], len(train))

    def test_invalid_quantum_scale_grid_is_rejected(self):
        train, validation = self.splits["train"], self.splits["validation"]
        for scales in ([], [0], [-0.1], [True], [np.nan], [np.inf], [1e308], "0.5", 1.0):
            with self.subTest(scales=scales), self.assertRaisesRegex(ValueError, "angle_scales"):
                fit_validate_classifiers(self.scores[train], self.labels[train],
                                         self.scores[validation], self.labels[validation],
                                         dict(self.config, angle_scales=scales))

    def test_validation_only_helper_rejects_bad_features_and_labels(self):
        train, validation = self.splits["train"], self.splits["validation"]
        bad = self.scores[validation].copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_validate_classifiers(self.scores[train], self.labels[train], bad,
                                     self.labels[validation], self.config)
        with self.assertRaisesRegex(ValueError, "matching"):
            fit_validate_classifiers(self.scores[train], self.labels[train],
                                     self.scores[validation, :1], self.labels[validation], self.config)
        with self.assertRaises(ClassificationUnavailableError):
            fit_validate_classifiers(self.scores[train], np.full(len(train), "one_state"),
                                     self.scores[validation], self.labels[validation], self.config)

    def test_data_limited_classification_has_a_distinct_exception(self):
        with self.assertRaises(ClassificationUnavailableError):
            compare_classifiers(np.zeros_like(self.scores), None, self.splits, self.config)
        invalid_config = dict(self.config, n_features=50)
        with self.assertRaises(ValueError) as context:
            compare_classifiers(self.scores, self.labels, self.splits, invalid_config)
        self.assertNotIsInstance(context.exception, ClassificationUnavailableError)


if __name__ == "__main__":
    unittest.main()
