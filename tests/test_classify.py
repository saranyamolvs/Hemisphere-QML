"""Meaningful safeguards for temporal partitions and classifier preprocessing."""
import json
import unittest
import warnings

import numpy as np

from schlieren_qml.classify import ClassificationUnavailableError, make_splits, compare_classifiers


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

    def test_data_limited_classification_has_a_distinct_exception(self):
        with self.assertRaises(ClassificationUnavailableError):
            compare_classifiers(np.zeros_like(self.scores), None, self.splits, self.config)
        invalid_config = dict(self.config, n_features=50)
        with self.assertRaises(ValueError) as context:
            compare_classifiers(self.scores, self.labels, self.splits, invalid_config)
        self.assertNotIsInstance(context.exception, ClassificationUnavailableError)


if __name__ == "__main__":
    unittest.main()
