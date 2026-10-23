import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from schlieren_qml.data import build_image_matrix, list_images, preprocess_image, read_image
from schlieren_qml.pod import fit_pod, load_pod


class DataAndPODTests(unittest.TestCase):
    def test_crop_area_binning_and_16_bit_intensity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame1.tiff"
            image = (np.arange(48).reshape(6, 8) * 1000).astype(np.uint16)
            Image.fromarray(image).save(path)
            read = read_image(path)
            np.testing.assert_array_equal(read, image)
            got = preprocess_image(read, [2, 0, 6, 4], 2)
            expected = image[:4, 2:6].reshape(2, 2, 2, 2).mean(axis=(1, 3))
            np.testing.assert_array_equal(got, expected)
            with self.assertRaises(ValueError):
                preprocess_image(read, [0, 0, 7, 6], 2)

    def test_natural_order_and_original_times(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name, value in [("image10.png", 10), ("image2.png", 2), ("image1.png", 1)]:
                Image.fromarray(np.full((4, 6), value, dtype=np.uint8)).save(folder / name)
            paths = list_images(folder)
            self.assertEqual([p.name for p in paths], ["image1.png", "image2.png", "image10.png"])
            matrix, metadata = build_image_matrix(paths, folder / "cache.npy", fps=1000,
                                                  start=0, limit=2, stride=2)
            np.testing.assert_array_equal(matrix[:, 0], [1, 10])
            self.assertEqual(metadata["frame_records"][1]["time_s"], .002)
            self.assertEqual(metadata["effective_fps"], 500)

    def test_rank_three_reconstruction_and_training_only_mean(self):
        random = np.random.default_rng(4)
        basis, _ = np.linalg.qr(random.standard_normal((30, 3)))
        coefficients = random.standard_normal((35, 3))
        X = 400 + coefficients @ basis.T
        X[25:] += 50  # held-out brightness offset must not affect fitted centering
        model = fit_pod(X, np.arange(25), rank=3, batch_size=7, oversampling=4,
                        power_iterations=1, seed=0)
        np.testing.assert_allclose(model.mean, X[:25].mean(axis=0), atol=1e-12)
        reconstruction = model.reconstruct(model.transform(X[:25], batch_size=6))
        np.testing.assert_allclose(reconstruction, X[:25], atol=1e-10)
        np.testing.assert_allclose(model.modes @ model.modes.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(model.energy_fraction.sum(), 1, places=11)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pod.npz"
            model.save(path)
            restored = load_pod(path)
            np.testing.assert_array_equal(restored.transform(X), model.transform(X))

    def test_retained_variance_keeps_truncation_tail(self):
        random = np.random.default_rng(5)
        X = random.normal(size=(50, 20))
        model = fit_pod(X, np.arange(40), rank=2, oversampling=18)
        exact_s = np.linalg.svd(X[:40] - X[:40].mean(axis=0), compute_uv=False)
        np.testing.assert_allclose(model.energy_fraction,
                                   exact_s[:2] ** 2 / (exact_s ** 2).sum(), rtol=1e-12)
        self.assertLess(model.energy_fraction.sum(), 1)

    def test_manifest_destination_is_checked_before_cache_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            raw = folder / "raw.tif"
            Image.fromarray(np.ones((4, 4), dtype=np.uint16)).save(raw)
            cache = folder / "images.npy"
            cache.write_bytes(b"existing cache must survive invalid destinations")
            before = cache.read_bytes()
            with self.assertRaisesRegex(ValueError, "different files"):
                build_image_matrix([raw], cache, manifest_path=cache)
            self.assertEqual(cache.read_bytes(), before)

    def test_manifest_cannot_overwrite_an_unselected_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            paths = [folder / "raw1.tif", folder / "raw2.tif"]
            for path in paths:
                Image.fromarray(np.full((4, 4), 1234, dtype=np.uint16)).save(path)
            before = paths[1].read_bytes()
            with self.assertRaisesRegex(ValueError, "source image"):
                build_image_matrix(paths, folder / "images.npy", limit=1, manifest_path=paths[1])
            self.assertEqual(paths[1].read_bytes(), before)
            self.assertFalse((folder / "images.npy").exists())

    def test_cmyk_image_is_converted_before_luminance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"color.tif"
            image = Image.new("CMYK", (3, 2), color=(255, 0, 0, 0))
            image.save(path)
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            expected = rgb @ np.asarray([.2126, .7152, .0722], dtype=np.float32)
            np.testing.assert_allclose(read_image(path), expected)

    def test_integer_settings_reject_invalid_values_consistently(self):
        X = np.arange(60, dtype=float).reshape(10, 6)
        for invalid in (True, 1.5, np.nan, np.inf, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    fit_pod(X, np.arange(8), rank=2, batch_size=invalid)
                with self.assertRaises(ValueError):
                    preprocess_image(X, roi=[0, 0, invalid, 4])
        model = fit_pod(X, np.arange(8), rank=2)
        with self.assertRaises(ValueError):
            model.transform(X, batch_size=1.5)
        with self.assertRaises(ValueError):
            model.reconstruct(np.ones(2), rank=1.5)

    def test_reconstruction_rejects_nonfinite_or_insufficient_scores(self):
        X = np.random.default_rng(23).normal(size=(18, 9))
        model = fit_pod(X, np.arange(14), rank=3)
        for scores in (np.array([1., np.nan, 2.]), np.ones((2, 1)), np.array(2.)):
            with self.subTest(shape=scores.shape), self.assertRaises(ValueError):
                model.reconstruct(scores)

    def test_stabilized_power_iteration_matches_svd_for_decaying_spectrum(self):
        rng = np.random.default_rng(19)
        left = rng.normal(size=(75, 24))
        left -= left.mean(axis=0)
        left, _ = np.linalg.qr(left)
        right, _ = np.linalg.qr(rng.normal(size=(90, 24)))
        singular_values = np.geomspace(1., 1e-10, 24)
        X = (left*singular_values) @ right.T
        model = fit_pod(X, np.arange(75), rank=12, oversampling=8,
                        power_iterations=2, batch_size=11, seed=7)
        exact = np.linalg.svd(X-X.mean(axis=0), compute_uv=False)
        np.testing.assert_allclose(model.singular_values, exact[:12], rtol=1e-8, atol=1e-14)
        np.testing.assert_allclose(model.modes @ model.modes.T, np.eye(12), atol=1e-12)
        np.testing.assert_allclose(np.sum(model.transform(X)**2, axis=0), model.singular_values**2, rtol=1e-8)

    def test_tiny_scale_variance_fractions_do_not_underflow(self):
        rng = np.random.default_rng(45)
        left = rng.normal(size=(24, 2))
        left -= left.mean(axis=0)
        left, _ = np.linalg.qr(left)
        right, _ = np.linalg.qr(rng.normal(size=(12, 2)))
        X = (left*np.array([1e-150, 1e-162])) @ right.T
        model = fit_pod(X, np.arange(24), rank=2, oversampling=2, power_iterations=2)
        self.assertGreater(model.energy_fraction[1], 5e-25)
        self.assertLess(model.energy_fraction[1], 2e-24)


if __name__ == "__main__":
    unittest.main()
