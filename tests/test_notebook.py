from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from schlieren_qml.notebook import resolve_notebook_config, run_from_notebook, replot_spatial_modes


class NotebookHelperTests(unittest.TestCase):
    def test_saved_mode_recoloring_uses_no_images_and_preserves_original_results(self):
        import numpy as np
        from schlieren_qml.pod import fit_pod

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rng = np.random.default_rng(32)
            matrix = rng.normal(size=(12, 6))
            model = fit_pod(matrix, np.arange(8), rank=2)
            model.save(folder/"pod_model.npz")
            (folder/"frames.json").write_text(json.dumps({"image_shape": [2, 3], "bin_factor": 2}))
            (folder/"config_used.json").write_text(json.dumps({"figures": {"pixel_size_mm": 0.2}}))
            originals = {name: (folder/name).read_bytes() for name in ("pod_model.npz", "frames.json", "config_used.json")}
            (folder/"figures").mkdir()
            original_plot = folder/"figures/04_spatial_modes.png"
            original_plot.write_bytes(b"original figure")
            with patch("schlieren_qml.data.read_image", side_effect=AssertionError("Raw image was opened")), patch("subprocess.Popen") as process:
                result = replot_spatial_modes(folder, vmax=0.1, n_modes=2, dpi=40)
            process.assert_not_called()
            self.assertEqual(len(result["figures"]), 3)
            self.assertTrue(all(Path(path).is_file() for path in result["figures"]))
            self.assertTrue(all("mode_color_adjusted" in Path(path).parts for path in result["figures"]))
            self.assertTrue(result["color_scale"])
            for name, content in originals.items():
                self.assertEqual((folder/name).read_bytes(), content)
            self.assertEqual(original_plot.read_bytes(), b"original figure")

    def project(self, root):
        project = root/"project with spaces"
        (project/"schlieren_qml").mkdir(parents=True)
        (project/"schlieren_qml/cli.py").touch()
        config = {"image_directory": str(root/"TIFF images"), "output_directory": str(root/"results")}
        return project, config

    def test_resolution_preserves_original_settings_and_rejects_nested_folders(self):
        config = {"image_directory": "images", "output_directory": "results", "figures": {"dpi": 100}}
        original = deepcopy(config)
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            resolved = resolve_notebook_config(config, project)
            self.assertEqual(config, original)
            self.assertEqual(resolved["image_directory"], str(project/"images"))
            with self.assertRaisesRegex(ValueError, "separate"):
                resolve_notebook_config({**config, "output_directory": "images/results"}, project)

    def test_same_interpreter_and_partial_status_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            project, config = self.project(Path(temporary))
            def start(command, **options):
                self.assertEqual(command[0], sys.executable)
                self.assertIn("schlieren_qml", command)
                self.assertEqual(options["cwd"], project)
                config_path = next(Path(arg) for arg in command if str(arg).endswith(".json"))
                settings = json.loads(config_path.read_text())
                output = Path(settings["output_directory"])
                output.mkdir()
                (output/"run_manifest.json").write_text(json.dumps({
                    "status": "partial", "classification_status": "unavailable",
                    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                }))
                class Process:
                    stdout = io.StringIO("POD complete\n")
                    def wait(self): return 1
                return Process()
            with patch("subprocess.Popen", side_effect=start), patch("builtins.print"):
                manifest = run_from_notebook(config, project)
            self.assertEqual(manifest["classification_status"], "unavailable")

    def test_existing_results_do_not_start_another_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            project, config = self.project(Path(temporary))
            output = Path(config["output_directory"])
            output.mkdir()
            (output/"run_manifest.json").write_text("{}")
            with patch("subprocess.Popen") as start, self.assertRaises(FileExistsError):
                run_from_notebook(config, project)
            start.assert_not_called()

    def test_interrupt_terminates_the_processing_child_and_closes_stdout(self):
        class InterruptingOutput:
            closed = False
            def __iter__(self): return self
            def __next__(self): raise KeyboardInterrupt
            def close(self): self.closed = True
        class Process:
            stdout = InterruptingOutput()
            terminated = False
            waited = False
            def poll(self): return None
            def terminate(self): self.terminated = True
            def wait(self, timeout=None): self.waited = True; return -15
        process = Process()
        with tempfile.TemporaryDirectory() as temporary:
            project, config = self.project(Path(temporary))
            with patch("subprocess.Popen", return_value=process), self.assertRaises(KeyboardInterrupt):
                run_from_notebook(config, project)
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)
        self.assertTrue(process.stdout.closed)
