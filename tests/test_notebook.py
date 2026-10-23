from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from schlieren_qml.notebook import resolve_notebook_config, run_from_notebook


class NotebookHelperTests(unittest.TestCase):
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
