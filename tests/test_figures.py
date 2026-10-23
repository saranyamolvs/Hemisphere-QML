"""Regression checks on quantitative figure sources, without expensive exports."""
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "schlieren_test_mpl"))
os.environ.setdefault("MPLBACKEND", "Agg")
from schlieren_qml import figures


def _close_without_export(fig, directory, name, dpi):
    figures.plt.close(fig)
    return []


class FigureSourceTests(unittest.TestCase):
    @staticmethod
    def fixture():
        times = np.arange(200) * 3 / 1200
        scores = np.column_stack([np.sin(2*np.pi*50*times), np.cos(2*np.pi*75*times)])
        model = SimpleNamespace(mean=np.zeros(4), modes=np.eye(4)[:2], energy_fraction=np.array([.5,.5]))
        X = scores @ model.modes
        metadata = {"fps":1200., "stride":3, "bin_factor":2,
                    "frame_records":[{"filename":f"f{i:04d}.tif", "frame_index":i*3, "time_s":time}
                                     for i,time in enumerate(times)]}
        return X,model,scores,times,metadata

    def test_stride_has_correct_spectral_frequency_and_original_acf_lags(self):
        X,model,scores,times,metadata = self.fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(figures,"_save",side_effect=_close_without_export):
            figures.make_figures(X,(2,2),model,scores,times,np.arange(100),tmp,metadata,
                                 config={"psd_nperseg":80,"n_modes_plot":2})
            spectrum = np.genfromtxt(Path(tmp)/"06_modal_spectra.csv", delimiter=",",names=True)
            self.assertAlmostEqual(spectrum["frequency_hz"][np.argmax(spectrum["mode_1_psd"])],50.)
            acf = np.genfromtxt(Path(tmp)/"13_modal_autocorrelation.csv",delimiter=",",names=True)
            np.testing.assert_allclose(acf["lag_original_frames"],3*acf["lag_selected_samples"])
            np.testing.assert_allclose(acf["lag_s"],acf["lag_original_frames"]/1200)
            notes=(Path(tmp)/"FIGURE_NOTES.md").read_text()
            self.assertIn("4 segments",notes)
            self.assertIn("source filename f0199.tif",notes)

    def test_calibration_accounts_for_binning_and_rejects_fractional_bins(self):
        extent,unit=figures._extent((40,60),{"pixel_size_mm":.1,"bin_factor":2})
        self.assertEqual(extent,[0,12.,8.,0])
        self.assertEqual(unit,"mm")
        extent,unit=figures._extent((40,60),{"bin_factor":2})
        self.assertEqual(extent,[0,120,80,0])
        with self.assertRaisesRegex(ValueError,"bin_factor"):
            figures._extent((40,60),{"bin_factor":1.5})

    def test_import_does_not_replace_selected_matplotlib_backend(self):
        result=subprocess.run([sys.executable,"-c",
            "import matplotlib; matplotlib.use('svg'); import schlieren_qml.figures; "
            "assert matplotlib.get_backend().lower() == 'svg'"], capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_irregular_timestamps_do_not_generate_a_uniform_welch_spectrum(self):
        X,model,scores,times,metadata=self.fixture()
        times[120]+=.0001
        with tempfile.TemporaryDirectory() as tmp, patch.object(figures,"_save",side_effect=_close_without_export):
            figures.make_figures(X,(2,2),model,scores,times,np.arange(100),tmp,metadata,
                                 config={"psd_nperseg":80,"n_modes_plot":2})
            self.assertFalse((Path(tmp)/"06_modal_spectra.csv").exists())
            self.assertIn("timestamps are irregular",(Path(tmp)/"FIGURE_NOTES.md").read_text())

    def test_invalid_shape_and_fractional_training_indices_fail_before_export(self):
        X,model,scores,times,metadata=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"pixel count"):
                figures.make_figures(X,(2,3),model,scores,times,np.arange(100),tmp,metadata)
            with self.assertRaisesRegex(ValueError,"Training indices"):
                figures.make_figures(X,(2,2),model,scores,times,np.arange(100)+.2,tmp,metadata)
            with self.assertRaisesRegex(ValueError,"Training indices"):
                figures.make_figures(X,(2,2),model,scores,times,np.array([5,1,2],dtype=np.uint64),tmp,metadata)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_missing_state_support_is_saved_as_missing_not_zero_image(self):
        X=np.arange(24,dtype=float).reshape(6,4)
        result={"labels":np.array(["A","A","A","B","B","B"])}
        with tempfile.TemporaryDirectory() as tmp, patch.object(figures,"_save",side_effect=_close_without_export):
            figures._state_figures(X,(2,2),[0,2,2,0],"sensor pixels",np.arange(3),np.arange(6),
                                  result,Path(tmp),40,[],2)
            with np.load(Path(tmp)/"11_state_conditional_means.npz",allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["mean_images"][0],X[:3].mean(axis=0).reshape(2,2))
                self.assertTrue(np.isnan(saved["mean_images"][1]).all())
                np.testing.assert_array_equal(saved["training_counts"],[3,0])

    def test_kernel_figure_mapping_matches_actual_matrix_rows(self):
        result={"labels":np.array(["a"]*8),"train_indices":np.array([0,2,4]),
                "test_indices":np.array([7]),"kernels":{"train":np.eye(3),"test":np.array([[.1,.2,.3]])}}
        records=[{"filename":f"f{i}.tif","frame_index":i*2,"time_s":i/10} for i in range(8)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(figures,"_save",side_effect=_close_without_export):
            figures._classification_figures(result,Path(tmp),40,[],metadata={"frame_records":records})
            with (Path(tmp)/"10_quantum_kernel_samples.csv").open(newline="") as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([(r["partition"],r["kernel_example_index"],r["filename"]) for r in rows],
                             [("train","0","f0.tif"),("train","1","f2.tif"),("train","2","f4.tif"),("test","0","f7.tif")])


if __name__=="__main__":
    unittest.main()
