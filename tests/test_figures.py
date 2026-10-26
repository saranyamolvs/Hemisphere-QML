"""Regression checks on quantitative figure sources, without expensive exports."""
import csv
import json
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

    def test_confusion_figure_reports_counts_label_source_and_computed_balanced_accuracy(self):
        counts = np.array([[7,1,0],[0,3,1],[0,0,2]])
        other = np.array([[6,2,0],[1,2,1],[0,1,1]])
        classes = ["candidate_1", "candidate_2", "candidate_3"]
        result = {"labels":np.array(classes), "label_mode":"cluster-surrogate",
                  "metrics":{"classes":classes, "test":{"rbf_svm":{"balanced_accuracy":0.0}}},
                  "confusion":{"rbf_svm":counts, "qsvm":other}}
        captured = []
        def inspect_and_close(fig, directory, name, dpi):
            self.assertEqual(name, "09_confusion_matrices")
            image_axes = [ax for ax in fig.axes if ax.images]
            self.assertEqual(len(image_axes), 2)
            self.assertIn("N = 14 held-out test examples", fig._suptitle.get_text())
            self.assertIn("training-defined K-means assignments", fig._suptitle.get_text())
            self.assertIn("Balanced accuracy: 87.5%", image_axes[0].get_title())
            self.assertIn("Balanced accuracy: 58.3%", image_axes[1].get_title())
            for ax in image_axes:
                self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["Candidate 1", "Candidate 2", "Candidate 3"])
                self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ["Candidate 1", "Candidate 2", "Candidate 3"])
                self.assertEqual(ax.images[0].get_clim(), (0,1))
            self.assertEqual(image_axes[0].texts[0].get_text(), "87.5%\n(n=7)")
            np.testing.assert_allclose(image_axes[0].images[0].get_array(), counts/counts.sum(axis=1,keepdims=True))
            captured.append(True)
            figures.plt.close(fig)
            return []
        with tempfile.TemporaryDirectory() as tmp, patch.object(figures,"_save",side_effect=inspect_and_close):
            notes = []
            figures._classification_figures(result,Path(tmp),40,notes)
            with (Path(tmp)/"09_confusion_matrices.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(sum(int(row["count"]) for row in rows if row["model"]=="rbf_svm"),14)
            self.assertEqual(rows[0]["reference_state"], "candidate_1")
            self.assertEqual(rows[0]["fraction_within_reference"], "0.875")
            self.assertIn("training-defined K-means assignments", notes[0])
        self.assertEqual(captured, [True])

    def test_confusion_figure_does_not_invent_label_source_or_absent_state_recall(self):
        result = {"labels":np.array(["one","two"]),
                  "confusion":{"rbf_svm":np.array([[1,1],[0,0]])}}
        def inspect_and_close(fig, directory, name, dpi):
            ax = next(ax for ax in fig.axes if ax.images)
            self.assertIn("Balanced accuracy: 50.0%", ax.get_title())
            self.assertIn("source not specified", fig._suptitle.get_text())
            self.assertIn("N = 2", fig._suptitle.get_text())
            self.assertEqual(ax.texts[2].get_text(), "No test\nsamples")
            figures.plt.close(fig)
            return []
        with tempfile.TemporaryDirectory() as tmp, patch.object(figures,"_save",side_effect=inspect_and_close):
            figures._classification_figures(result,Path(tmp),40,[])

    def test_manual_mode_scale_is_shared_symmetric_and_does_not_modify_modes(self):
        modes = np.array([[1.,-1.,0.,0.],[0.,0.,1.,-1.]])/np.sqrt(2)
        original = modes.copy()
        model = SimpleNamespace(modes=modes,energy_fraction=np.array([.5,.5]))
        checked = []
        def inspect_and_close(fig,directory,name,dpi):
            image_axes = [ax for ax in fig.axes if ax.images]
            self.assertEqual(len(image_axes),2)
            for ax in image_axes:
                self.assertEqual(ax.images[0].get_clim(),(-.2,.2))
                np.testing.assert_array_equal(ax.images[0].get_array(),original[len(checked)%2].reshape(2,2))
                checked.append(True)
            np.testing.assert_allclose(fig.axes[-1].get_yticks(),[-.2,-.1,0,.1,.2])
            self.assertEqual(fig.axes[-1].get_ylim(),(-.2,.2))
            figures.plt.close(fig)
            return []
        with tempfile.TemporaryDirectory() as tmp,patch.object(figures,"_save",side_effect=inspect_and_close):
            figures.plot_spatial_modes(model,(2,2),tmp,{"mode_color_vmax":.2})
            scale=json.loads((Path(tmp)/"04_spatial_modes_color_scale.json").read_text())
            self.assertEqual(scale["colorbar_extend"],"both")
            self.assertEqual(scale["per_mode"][0]["fraction_clipped"],.5)
            self.assertTrue(scale["display_only"])
        np.testing.assert_array_equal(modes,original)
        self.assertEqual(len(checked),2)

    def test_percentile_mode_scale_uses_the_pool_of_all_displayed_modes(self):
        values=np.arange(-10.,10.).reshape(2,10)
        model=SimpleNamespace(modes=values,energy_fraction=np.array([.6,.3]))
        with tempfile.TemporaryDirectory() as tmp,patch.object(figures,"_save",side_effect=_close_without_export):
            figures.plot_spatial_modes(model,(2,5),tmp,{"mode_color_percentile":80})
            scale=json.loads((Path(tmp)/"04_spatial_modes_color_scale.json").read_text())
            self.assertAlmostEqual(scale["vmax"],np.percentile(np.abs(values),80))
            self.assertEqual(scale["pooled_pixel_count"],20)
            self.assertEqual(scale["displayed_mode_count"],2)
            self.assertEqual(scale["vmin"],-scale["vmax"])

    def test_invalid_mode_scales_fail_before_any_export_and_cli_accepts_valid_options(self):
        from schlieren_qml.cli import _validate_config
        X,model,scores,times,metadata=self.fixture()
        invalid=[{"mode_color_vmax":0},{"mode_color_vmax":float("nan")},
                 {"mode_color_vmax":True},{"mode_color_percentile":101},
                 {"mode_color_percentile":0},{"mode_color_vmax":.1,"mode_color_percentile":99}]
        for options in invalid:
            with self.subTest(options=options),tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    figures.plot_spatial_modes(model,(2,2),tmp,options)
                with self.assertRaises(ValueError):
                    figures.make_figures(X,(2,2),model,scores,times,np.arange(100),tmp,metadata,config=options)
                self.assertEqual(list(Path(tmp).iterdir()),[])
                with self.assertRaises(ValueError):
                    _validate_config({"image_directory":"images","output_directory":"results","figures":options})
        _validate_config({"image_directory":"images","output_directory":"results",
                          "figures":{"mode_color_vmax":.01,"mode_color_percentile":None}})


if __name__=="__main__":
    unittest.main()
