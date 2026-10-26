"""Image-to-candidate provenance, faithful display, and export safeguards."""
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"schlieren_candidate_test_mpl"))
from schlieren_qml.candidates import candidate_gallery, candidate_image, export_candidate_lists


class CandidateViewerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root/"results"
        self.images = self.root/"TIFF images"
        self.results.mkdir()
        self.images.mkdir()
        self.columns = ["filename", "sequence_frame_index", "time_s", "partition", "state",
                        "rbf_test_prediction", "qsvm_test_prediction", "majority_test_prediction"]
        self.records, self.frames, self.arrays = [], [], {}
        for index in range(12):
            state = f"candidate_{index//4+1}"
            filename = f"Img{index:06}.tif"
            image = (np.arange(48).reshape(6, 8)+1000+index*50).astype(np.uint16)
            self.arrays[filename] = image
            path = self.images/filename
            Image.fromarray(image).save(path)
            partition = "test" if index%4 == 3 else "train"
            prediction = ("candidate_2" if index == 3 else state) if partition == "test" else ""
            self.records.append(dict(zip(self.columns, [filename,index,index/51032,partition,state,
                                                       prediction,prediction,"candidate_2" if partition=="test" else ""])))
            self.frames.append({"row":index,"frame_index":index,"filename":filename,
                                "path":str(path),"time_s":index/51032})
        self.config = {"image_directory":str(self.images),"roi":[2,0,8,6],"bin_factor":2}
        self.metadata = {"original_shape":[6,8],"image_shape":[3,3],"bin_factor":2,
                         "roi":[2,0,8,6],"frame_records":self.frames}
        self.save_metadata()
        self.write_assignments()

    def save_metadata(self):
        (self.results/"config_used.json").write_text(json.dumps(self.config))
        (self.results/"frames.json").write_text(json.dumps(self.metadata))

    def write_assignments(self):
        with (self.results/"state_assignments.csv").open("w",newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.records)

    def close(self, result):
        import matplotlib.pyplot as plt
        plt.close(result["figure"])

    def test_training_gallery_preserves_depth_shared_scale_and_saved_preprocessing(self):
        result = candidate_gallery(self.results, examples_per_candidate=2)
        self.addCleanup(self.close, result)
        self.assertEqual(result["counts"], {f"candidate_{i}":3 for i in (1,2,3)})
        self.assertEqual(len(result["selected_rows"]),6)
        self.assertTrue(all(row["partition"]=="train" for row in result["selected_rows"]))
        images = [ax.images[0] for ax in result["figure"].axes if ax.images]
        self.assertEqual(len(images),6)
        self.assertEqual(len({image.get_clim() for image in images}),1)
        self.assertGreater(result["display_limits"][0],255)
        for row, shown in zip(result["selected_rows"], images):
            original = self.arrays[row["filename"]].astype(np.float32)[:,2:8]
            expected = original.reshape(3,2,3,2).mean(axis=(1,3))
            np.testing.assert_array_equal(shown.get_array(),expected)
        repeated = candidate_gallery(self.results, examples_per_candidate=2)
        self.addCleanup(self.close,repeated)
        self.assertEqual(result["selected_rows"],repeated["selected_rows"])

    def test_prediction_gallery_uses_only_nonblank_test_predictions(self):
        with self.assertRaisesRegex(ValueError,"partition='test'"):
            candidate_gallery(self.results, label_column="qsvm_test_prediction")
        result = candidate_gallery(self.results, examples_per_candidate=2,
                                   label_column="qsvm_test_prediction",partition="test",processed=False)
        self.addCleanup(self.close,result)
        self.assertEqual(result["counts"],{"candidate_2":2,"candidate_3":1})
        self.assertEqual(len(result["selected_rows"]),3)
        for row in result["selected_rows"]:
            self.assertEqual(row["partition"],"test")
            self.assertTrue(row["qsvm_test_prediction"])
        for ax in result["figure"].axes:
            if ax.images:
                self.assertIn("assigned",ax.get_title())
                self.assertIn("predicted",ax.get_title())
                self.assertEqual(ax.images[0].get_array().shape,(6,8))

    def test_one_candidate_one_image_and_original_lookup(self):
        result = candidate_gallery(self.results,examples_per_candidate=1,only_candidate="candidate_1")
        self.addCleanup(self.close,result)
        self.assertEqual(len(result["selected_rows"]),1)
        single = candidate_image(self.results,"Img000003.tif")
        self.addCleanup(self.close,single)
        self.assertEqual(single["row"]["state"],"candidate_1")
        np.testing.assert_array_equal(single["figure"].axes[0].images[0].get_array(),self.arrays["Img000003.tif"])
        with self.assertRaisesRegex(ValueError,"exactly one"):
            candidate_image(self.results,"missing.tif")

    def test_source_alignment_and_saved_shape_mismatches_are_rejected(self):
        self.frames[0]["filename"] = "another.tif"
        self.save_metadata()
        with self.assertRaisesRegex(ValueError,"same completed run"):
            candidate_gallery(self.results)
        self.frames[0]["filename"] = self.records[0]["filename"]
        self.metadata["image_shape"] = [1,1]
        self.save_metadata()
        with self.assertRaisesRegex(ValueError,"recorded image shape"):
            candidate_gallery(self.results)
        self.metadata["image_shape"] = [3,3]
        self.config["roi"] = [0,0,6,6]
        self.save_metadata()
        with self.assertRaisesRegex(ValueError,"disagree about the crop"):
            candidate_gallery(self.results)

    def test_moved_image_directory_can_be_overridden(self):
        moved = self.root/"relocated images"
        shutil.move(self.images,moved)
        with self.assertRaisesRegex(FileNotFoundError,"image_directory_override"):
            candidate_image(self.results,"Img000000.tif")
        result = candidate_image(self.results,"Img000000.tif",image_directory_override=moved)
        self.addCleanup(self.close,result)
        np.testing.assert_array_equal(result["figure"].axes[0].images[0].get_array(),self.arrays["Img000000.tif"])

    def test_full_image_roi_normalizes_none_and_explicit_bounds(self):
        self.config["roi"] = None
        self.metadata["image_shape"] = [3,4]
        for recorded_roi in ([0,0,8,6], None):
            self.metadata["roi"] = recorded_roi
            self.save_metadata()
            result = candidate_gallery(self.results, examples_per_candidate=1)
            self.addCleanup(self.close,result)
            self.assertTrue(all(ax.images[0].get_array().shape == (3,4)
                                for ax in result["figure"].axes if ax.images))
        # Earlier frame manifests without an ROI key remain readable.
        del self.metadata["roi"]
        self.save_metadata()
        result = candidate_gallery(self.results, examples_per_candidate=1)
        self.addCleanup(self.close,result)

    def test_export_keeps_every_assignment_and_never_reads_tiffs(self):
        source_bytes = (self.results/"state_assignments.csv").read_bytes()
        with patch("schlieren_qml.candidates.read_image",side_effect=AssertionError("TIFF was read")):
            exported = export_candidate_lists(self.results,self.results/"candidate_lists")
        self.assertEqual(exported["counts"],{f"candidate_{i}":4 for i in (1,2,3)})
        self.assertFalse(exported["images_copied"])
        self.assertEqual(exported["total_frames"],12)
        for label,filename in exported["files"].items():
            with Path(filename).open(newline="",encoding="utf-8-sig") as handle:
                reader=csv.DictReader(handle)
                self.assertEqual(reader.fieldnames,self.columns)
                actual=list(reader)
            expected = [{key:str(value) for key,value in row.items()}
                        for row in self.records if row["state"]==label]
            self.assertEqual(actual,expected)
        self.assertEqual((self.results/"state_assignments.csv").read_bytes(),source_bytes)
        with self.assertRaises(FileExistsError):
            export_candidate_lists(self.results,self.results/"candidate_lists")

    def test_export_preflights_all_destinations_and_protects_source_aliases(self):
        output=self.root/"lists"
        output.mkdir()
        source=self.images/"Img000000.tif"
        original=source.read_bytes()
        os.link(source,output/"candidate_2.csv")
        with self.assertRaisesRegex(ValueError,"file alias"):
            export_candidate_lists(self.results,output,overwrite=True)
        self.assertFalse((output/"candidate_1.csv").exists())
        self.assertEqual(source.read_bytes(),original)
        with self.assertRaisesRegex(ValueError,"source image folder"):
            export_candidate_lists(self.results,self.images/"lists")
        with self.assertRaisesRegex(ValueError,"separate"):
            export_candidate_lists(self.results,self.results)

    def test_prediction_partition_corruption_and_ambiguous_override_are_rejected(self):
        self.records[0]["qsvm_test_prediction"] = "candidate_1"
        self.write_assignments()
        with self.assertRaisesRegex(ValueError,"same completed run"):
            candidate_gallery(self.results)
        self.records[0]["qsvm_test_prediction"] = ""
        self.records[1]["filename"] = self.records[0]["filename"]
        self.frames[1]["filename"] = self.frames[0]["filename"]
        self.frames[1]["path"] = self.frames[0]["path"]
        self.save_metadata()
        self.write_assignments()
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            candidate_gallery(self.results, image_directory_override=self.images)

    def test_duplicate_identity_and_invalid_controls_fail_before_image_read(self):
        for options in [{"examples_per_candidate":True},{"seed":-1},{"partition":"unknown"},
                        {"label_column":"wrong"},{"processed":"yes"}]:
            with self.subTest(options=options),patch("schlieren_qml.candidates.read_image") as read:
                with self.assertRaises(ValueError):
                    candidate_gallery(self.results,**options)
                read.assert_not_called()
        self.records[1]["sequence_frame_index"] = self.records[0]["sequence_frame_index"]
        self.write_assignments()
        with self.assertRaisesRegex(ValueError,"same completed run"):
            candidate_gallery(self.results)
