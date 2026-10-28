"""TP measurement geometry, quality gating, and train-only category boundaries."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from schlieren_qml.triple_point import track_tp, tp_categories, tp_gallery, _write_csv


class TriplePointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.images=self.root/'images';self.images.mkdir()
        rng=np.random.default_rng(25)
        self.patch=rng.integers(1000,5000,size=(11,11),dtype=np.uint16)
        self.points=[]
        for i in range(60):
            x=30+int(3*np.sin(i*.19));y=30+int(8*np.sin(i*.3))
            image=np.full((64,72),700,dtype=np.uint16)
            image[y-5:y+6,x-5:x+6]=self.patch
            Image.fromarray(image).save(self.images/f'frame{i:03d}.tif')
            self.points.append([x,y])
        self.config=dict(image_directory=str(self.images),start=0,stride=1,limit=60,
                         fps=51032.,train_fraction=.6,validation_fraction=.2,gap_frames=1,
                         output_directory=str(self.root/'original_run'),rank=4,
                         classifier={'n_features':4,'n_clusters':3})
        self.annotations=self.root/'annotations.csv'
        _write_csv(self.annotations,['filename','x_px','y_px'],
                   [dict(filename=f'frame{i:03d}.tif',x_px=x,y_px=y) for i,(x,y) in enumerate(self.points[:3])])

    def track(self,**kwargs):
        return track_tp(self.config,self.annotations,self.root/'tracking',manual_count=3,
                        patch_size=11,search_radius=12,min_score=.9,min_margin=.05,**kwargs)

    def test_tracks_original_pixels_and_keeps_tiffs_unchanged(self):
        before=(self.images/'frame020.tif').read_bytes()
        result=self.track()
        self.assertEqual(result['n_uncertain'],0)
        with Path(result['tracks_csv']).open() as f: rows=list(csv.DictReader(f))
        np.testing.assert_allclose([[float(r['x_px']),float(r['y_px'])] for r in rows],self.points,atol=1e-6)
        self.assertEqual((self.images/'frame020.tif').read_bytes(),before)
        self.assertEqual(result['template_filenames'],['frame000.tif','frame001.tif','frame002.tif'])

    def test_vertical_direction_and_training_mean_are_fixed(self):
        tracked=self.track();out=self.root/'labels'
        result=tp_categories(tracked['tracks_csv'],out)
        ys=np.array(self.points)[:36,1]
        self.assertAlmostEqual(result['mean_y_px'],ys.mean())
        self.assertAlmostEqual(result['standard_deviation_y_px'],ys.std(ddof=0))
        with (out/'tp_states.csv').open() as f:rows=list(csv.DictReader(f))
        for row in rows:
            y=float(row['y_px']);mean=result['mean_y_px'];sd=result['standard_deviation_y_px']
            expected='candidate_1' if y>mean+sd else 'candidate_3' if y<mean-sd else 'candidate_2'
            self.assertEqual(row['label'],expected)
        config=json.loads(Path(result['config_path']).read_text())
        self.assertEqual(config['labels_csv'],str(out/'tp_labels.csv'))
        self.assertEqual(config['image_directory'],str(self.images))
        # Changing held-out measurements cannot move the training thresholds.
        for row in rows[36:]:row['y_px']='61'
        fields=[k for k in rows[0] if k!='label']
        _write_csv(Path(tracked['tracks_csv']),fields,[{k:r[k] for k in fields} for r in rows])
        second=tp_categories(tracked['tracks_csv'],self.root/'second_labels')
        self.assertEqual(result['mean_y_px'],second['mean_y_px'])
        self.assertEqual(result['standard_deviation_y_px'],second['standard_deviation_y_px'])

    def test_uncertain_track_blocks_labels_and_can_be_manually_corrected(self):
        Image.fromarray(np.full((64,72),700,dtype=np.uint16)).save(self.images/'frame020.tif')
        result=self.track();self.assertGreater(result['n_uncertain'],0)
        with self.assertRaisesRegex(ValueError,'Uncertain'):
            tp_categories(result['tracks_csv'],self.root/'labels')
        self.assertFalse((self.root/'labels').exists())
        with self.annotations.open() as f: rows=list(csv.DictReader(f))
        rows.append(dict(filename='frame020.tif',x_px=30,y_px=30))
        _write_csv(self.annotations,['filename','x_px','y_px'],rows)
        corrected=self.track(overwrite=True)
        self.assertEqual(corrected['n_uncertain'],0)
        self.assertNotIn('frame020.tif',corrected['template_filenames'])

    def test_annotations_and_outputs_are_protected(self):
        with self.assertRaisesRegex(ValueError,'separate'):
            track_tp(self.config,self.annotations,self.images,manual_count=3)
        self.track()
        before=(self.root/'tracking/tp_tracks.csv').read_bytes()
        with self.assertRaises(FileExistsError):self.track()
        self.assertEqual((self.root/'tracking/tp_tracks.csv').read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'odd'):
            track_tp(self.config,self.annotations,self.root/'other',manual_count=3,patch_size=10)

    def test_manual_seed_completeness_and_bounds(self):
        with self.assertRaisesRegex(ValueError,'seed'):
            track_tp(self.config,self.annotations,self.root/'missing',manual_count=4)
        _write_csv(self.annotations,['filename','x_px','y_px'],
                   [dict(filename=f'frame{i:03d}.tif',x_px=1,y_px=1) for i in range(3)])
        with self.assertRaisesRegex(ValueError,'edge'):
            self.track()

    def test_overlay_returns_sensor_coordinate_marker(self):
        tracked=self.track()
        result=tp_gallery(self.config,tracked['tracks_csv'],examples=3)
        import matplotlib.pyplot as plt
        self.addCleanup(plt.close,result['figure'])
        axes=[ax for ax in result['figure'].axes if ax.images]
        self.assertEqual(len(axes),3)
        for ax,row in zip(axes,result['rows']):
            self.assertEqual(ax.images[0].get_array().shape,(64,72))
            self.assertEqual(float(ax.lines[0].get_ydata()[0]),float(row['y_px']))

    def test_annotation_window_controller_saves_clicks_and_resumes(self):
        from schlieren_qml.triple_point import _annotation_window
        import matplotlib.pyplot as plt
        from matplotlib.backend_bases import MouseEvent, KeyEvent
        self.annotations.unlink()

        def click_through(**kwargs):
            fig=plt.gcf()
            for point in self.points[:3]:
                fig.canvas.draw()
                x,y=fig.axes[0].transData.transform(point)
                event=MouseEvent('button_press_event',fig.canvas,x,y,button=1)
                fig.canvas.callbacks.process('button_press_event',event)
                fig.canvas.callbacks.process('key_press_event',KeyEvent('key_press_event',fig.canvas,key='enter'))

        # Exercise actual click/key callbacks with Agg; the Windows GUI itself
        # still needs checking on the user's computer.
        with patch('matplotlib.use'),patch('matplotlib.pyplot.show',side_effect=click_through):
            _annotation_window(self.config,self.annotations,3,None,'TkAgg')
        with self.annotations.open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),3)
        np.testing.assert_allclose([[float(r['x_px']),float(r['y_px'])] for r in rows],self.points[:3],atol=1e-10)
        with patch('matplotlib.use'),patch('matplotlib.pyplot.show') as show,patch('builtins.print'):
            _annotation_window(self.config,self.annotations,3,None,'TkAgg')
        show.assert_not_called()

    def test_exported_labels_bypass_kmeans_in_the_pod_classifier_pipeline(self):
        from schlieren_qml.cli import run
        result=self.track()
        categories=tp_categories(result['tracks_csv'],self.root/'categories')
        with patch('schlieren_qml.classify.KMeans.fit',side_effect=AssertionError('KMeans was used')), \
             patch('schlieren_qml.figures.make_figures',return_value=[]), \
             patch('schlieren_qml.figures.plot_roi',return_value=[]),patch('builtins.print'):
            manifest=run(categories['config_path'])
        self.assertEqual(manifest['classification_status'],'complete')
        config=json.loads(Path(categories['config_path']).read_text())
        metrics=json.loads((Path(config['output_directory'])/'classification_metrics.json').read_text())
        self.assertEqual(metrics['label_mode'],'external')
        self.assertIsNone(metrics['state_discovery'])
        self.assertEqual(metrics['classes'],['candidate_1','candidate_2','candidate_3'])


if __name__=='__main__':unittest.main()
