"""Human TP review: keep a valid flagged match, correct, defer, and resume."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from schlieren_qml.triple_point import (
    _REVIEW_FIELDS, _annotation_window, _write_csv, apply_tp_review,
    track_tp, tp_categories,
)


class TPReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.images = self.root/'images'; self.images.mkdir()
        texture = np.random.default_rng(4).integers(1000,5000,(7,7),dtype=np.uint16)
        self.points = []
        for i in range(40):
            x,y = 24,20+int(4*np.sin(i*.4))
            image = np.full((40,48),700,dtype=np.uint16)
            image[y-3:y+4,x-3:x+4] = texture
            Image.fromarray(image).save(self.images/f'f{i:03d}.tif')
            self.points.append([x,y])
        self.config = dict(image_directory=str(self.images),limit=40,fps=51032.,
                           train_fraction=.6,validation_fraction=.2,gap_frames=1)
        self.annotations = self.root/'tp_manual.csv'
        self.reviews = self.root/'tp_review.csv'
        _write_csv(self.annotations,['filename','x_px','y_px'],[
            dict(filename=f'f{i:03d}.tif',x_px=x,y_px=y) for i,(x,y) in enumerate(self.points[:2])])
        self.tracks = self.root/'tracking/tp_tracks.csv'

    def track(self,**kwargs):
        return track_tp(self.config,self.annotations,self.tracks.parent,manual_count=2,
                        patch_size=7,search_radius=6,min_score=.9,min_margin=.05,
                        review_path=kwargs.pop('review_path',self.reviews),**kwargs)

    def rows(self,path=None):
        with (path or self.tracks).open(newline='') as handle:
            return list(csv.DictReader(handle))

    def mark_flagged(self,*indices):
        rows = self.rows()
        for i in indices:
            rows[i]['accepted']='False'; rows[i]['automatic_accepted']='False'
        _write_csv(self.tracks,list(rows[0]),rows)
        return rows

    def decision(self,row,kind='accept_existing',point=None):
        point = point or [float(row['x_px']),float(row['y_px'])]
        return dict(filename=row['filename'],x_px=point[0],y_px=point[1],decision=kind,
                    proposal_x_px=row['x_px'],proposal_y_px=row['y_px'],
                    tracks_sha256=hashlib.sha256(self.tracks.read_bytes()).hexdigest())

    def test_accept_existing_keeps_automatic_flag_and_does_not_create_manual_anchor(self):
        self.track(); rows=self.mark_flagged(5)
        before=self.annotations.read_bytes()
        _write_csv(self.reviews,_REVIEW_FIELDS,[self.decision(rows[5])])
        result=apply_tp_review(self.tracks,self.annotations,self.reviews)
        reviewed=self.rows()[5]
        self.assertEqual(result['n_uncertain'],0)
        self.assertEqual(reviewed['accepted'],'True')
        self.assertEqual(reviewed['automatic_accepted'],'False')
        self.assertEqual(reviewed['review_status'],'accepted_existing')
        self.assertEqual(reviewed['source'],'template')
        self.assertEqual(self.annotations.read_bytes(),before)
        self.assertEqual(reviewed['x_px'],rows[5]['x_px'])
        self.assertEqual(reviewed['y_px'],rows[5]['y_px'])
        categories=tp_categories(self.tracks,self.root/'categories')
        self.assertEqual(categories['tracking_review']['n_human_accepted_existing'],1)
        self.assertEqual(categories['tracking_review']['n_original_automatic_flags'],1)

    def test_acceptance_does_not_apply_at_changed_coordinate(self):
        self.track(); rows=self.mark_flagged(5)
        decision=self.decision(rows[5]); decision['x_px']=float(decision['x_px'])+1
        _write_csv(self.reviews,_REVIEW_FIELDS,[decision])
        result=apply_tp_review(self.tracks,self.annotations,self.reviews)
        self.assertEqual(result['n_uncertain'],1)
        self.assertEqual(self.rows()[5]['accepted'],'False')

    def test_correction_requires_saved_manual_point_and_remains_traceable(self):
        self.track(); rows=self.mark_flagged(5)
        point=[25.,21.]
        _write_csv(self.reviews,_REVIEW_FIELDS,[self.decision(rows[5],'corrected',point)])
        first=apply_tp_review(self.tracks,self.annotations,self.reviews)
        self.assertEqual(first['n_uncertain'],1)
        annotations=self.rows(self.annotations)
        annotations.append(dict(filename=rows[5]['filename'],x_px=point[0],y_px=point[1]))
        _write_csv(self.annotations,['filename','x_px','y_px'],annotations)
        second=apply_tp_review(self.tracks,self.annotations,self.reviews)
        corrected=self.rows()[5]
        self.assertEqual(second['n_uncertain'],0)
        self.assertEqual(corrected['review_status'],'corrected')
        self.assertEqual(corrected['source'],'manual')
        self.assertEqual(corrected['automatic_accepted'],'False')
        self.assertEqual(float(corrected['x_px']),point[0])
        self.assertEqual(float(corrected['y_px']),point[1])
        # Idempotent application remains valid after the first update.
        self.assertEqual(apply_tp_review(self.tracks,self.annotations,self.reviews)['n_uncertain'],0)

    def test_retracking_accepts_only_same_proposal_and_not_manual_appearance(self):
        # Blank frame forces an uncertain deterministic proposal.
        Image.fromarray(np.full((40,48),700,dtype=np.uint16)).save(self.images/'f005.tif')
        self.track(); rows=self.rows(); self.assertEqual(rows[5]['accepted'],'False')
        _write_csv(self.reviews,_REVIEW_FIELDS,[self.decision(rows[5])])
        result=self.track(overwrite=True)
        self.assertEqual(self.rows()[5]['review_status'],'accepted_existing')
        self.assertEqual(self.rows()[5]['automatic_accepted'],'False')
        self.assertNotIn('f005.tif',result['template_filenames'])
        decision=self.decision(rows[5]); decision['y_px']=float(decision['y_px'])+1
        _write_csv(self.reviews,_REVIEW_FIELDS,[decision])
        self.track(overwrite=True)
        self.assertEqual(self.rows()[5]['accepted'],'False')

    def test_review_gui_accepts_then_defers_without_annotation_changes(self):
        import matplotlib.pyplot as plt
        from matplotlib.backend_bases import KeyEvent
        self.track(); self.mark_flagged(5,6)
        before=self.annotations.read_bytes()
        def interact(**kwargs):
            fig=plt.gcf()
            for key in ['a','s']:
                fig.canvas.callbacks.process('key_press_event',KeyEvent('key_press_event',fig.canvas,key=key))
        with patch('matplotlib.use'),patch('matplotlib.pyplot.show',side_effect=interact):
            _annotation_window(self.config,self.annotations,2,self.tracks,'TkAgg',self.reviews)
        self.assertEqual(self.annotations.read_bytes(),before)
        self.assertEqual(len(self.rows(self.reviews)),1)
        result=apply_tp_review(self.tracks,self.annotations,self.reviews)
        self.assertEqual(result['n_uncertain'],1)
        self.assertEqual(self.rows()[6]['accepted'],'False')
        plt.close('all')

    def test_review_all_can_correct_existing_manual_seed(self):
        import matplotlib.pyplot as plt
        from matplotlib.backend_bases import MouseEvent,KeyEvent
        self.track()
        point=[24.,21.]
        def interact(**kwargs):
            fig=plt.gcf(); fig.canvas.draw()
            x,y=fig.axes[0].transData.transform(point)
            fig.canvas.callbacks.process('button_press_event',MouseEvent('button_press_event',fig.canvas,x,y,button=1))
            fig.canvas.callbacks.process('key_press_event',KeyEvent('key_press_event',fig.canvas,key='enter'))
        with patch('matplotlib.use'),patch('matplotlib.pyplot.show',side_effect=interact):
            _annotation_window(self.config,self.annotations,2,self.tracks,'TkAgg',self.reviews,True)
        result=apply_tp_review(self.tracks,self.annotations,self.reviews)
        row=self.rows()[0]
        self.assertEqual(row['review_status'],'corrected')
        self.assertAlmostEqual(float(row['y_px']),21.)
        self.assertEqual(result['n_uncertain'],0)
        plt.close('all')

    def test_review_shortcuts_reserve_s_and_a_and_restore_matplotlib_keymaps(self):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backend_bases import KeyEvent
        self.track(); self.mark_flagged(5)
        original = {name:list(matplotlib.rcParams[name]) for name in matplotlib.rcParams
                    if name.startswith('keymap.')}
        observed = []
        def interact(**kwargs):
            observed.append(True)
            self.assertNotIn('s',matplotlib.rcParams['keymap.save'])
            self.assertIn('ctrl+s',matplotlib.rcParams['keymap.save'])
            self.assertFalse(any(key in {'a','s','enter','backspace'}
                for name in matplotlib.rcParams if name.startswith('keymap.')
                for key in matplotlib.rcParams[name]))
            fig=plt.gcf()
            fig.canvas.callbacks.process('key_press_event',KeyEvent('key_press_event',fig.canvas,key='s'))
        with matplotlib.rc_context({'keymap.save':['s','ctrl+s'],'keymap.home':['a','h']}):
            with patch('matplotlib.use'),patch('matplotlib.pyplot.show',side_effect=interact):
                _annotation_window(self.config,self.annotations,2,self.tracks,'TkAgg',self.reviews)
            self.assertEqual(matplotlib.rcParams['keymap.save'],['s','ctrl+s'])
            self.assertEqual(matplotlib.rcParams['keymap.home'],['a','h'])
        self.assertEqual(len(observed),1)
        for name,value in original.items():
            self.assertEqual(matplotlib.rcParams[name],value)
        # Defer did not create a human acceptance or manual anchor.
        self.assertFalse(self.reviews.exists())
        self.assertEqual(apply_tp_review(self.tracks,self.annotations,self.reviews)['n_uncertain'],1)
        plt.close('all')

    def test_review_all_defer_preserves_an_already_accepted_manual_point(self):
        import matplotlib.pyplot as plt
        from matplotlib.backend_bases import KeyEvent
        self.track(); before=self.tracks.read_bytes()
        def interact(**kwargs):
            fig=plt.gcf()
            fig.canvas.callbacks.process('key_press_event',KeyEvent('key_press_event',fig.canvas,key='s'))
        with patch('matplotlib.use'),patch('matplotlib.pyplot.show',side_effect=interact):
            _annotation_window(self.config,self.annotations,2,self.tracks,'TkAgg',self.reviews,True)
        self.assertEqual(self.tracks.read_bytes(),before)
        self.assertEqual(self.rows()[0]['accepted'],'True')
        self.assertFalse(self.reviews.exists())
        plt.close('all')

    def test_review_and_annotation_paths_cannot_alias(self):
        self.track()
        with self.assertRaisesRegex(ValueError,'different files'):
            self.track(overwrite=True,review_path=self.annotations)


if __name__=='__main__':
    unittest.main()
