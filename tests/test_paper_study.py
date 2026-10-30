"""Conference-study leakage, physical labels, reviews, and matched evaluation."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from schlieren_qml.pod import fit_pod
from schlieren_qml.paper_data import load_study, category, write_rows, write_json
from schlieren_qml.paper_study import run_validation_study, evaluate_frozen_test, _boundary_rows, _paired_block_difference
from schlieren_qml.paper_measurement import prepare_repeat_sample, measurement_report, _repeat_window


class PaperStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.images=self.root/'images';self.images.mkdir()
        self.results=self.root/'results';self.results.mkdir()
        self.categories=self.root/'categories';self.categories.mkdir()
        self.states=self.categories/'tp_states.csv'
        rng=np.random.default_rng(9);texture=rng.integers(1000,5000,(7,7),dtype=np.uint16)
        matrix=[];points=[]
        for i in range(80):
            x=20+int(2*np.sin(i*.2));y=16+int(5*np.sin(i*.4))
            image=np.full((32,40),700,dtype=np.uint16)
            image[y-3:y+4,x-3:x+4]=texture
            Image.fromarray(image).save(self.images/f'f{i:03d}.tif')
            matrix.append(image.ravel());points.append([x,y])
        points=np.array(points);train=np.arange(48)
        mean=float(points[train,1].mean());sd=float(points[train,1].std())
        definition=dict(fit_partition='train',tracks_sha256='software_fixture',training_frames=48,
            mean_y_px=mean,standard_deviation_y_px=sd,upper_boundary_y_px=mean-sd,lower_boundary_y_px=mean+sd)
        for p in [self.categories/'tp_category_definition.json',self.results/'tp_category_definition.json']:write_json(p,definition)
        config=dict(image_directory=str(self.images),start=0,stride=1,limit=80,roi=None,bin_factor=1,
                    fps=51032.,train_fraction=.6,validation_fraction=.2,gap_frames=2)
        write_json(self.results/'config_used.json',config)
        model=fit_pod(np.asarray(matrix),train,rank=6,seed=42);model.save(self.results/'pod_model.npz')
        features=model.transform(np.asarray(matrix));states=[];coefficients=[]
        for i,(x,y) in enumerate(points):
            part='train' if i<48 else 'validation' if 50<=i<64 else 'test' if i>=66 else 'excluded'
            common=dict(filename=f'f{i:03d}.tif',sequence_frame_index=i,time_s=i/51032.)
            states.append(dict(**common,x_px=int(x),y_px=int(y),label=str(category([y],definition)[0]),accepted=True,source='template'))
            coefficients.append(dict(**common,partition=part,**{f'a_{j+1}':float(v) for j,v in enumerate(features[i])}))
        write_rows(self.states,states);write_rows(self.results/'modal_coefficients.csv',coefficients)
        self.grid=dict(C_values=[1.],gamma_values=['scale'],angle_scales=[.25],reps=[1],entanglement='linear')

    def study(self,path='study',crops=None):
        return run_validation_study(self.results,self.states,self.root/path,dimension=2,
            crops=crops,crop_bin_factor=1,crop_rank=4,train_sizes=[12,24],seeds=[11,23],
            max_validation=8,final_train_size=24,classifier=self.grid,dpi=35)

    def test_test_values_are_not_parsed_during_validation(self):
        before=load_study(self.results,self.states)
        from schlieren_qml.paper_data import rows
        path=self.results/'modal_coefficients.csv';values=rows(path)
        test={r['filename'] for r in values if r['partition']=='test'}
        for r in values:
            if r['filename'] in test:
                for k in list(r):
                    if k.startswith('a_'):r[k]='DO_NOT_PARSE'
        write_rows(path,values);values=rows(self.states)
        for r in values:
            if r['filename'] in test:r['y_px']='DO_NOT_PARSE'
        write_rows(self.states,values)
        after=load_study(self.results,self.states)
        np.testing.assert_array_equal(before['X'],after['X'])
        np.testing.assert_array_equal(before['y'],after['y'])
        with self.assertRaises(ValueError):load_study(self.results,self.states,include_test=True)

    def test_complete_study_matches_samples_and_freezes_before_test(self):
        from schlieren_qml.paper_data import rows
        with patch('builtins.print'):
            result=self.study(crops={'context':[8,4,34,28]})
        out=Path(result['output_directory'])
        curves=rows(out/'learning_curves.csv')
        self.assertEqual(len(curves),32)
        self.assertEqual({r['dimension'] for r in curves},{'2'})
        for i in range(0,len(curves),2):
            self.assertEqual(curves[i]['n_train'],curves[i+1]['n_train'])
            self.assertEqual(curves[i]['modes'],curves[i+1]['modes'])
        self.assertFalse(list(out.glob('crop_*/images_train_validation.npy')))
        self.assertFalse((out/'final_evaluation.json').exists())
        with patch('schlieren_qml.paper_study.fit_validate_classifiers',side_effect=AssertionError('Test tuning!')):
            final=evaluate_frozen_test(out,self.root/'final',max_test=8,n_bootstrap=20,dpi=35)
        self.assertEqual(final['n_test'],8)
        self.assertEqual(final['test_usage'],'previously_inspected')
        records=rows(self.root/'final/test_predictions.csv')
        self.assertTrue(all(r['partition']=='test' for r in records))
        self.assertEqual(sum(int(r['n']) for r in rows(self.root/'final/test_boundary_bins.csv')),8)
        self.assertEqual(final['paired_temporal_block_difference']['status'],'insufficient_blocks')
        with self.assertRaises(FileExistsError):evaluate_frozen_test(out,self.root/'final',max_test=8)
        (out/'selection.json').write_text((out/'selection.json').read_text()+' ')
        with self.assertRaisesRegex(ValueError,'selection changed'):evaluate_frozen_test(out,self.root/'changed')

    def test_boundary_sensitivity_reports_full_primary_coverage(self):
        data=load_study(self.results,self.states);indices=data['split']['validation']
        predicted=data['labels'][indices].copy();predicted[0]='candidate_1'
        records,bins,secondary=_boundary_rows(data,indices,dict(rbf_svm=predicted,qsvm=predicted),'validation',100.)
        self.assertEqual(len(records),14);self.assertEqual(sum(r['n'] for r in bins),14)
        self.assertEqual(secondary['n_total'],14);self.assertEqual(secondary['n_retained'],0)
        self.assertEqual(secondary['metrics'],{})

    def test_repeat_uncertainty_is_measured_not_assumed(self):
        from schlieren_qml.paper_data import rows
        sample=self.root/'repeats'
        prepare_repeat_sample(self.results,self.states,sample,n_train=6,n_validation=4)
        pending=measurement_report(sample,self.root/'pending',dpi=35)
        self.assertEqual(pending['status'],'pending')
        data=load_study(self.results,self.states);lookup={name:i for i,name in enumerate(data['names'])}
        selected=rows(sample/'repeat_sample.csv')
        self.assertFalse(any(r['partition']=='test' for r in selected))
        for round_number in [1,2]:
            values=[]
            for j,r in enumerate(selected):
                xy=data['xy'][lookup[r['filename']]].copy()
                if round_number==2:xy[1]+=(-1)**j*.5
                values.append(dict(filename=r['filename'],x_px=float(xy[0]),y_px=float(xy[1])))
            write_rows(sample/f'repeat_round_{round_number}.csv',values)
        report=measurement_report(sample,self.root/'measured',dpi=35)
        self.assertEqual(report['status'],'complete')
        expected=np.std([.5,-.5]*3,ddof=1)/np.sqrt(2)
        self.assertAlmostEqual(report['partitions']['train']['single_annotation_repeatability_sd_px'],expected)
        self.assertEqual(report['partitions']['validation']['n'],4)

    def test_repeat_window_is_blind_and_saves_clicks(self):
        import matplotlib.pyplot as plt
        from matplotlib.backend_bases import MouseEvent,KeyEvent
        sample=self.root/'repeats';prepare_repeat_sample(self.results,self.states,sample,n_train=2,n_validation=2)
        def interact(**kwargs):
            fig=plt.gcf();self.assertEqual(len(fig.axes[0].lines),0)
            fig.canvas.draw();x,y=fig.axes[0].transData.transform([20.,16.])
            fig.canvas.callbacks.process('button_press_event',MouseEvent('button_press_event',fig.canvas,x,y,button=1))
            fig.canvas.callbacks.process('key_press_event',KeyEvent('key_press_event',fig.canvas,key='enter'))
        with patch('matplotlib.use'),patch('matplotlib.pyplot.show',side_effect=interact):
            _repeat_window(sample,1,'TkAgg')
        self.assertTrue((sample/'repeat_round_1.csv').is_file());plt.close('all')

    def test_output_and_dimension_validation(self):
        with self.assertRaises(ValueError):
            run_validation_study(self.results,self.states,self.images/'study',dimension=9)
        with self.assertRaises(ValueError):
            prepare_repeat_sample(self.results,self.states,self.images/'study')

    def test_paired_block_difference_retains_pairing(self):
        labels=np.tile(np.array(['candidate_1','candidate_2','candidate_3']),30)
        result=_paired_block_difference(labels,labels,labels,np.arange(90),block_frames=10,n_bootstrap=30)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['percentile_95_interval'],[0.,0.])

    def test_frozen_crop_projects_only_sampled_test_images(self):
        from schlieren_qml.paper_data import rows
        # Deliberately remove information from saved features so a new crop wins.
        path=self.results/'modal_coefficients.csv';values=rows(path)
        for r in values:
            for k in list(r):
                if k.startswith('a_'):r[k]=0.
        write_rows(path,values)
        with patch('builtins.print'):
            result=self.study(crops={'context':[8,4,34,28]})
        out=Path(result['output_directory']);selection=json.loads((out/'selection.json').read_text())
        self.assertEqual(selection['source'],'crop_01')
        from schlieren_qml.data import read_image
        with patch('schlieren_qml.data.read_image',wraps=read_image) as read:
            final=evaluate_frozen_test(out,self.root/'final_crop',max_test=8,n_bootstrap=20,dpi=35)
        self.assertEqual(read.call_count,8)
        self.assertTrue(all(int(Path(call.args[0]).stem[1:])>=66 for call in read.call_args_list))
        self.assertEqual(final['n_test'],8)


if __name__=='__main__':unittest.main()
