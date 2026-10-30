"""Blind repeat annotation and empirical TP localization repeatability."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .paper_data import load_study, check_output, rows, write_rows, write_json, sha, save_figure

def prepare_repeat_sample(results_directory, states_csv, output_directory, *, n_train=60, n_validation=30):
    data=load_study(results_directory,states_csv)
    for value in [n_train,n_validation]:
        if isinstance(value,bool) or not isinstance(value,int) or value<2:
            raise ValueError('Repeat samples need integer counts >=2 per partition.')
    output=check_output(output_directory,data)
    selected=[]
    for partition,count in [('train',n_train),('validation',n_validation)]:
        indices=data['split'][partition]
        indices=indices[np.linspace(0,len(indices)-1,min(count,len(indices)),dtype=int)]
        selected += [dict(filename=str(data['names'][i]),partition=partition,
                          sequence_frame_index=int(data['index'][i])) for i in indices]
    write_rows(output/'repeat_sample.csv',selected)
    write_json(output/'repeat_manifest.json',dict(config=data['config'],source_hashes=data['hashes'],
        results_directory=str(data['results']),states_csv=str(data['states_path']),
        n_frames=len(selected),method='Evenly spaced in original train/validation blocks; no test frames.',
        instructions='Two independent blind rounds. No automatic TP proposals are displayed.'))
    return dict(sample_csv=str(output/'repeat_sample.csv'),n_frames=len(selected),directory=str(output))

def launch_repeat_annotation(sample_directory, *, round_number=1, backend='TkAgg'):
    if round_number not in (1,2): raise ValueError('round_number must be 1 or 2.')
    sample=Path(sample_directory).resolve()
    subprocess.run([sys.executable,'-m','schlieren_qml.paper_measurement',str(sample),
                    '--round',str(round_number),'--backend',backend],
                   check=True,cwd=Path(__file__).resolve().parent.parent)
    return sample/f'repeat_round_{round_number}.csv'

def _repeat_window(sample_directory, round_number, backend):
    import matplotlib
    if backend not in ('TkAgg','QtAgg'): raise ValueError('Use TkAgg or QtAgg.')
    matplotlib.use(backend,force=True)
    import matplotlib.pyplot as plt
    from .data import read_image
    from .triple_point import _sequence, _annotations, _write_csv
    sample=Path(sample_directory).resolve()
    manifest=json.loads((sample/'repeat_manifest.json').read_text())
    config=manifest['config']
    paths,_=_sequence(config);lookup={p.name:p for p in paths}
    selected=rows(sample/'repeat_sample.csv')
    order=np.random.default_rng(271+round_number).permutation(len(selected))
    destination=sample/f'repeat_round_{round_number}.csv'
    points=_annotations(destination,{r['filename'] for r in selected})
    pending=[selected[i]['filename'] for i in order if selected[i]['filename'] not in points]
    if not pending: print('This independent round is complete.');return
    fig,ax=plt.subplots(figsize=(11,8));state=dict(i=0,point=None)
    def draw():
        name=pending[state['i']];ax.clear()
        ax.imshow(read_image(lookup[name]),cmap='gray',origin='upper')
        state['point']=points.get(name)
        if state['point'] is not None:ax.plot(*state['point'],'+',color='#D55E00',ms=14)
        ax.set(title=f"Blind round {round_number}: {state['i']+1}/{len(pending)} | {name}\n"
                     'Click TP + Enter. S: defer. Backspace: previous. Close: save progress.',
               xlabel='Original sensor x (px)',ylabel='Original sensor y (px)')
        fig.canvas.draw_idle()
    def click(event):
        if (event.inaxes!=ax or event.button!=1 or event.xdata is None
                or getattr(fig.canvas.manager.toolbar,'mode','')):return
        h,w=ax.images[0].get_array().shape
        if not 0<=event.xdata<=w-1 or not 0<=event.ydata<=h-1:return
        state['point']=np.array([event.xdata,event.ydata])
        for line in list(ax.lines):line.remove()
        ax.plot(*state['point'],'+',color='#D55E00',ms=14);fig.canvas.draw_idle()
    def key(event):
        if state['i']>=len(pending):return
        if event.key=='backspace' and state['i']>0:state['i']-=1;draw();return
        if event.key=='enter' and state['point'] is not None:
            points[pending[state['i']]]=state['point'].copy()
            _write_csv(destination,['filename','x_px','y_px'],[
                dict(filename=r['filename'],x_px=points[r['filename']][0],y_px=points[r['filename']][1])
                for r in selected if r['filename'] in points])
        elif event.key!='s':return
        state['i']+=1
        if state['i']==len(pending):plt.close(fig)
        else:draw()
    keymaps={k:[v for v in plt.rcParams[k] if v not in ['s','enter','backspace']]
             for k in plt.rcParams if k.startswith('keymap.')}
    fig.canvas.mpl_connect('button_press_event',click);fig.canvas.mpl_connect('key_press_event',key)
    with plt.rc_context(keymaps):draw();plt.show(block=True)

def measurement_report(sample_directory, output_directory, *, dpi=300):
    sample=Path(sample_directory).resolve()
    manifest=json.loads((sample/'repeat_manifest.json').read_text())
    data=load_study(manifest['results_directory'],manifest['states_csv'])
    if data['hashes']!=manifest['source_hashes']:raise ValueError('Measurement sources changed after sample selection.')
    output=check_output(output_directory,data)
    selected=rows(sample/'repeat_sample.csv'); names={r['filename'] for r in selected}
    rounds=[]
    frames_path=data['results']/'frames.json'
    shape=json.loads(frames_path.read_text())['original_shape'] if frames_path.is_file() else None
    for number in [1,2]:
        path=sample/f'repeat_round_{number}.csv'
        values=rows(path) if path.is_file() else []
        points={r['filename']:np.array([float(r['x_px']),float(r['y_px'])]) for r in values}
        if len(points)!=len(values) or not set(points).issubset(names) or any(not np.isfinite(p).all() for p in points.values()):
            raise ValueError('Invalid or duplicate repeat annotation.')
        if any(np.any(p<0) or (shape is not None and (p[0]>shape[1]-1 or p[1]>shape[0]-1)) for p in points.values()):
            raise ValueError('Repeat annotation is outside the sensor.')
        rounds.append(points)
    missing=[len(names-set(p)) for p in rounds]
    if any(missing):
        report=dict(status='pending',missing_round_1=missing[0],missing_round_2=missing[1],
                    explanation='Complete both blind rounds; no localization uncertainty is assumed.')
        write_json(output/'measurement_report.json',report);return report
    lookup={name:i for i,name in enumerate(data['names'])}; records=[]
    for r in selected:
        name=r['filename'];a,b=rounds[0][name],rounds[1][name];i=lookup[name]
        reference=(a+b)/2
        records.append(dict(filename=name,partition=r['partition'],
            round1_x_px=float(a[0]),round1_y_px=float(a[1]),round2_x_px=float(b[0]),round2_y_px=float(b[1]),
            reference_x_px=float(reference[0]),tracked_x_px=float(data['xy'][i,0]),
            repeat_difference_x_px=float(b[0]-a[0]),tracking_minus_reference_x_px=float(data['xy'][i,0]-reference[0]),
            reference_y_px=float(reference[1]),tracked_y_px=float(data['y'][i]),
            repeat_difference_y_px=float(b[1]-a[1]),tracking_minus_reference_y_px=float(data['y'][i]-reference[1])))
    report=dict(status='complete',source_hashes=data['hashes'],
        annotation_hashes={str(i+1):sha(sample/f'repeat_round_{i+1}.csv') for i in range(2)},partitions={})
    from .paper_data import category
    for partition in ['train','validation']:
        subset=[r for r in records if r['partition']==partition]
        delta=np.array([r['repeat_difference_y_px'] for r in subset])
        errors=np.array([r['tracking_minus_reference_y_px'] for r in subset])
        single_sigma=float(np.std(delta,ddof=1)/np.sqrt(2))
        disagreement=np.mean(category([r['round1_y_px'] for r in subset],data['definition'])!=
                             category([r['round2_y_px'] for r in subset],data['definition']))
        report['partitions'][partition]=dict(n=len(subset),round2_minus_round1_bias_px=float(delta.mean()),
            single_annotation_repeatability_sd_px=single_sigma,
            rms_difference_over_sqrt2_px=float(np.sqrt(np.mean(delta**2)/2)),
            tracking_bias_vs_repeat_mean_px=float(errors.mean()),
            tracking_mae_vs_repeat_mean_px=float(np.mean(np.abs(errors))),
            tracking_rmse_vs_repeat_mean_px=float(np.sqrt(np.mean(errors**2))),
            repeat_category_disagreement_fraction=float(disagreement))
        dx=np.array([r['repeat_difference_x_px'] for r in subset])
        ex=np.array([r['tracking_minus_reference_x_px'] for r in subset])
        report['partitions'][partition]['horizontal']=dict(round2_minus_round1_bias_px=float(dx.mean()),
            single_annotation_repeatability_sd_px=float(np.std(dx,ddof=1)/np.sqrt(2)),
            tracking_mae_vs_repeat_mean_px=float(np.mean(np.abs(ex))),
            tracking_rmse_vs_repeat_mean_px=float(np.sqrt(np.mean(ex**2))))
    report['boundary_sensitivity_margin_px']=2*max(
        report['partitions']['train']['single_annotation_repeatability_sd_px'],
        report['partitions']['train']['rms_difference_over_sqrt2_px'])
    report['margin_method']='2 × max(training repeatability SD, training RMS repeat difference / sqrt(2)); sensitivity width, not a calibrated confidence bound.'
    report['limitations']='Independent equal-variance annotation-error assumption; repeatability excludes common bias, calibration error and reference error. Repeats/frames may be correlated.'
    write_rows(output/'repeat_measurements.csv',records)
    write_json(output/'measurement_report.json',report)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for partition,color in [('train','#0072B2'),('validation','#D55E00')]:
        subset=[r for r in records if r['partition']==partition]
        axes[0].scatter([r['reference_y_px'] for r in subset],[r['tracked_y_px'] for r in subset],s=16,color=color,label=partition)
        axes[1].hist([r['repeat_difference_y_px'] for r in subset],bins=12,alpha=.5,color=color,label=partition)
    values=[r['reference_y_px'] for r in records]+[r['tracked_y_px'] for r in records]
    axes[0].plot([min(values),max(values)],[min(values),max(values)],'k--',lw=.8)
    axes[0].set(xlabel='Mean of blind repeat clicks, y (px)',ylabel='Tracked TP y (px)')
    axes[1].set(xlabel='Round 2 minus round 1 TP y (px)',ylabel='Sampled frames')
    for ax in axes:ax.legend(frameon=False)
    save_figure(fig,output,'01_tp_measurement_repeatability',dpi)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('sample_directory')
    parser.add_argument('--round',type=int,default=1);parser.add_argument('--backend',default='TkAgg')
    args=parser.parse_args();_repeat_window(args.sample_directory,args.round,args.backend)
