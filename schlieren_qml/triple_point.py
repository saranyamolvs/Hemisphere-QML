"""Manual TP annotations, template tracking, and training-defined vertical states.

Coordinates always refer to the ORIGINAL sensor image, before ROI or binning.
Template correlation is an exploratory measurement aid, not a validated TP
detector. Scores are similarity measures, not probabilities or uncertainties.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.signal import correlate

from .data import list_images, read_image, _same_file
from .classify import make_splits


def _sequence(config):
    paths = list_images(config['image_directory'], config.get('pattern', '*'))
    start, stride = config.get('start', 0), config.get('stride', 1)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError('start must be a nonnegative integer.')
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError('stride must be a positive integer.')
    selected = paths[start::stride]
    limit = config.get('limit', 20000)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('limit must be positive or None.')
        selected = selected[:limit]
    if not selected or len({p.name for p in selected}) != len(selected):
        raise ValueError('Use a nonempty sequence with unique image basenames.')
    indices = start + np.arange(len(selected))*stride
    fps = config.get('fps', 51032.)
    if isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite.')
    return selected, indices


def _outside_images(path, config):
    source = Path(config['image_directory']).expanduser().resolve()
    path = Path(path).expanduser().resolve()
    if path == source or source in path.parents or path in source.parents:
        raise ValueError('Keep TP annotations and outputs separate from the TIFF source folder.')
    return path


def _write_csv(path, fields, rows):
    path = Path(path)
    if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink > 1)):
        raise ValueError('Output must not replace a directory or file alias.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile('w', newline='', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as handle:
            temp = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _annotations(path, names):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {'filename', 'x_px', 'y_px'}.issubset(fields):
            raise ValueError('TP annotations need unique filename,x_px,y_px columns.')
        result = {}
        for row in reader:
            try:
                name = row['filename']
                point = np.array([float(row['x_px']), float(row['y_px'])])
                if None in row or name not in names or name in result or not np.all(np.isfinite(point)):
                    raise ValueError('Invalid or duplicate annotation')
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Invalid TP annotation row; use coordinates in original sensor pixels.') from exc
            result[name] = point
    return result


def launch_tp_annotation(config, annotation_path, *, count=20, review_csv=None, backend='TkAgg'):
    """Open a desktop click window in a separate process of the active kernel.

    Left click selects TP; Enter confirms and advances; Backspace goes back.
    Closing the window preserves confirmed positions. Rerunning resumes missing
    annotations. review_csv selects only uncertain tracking rows for correction.
    """
    destination = _outside_images(annotation_path, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', suffix='.json', dir=destination.parent,
                                     delete=False, encoding='utf-8') as handle:
        json.dump(config, handle); config_file = Path(handle.name)
    command = [sys.executable, '-m', 'schlieren_qml.triple_point', str(config_file),
               str(destination), '--count', str(count), '--backend', backend]
    if review_csv is not None:
        command += ['--review', str(Path(review_csv).resolve())]
    try:
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent.parent)
    finally:
        config_file.unlink(missing_ok=True)
    return destination


def _annotation_window(config, annotation_path, count, review_csv, backend):
    if backend not in {'TkAgg', 'QtAgg'}:
        raise ValueError('Choose the desktop backend TkAgg or QtAgg.')
    import matplotlib
    matplotlib.use(backend, force=True)
    import matplotlib.pyplot as plt

    paths, _ = _sequence(config)
    if count < 1:
        raise ValueError('count must be positive.')
    destination = _outside_images(annotation_path, config)
    points = _annotations(destination, {p.name for p in paths})
    proposed = {}
    if review_csv is None:
        pending = [p for p in paths[:count] if p.name not in points]
    else:
        with Path(review_csv).open(newline='', encoding='utf-8-sig') as handle:
            bad = [r for r in csv.DictReader(handle) if r['accepted'] != 'True']
        lookup = {p.name:p for p in paths}
        pending = []
        for row in bad:
            if row['filename'] not in lookup:
                raise ValueError('Review file is from another image sequence.')
            if row['filename'] not in points:
                pending.append(lookup[row['filename']])
                proposed[row['filename']] = [float(row['x_px']), float(row['y_px'])]
    if not pending:
        print('No missing manual annotations in this selection.'); return
    fig, ax = plt.subplots(figsize=(11, 7))
    state = {'index':0, 'point':None}

    def draw():
        path = pending[state['index']]
        image = read_image(path)
        ax.clear(); ax.imshow(image, cmap='gray', origin='upper')
        state['point'] = points.get(path.name)
        if path.name in proposed:
            x, y = proposed[path.name]; ax.plot(x, y, '+', color='#D55E00', ms=14)
        if state['point'] is not None:
            ax.plot(*state['point'], 'x', color='#0072B2', ms=12)
        ax.set(title=f"{state['index']+1}/{len(pending)}: {path.name}\n"
                     'Click TP, then Enter to confirm. Backspace: previous. Close: save and stop.',
               xlabel='Original sensor x (pixels)', ylabel='Original sensor y (pixels)')
        fig.canvas.draw_idle()

    def click(event):
        if event.inaxes != ax or event.button != 1 or event.xdata is None:
            return
        # Ignore pan/zoom gestures; the selected coordinates remain sensor pixels.
        if getattr(fig.canvas.manager.toolbar, 'mode', ''):
            return
        image = ax.images[0].get_array(); h, w = image.shape
        if not (0 <= event.xdata <= w-1 and 0 <= event.ydata <= h-1):
            return
        state['point'] = np.array([event.xdata, event.ydata])
        for line in list(ax.lines): line.remove()
        ax.plot(*state['point'], 'x', color='#0072B2', ms=12)
        fig.canvas.draw_idle()

    def key(event):
        if event.key == 'backspace' and state['index'] > 0:
            state['index'] -= 1; draw()
        elif event.key == 'enter' and state['point'] is not None:
            points[pending[state['index']].name] = state['point'].copy()
            _write_csv(destination, ['filename','x_px','y_px'],
                       [{'filename':p.name,'x_px':points[p.name][0],'y_px':points[p.name][1]}
                        for p in paths if p.name in points])
            state['index'] += 1
            if state['index'] == len(pending): plt.close(fig)
            else: draw()
    fig.canvas.mpl_connect('button_press_event', click)
    fig.canvas.mpl_connect('key_press_event', key)
    draw(); plt.show(block=True)


def _window_sums(image, size):
    integral = np.pad(image.cumsum(0).cumsum(1), ((1,0),(1,0)))
    return integral[size:,size:] - integral[:-size,size:] - integral[size:,:-size] + integral[:-size,:-size]


def _locate(image, templates, center, radius, roi, size):
    h, w = image.shape; half = size//2
    x0 = max(half, roi[0], int(np.floor(center[0]-radius)))
    x1 = min(w-half-1, roi[2]-1, int(np.ceil(center[0]+radius)))
    y0 = max(half, roi[1], int(np.floor(center[1]-radius)))
    y1 = min(h-half-1, roi[3]-1, int(np.ceil(center[1]+radius)))
    if x1 < x0 or y1 < y0:
        raise ValueError('Tracking search contains no complete template patches.')
    search = np.asarray(image[y0-half:y1+half+1, x0-half:x1+half+1], dtype=float)
    sums = _window_sums(search, size)
    variance = np.maximum(_window_sums(search*search, size)-sums*sums/(size*size), 0.)
    norm = np.sqrt(variance)
    maps = []
    for template, offset in templates:
        numerator = correlate(search, template, mode='valid', method='fft')
        maps.append(np.divide(numerator, norm, out=np.full_like(norm, -1.), where=norm > 1e-12))
    scores = np.clip(np.asarray(maps), -1., 1.)
    k, row, col = np.unravel_index(np.argmax(scores), scores.shape)
    score = float(scores[k,row,col]); combined = scores.max(0)
    other = combined.copy()
    yy, xx = np.ogrid[:other.shape[0], :other.shape[1]]
    other[(yy-row)**2+(xx-col)**2 <= max(2, half//2)**2] = -1.
    margin = score-float(other.max())
    point = np.array([x0+col,y0+row], dtype=float)+templates[k][1]
    return point, score, margin


def track_tp(config, annotation_path, output_directory, *, manual_count=20,
             patch_size=25, search_radius=20, search_roi=None, min_score=.75,
             min_margin=.05, max_templates=5, overwrite=False):
    """Track fixed TP templates locally; uncertain rows require manual review.

    Templates are drawn ONLY from the first manual_count frames. Later manual
    corrections are position anchors, never new appearance templates. A failed
    match does not update the tracking anchor. No uncertain row becomes a class.
    """
    paths, indices = _sequence(config)
    for value, name, minimum in [(manual_count,'manual_count',1),(patch_size,'patch_size',5),
                                  (search_radius,'search_radius',1),(max_templates,'max_templates',1)]:
        if isinstance(value,bool) or not isinstance(value,int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}.')
    if patch_size%2 != 1:
        raise ValueError('patch_size must be odd.')
    if not np.isfinite(min_score) or not 0 <= min_score <= 1 or not np.isfinite(min_margin) or not 0 <= min_margin <= 2:
        raise ValueError('Use min_score in [0,1] and min_margin in [0,2].')
    output = _outside_images(output_directory, config)
    annotation_path = Path(annotation_path).resolve()
    points = _annotations(annotation_path, {p.name for p in paths})
    seeds = paths[:min(manual_count,len(paths))]
    if any(p.name not in points for p in seeds):
        raise ValueError('Finish manually marking every initial seed image before tracking.')
    splits = make_splits(indices, config.get('train_fraction',.6), config.get('validation_fraction',.2), config.get('gap_frames',128))
    if len(seeds) > len(splits['train']):
        raise ValueError('Initial template annotations must lie entirely within the training block.')
    first = read_image(paths[0]); h,w = first.shape; half=patch_size//2
    roi = [0,0,w,h] if search_roi is None else list(search_roi)
    if (len(roi)!=4 or any(isinstance(v,bool) or not isinstance(v,int) for v in roi)
            or not 0 <= roi[0] < roi[2] <= w or not 0 <= roi[1] < roi[3] <= h):
        raise ValueError('search_roi must be original-pixel [x0,y0,x1,y1) bounds.')
    for point in points.values():
        if not (half <= point[0] <= w-half-1 and half <= point[1] <= h-half-1
                and roi[0] <= point[0] < roi[2] and roi[1] <= point[1] < roi[3]):
            raise ValueError('An annotation is outside the search ROI or too near an edge for patch_size.')
    names = ['tp_tracks.csv','tp_tracking_manifest.json']
    _preflight(output, names, [annotation_path], overwrite)
    templates=[]
    for j in np.linspace(0,len(seeds)-1,min(max_templates,len(seeds)),dtype=int):
        point=points[seeds[j].name]; x,y=np.rint(point).astype(int)
        image=read_image(seeds[j]); patch=np.asarray(image[y-half:y+half+1,x-half:x+half+1],dtype=float)
        patch-=patch.mean(); norm=np.linalg.norm(patch)
        if norm <= 1e-12:
            raise ValueError('A seed patch has no intensity structure; choose a smaller/more informative patch.')
        templates.append((patch/norm,point-[x,y]))
    rows=[]; anchor=points[paths[0].name].copy()
    for i,path in enumerate(paths):
        image = first if i==0 else read_image(path)
        if image.shape != first.shape:
            raise ValueError('Image dimensions changed during tracking.')
        if path.name in points:
            point=points[path.name]; score=margin=''; accepted=True; source='manual'
        else:
            point,score,margin=_locate(image,templates,anchor,search_radius,roi,patch_size)
            accepted=score >= min_score and margin >= min_margin; source='template'
        if accepted: anchor=point.copy()
        rows.append(dict(filename=path.name,sequence_frame_index=int(indices[i]),
                         time_s=float(indices[i]/config.get('fps',51032.)),x_px=float(point[0]),y_px=float(point[1]),
                         score=score,peak_margin=margin,accepted=bool(accepted),source=source))
        if (i+1)%1000==0: print(f'TP tracking: {i+1:,}/{len(paths):,}',flush=True)
    manifest={'method':'fixed-template normalized cross correlation; exploratory TP measurement',
              'coordinates':'original sensor pixels; y increases downward', 'config':config,
              'original_shape':[h,w],
              'annotation_sha256':hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
              'patch_size':patch_size,'search_radius':search_radius,'search_roi':roi,
              'min_score':min_score,'min_margin':min_margin,'manual_count':len(seeds),
              'template_filenames':[seeds[j].name for j in np.linspace(0,len(seeds)-1,min(max_templates,len(seeds)),dtype=int)],
              'n_frames':len(rows),'n_uncertain':sum(not r['accepted'] for r in rows),
              'localization':'Integer-pixel correlation peak plus the selected seed click offset; subpixel accuracy is not validated.',
              'limitation':'Correlation scores are not calibrated confidence probabilities. Inspect overlays and correct uncertain points.'}
    output.mkdir(parents=True,exist_ok=True)
    _write_csv(output/names[0],list(rows[0]),rows)
    _write_json(output/names[1],manifest)
    return {**manifest,'tracks_csv':str(output/names[0]),'output_directory':str(output)}


def _preflight(output, names, protected, overwrite):
    if not isinstance(overwrite,bool): raise ValueError('overwrite must be boolean.')
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not overwrite)):
        raise FileExistsError('Choose a new TP output directory or explicitly enable overwrite.')
    for name in names:
        path=output/name
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink>1)):
            raise ValueError('TP output must not replace a directory or file alias.')
        if any(_same_file(path,p) for p in protected):
            raise ValueError('TP outputs must not overwrite source annotations or tracks.')


def _write_json(path, value):
    # Serialize before touching a previous output.
    text=json.dumps(value,indent=2,allow_nan=False)
    path.write_text(text,encoding='utf-8')


def tp_categories(tracks_csv, output_directory, *, overwrite=False):
    """Freeze mean ± population SD of training y; export external SVM labels.

    Physically lower => larger y => category 1. Boundaries belong to category 2.
    Any uncertain track blocks final labels until corrected with the click tool.
    """
    tracks_csv=Path(tracks_csv).resolve()
    manifest_path=tracks_csv.parent/'tp_tracking_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    config=manifest['config']; paths,indices=_sequence(config)
    with tracks_csv.open(newline='',encoding='utf-8-sig') as handle: rows=list(csv.DictReader(handle))
    if len(rows)!=len(paths): raise ValueError('TP tracks do not match the configured image sequence.')
    for row,path,index in zip(rows,paths,indices):
        if row['filename']!=path.name or int(row['sequence_frame_index'])!=index:
            raise ValueError('TP track filename/index order differs from the selected images.')
        if row['accepted'] not in {'True','False'} or not np.all(np.isfinite([float(row['x_px']),float(row['y_px'])])):
            raise ValueError('Invalid TP measurement row.')
        h,w=manifest['original_shape']
        if not (0<=float(row['x_px'])<=w-1 and 0<=float(row['y_px'])<=h-1):
            raise ValueError('A TP coordinate is outside the original image.')
    if any(r['accepted']!='True' for r in rows):
        raise ValueError('Uncertain TP tracks remain. Inspect/correct them before exporting classification labels.')
    splits=make_splits(indices,config.get('train_fraction',.6),config.get('validation_fraction',.2),config.get('gap_frames',128))
    y=np.array([float(r['y_px']) for r in rows]); training=y[splits['train']]
    mean=float(training.mean()); sd=float(training.std(ddof=0))
    if len(training)<2 or sd<=0: raise ValueError('Training TP motion has zero spread; three position categories cannot be defined.')
    labels=np.where(y>mean+sd,'candidate_1',np.where(y<mean-sd,'candidate_3','candidate_2'))
    output=_outside_images(output_directory,config)
    names=['tp_labels.csv','tp_states.csv','tp_category_definition.json','tp_run_config.json',
           'tp_position_categories.png','tp_position_categories.pdf','tp_position_categories.svg']
    _preflight(output,names,[tracks_csv,manifest_path],overwrite)
    output.mkdir(parents=True,exist_ok=True)
    _write_csv(output/names[0],['filename','label'],[{'filename':p.name,'label':str(label)} for p,label in zip(paths,labels)])
    state_rows=[{**r,'label':str(label)} for r,label in zip(rows,labels)]
    _write_csv(output/names[1],list(state_rows[0]),state_rows)
    definition={'label_mode':'TP vertical position','fit_partition':'train','training_frames':len(training),
                'tracks_sha256':hashlib.sha256(tracks_csv.read_bytes()).hexdigest(),
                'mean_y_px':mean,'standard_deviation_y_px':sd,'ddof':0,
                'upper_boundary_y_px':mean-sd,'lower_boundary_y_px':mean+sd,
                'candidate_1':'physically lower: y > mean_y + sd_y',
                'candidate_2':'average band: mean_y - sd_y <= y <= mean_y + sd_y',
                'candidate_3':'physically higher: y < mean_y - sd_y',
                'limitation':'Position bins are operational states, not proof of three discrete flow regimes. Tracking error affects labels.'}
    _write_json(output/names[2],definition)
    run_config=dict(config);run_config['labels_csv']=str((output/names[0]).resolve())
    run_config['output_directory']=str((output.parent/(output.name+'_pod_qsvm')).resolve())
    _write_json(output/names[3],run_config)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    times=np.array([float(r['time_s']) for r in rows]);times=(times-times[0])*1000
    ax.plot(times,y,color='#0072B2',lw=.7,label='Measured/tracked TP')
    ax.axhline(mean,color='0.25',lw=1,label='Training mean')
    ax.axhspan(mean-sd,mean+sd,color='0.9',label='Category 2: training mean ± SD')
    for bound in [mean-sd,mean+sd]:ax.axhline(bound,color='#D55E00',ls='--',lw=.8)
    ax.invert_yaxis()
    ax.set(xlabel='Time from first selected frame (ms)',ylabel='TP y (sensor px)',
           title='TP position and training-defined category limits')
    ax.legend(frameon=False,fontsize=8)
    for extension in ['png','pdf','svg']:fig.savefig(output/f'tp_position_categories.{extension}',dpi=300)
    plt.close(fig)
    return {**definition,'labels_csv':str(output/names[0]),'config_path':str(output/names[3]),
            'states_csv':str(output/names[1]),'counts':{k:int(np.sum(labels==k)) for k in np.unique(labels)}}


def tp_gallery(config, tracks_csv, *, examples=9, seed=42, flagged_only=False, only_category=None):
    """Overlay proposed/measured TPs on original images for inspection."""
    paths,_=_sequence(config);lookup={p.name:p for p in paths}
    with Path(tracks_csv).open(newline='',encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
    if flagged_only:rows=[r for r in rows if r['accepted']!='True']
    if only_category is not None:rows=[r for r in rows if r.get('label')==only_category]
    if not rows:raise ValueError('No TP rows match this inspection selection.')
    if not isinstance(examples,int) or isinstance(examples,bool) or not 1<=examples<=24:raise ValueError('Choose 1–24 gallery examples.')
    rng=np.random.default_rng(seed); rows=[rows[i] for i in np.sort(rng.choice(len(rows),min(examples,len(rows)),replace=False))]
    images=[read_image(lookup[r['filename']]) for r in rows]
    import matplotlib.pyplot as plt
    cols=min(3,len(rows));fig,axes=plt.subplots(int(np.ceil(len(rows)/cols)),cols,squeeze=False,
                                               figsize=(4.5*cols,3.8*int(np.ceil(len(rows)/cols))),layout='constrained')
    low=min(im.min() for im in images);high=max(im.max() for im in images)
    for ax in axes.flat:ax.axis('off')
    for ax,row,image in zip(axes.flat,rows,images):
        ax.imshow(image,cmap='gray',vmin=low,vmax=high,origin='upper')
        ax.plot(float(row['x_px']),float(row['y_px']),'+',color='#D55E00',ms=12,mew=1.5)
        state=row.get('label',row['source']);quality='CHECK' if row['accepted']!='True' else 'accepted'
        ax.set_title(f"{row['filename']} | {state}\ny={float(row['y_px']):.2f} px | {quality}",fontsize=10)
    return {'figure':fig,'rows':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Click TP positions without altering TIFF images.')
    parser.add_argument('config');parser.add_argument('annotations');parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--review');parser.add_argument('--backend',default='TkAgg')
    args=parser.parse_args()
    _annotation_window(json.loads(Path(args.config).read_text(encoding='utf-8-sig')),
                       args.annotations,args.count,args.review,args.backend)
