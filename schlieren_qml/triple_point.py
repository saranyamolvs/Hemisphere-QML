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


_REVIEW_FIELDS = ['filename', 'x_px', 'y_px', 'decision', 'proposal_x_px',
                  'proposal_y_px', 'tracks_sha256']


def _review_destination(annotation_path, review_path=None):
    return Path(review_path) if review_path is not None else Path(annotation_path).with_name('tp_review.csv')


def _reviews(path, names):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _REVIEW_FIELDS:
            raise ValueError('TP review CSV has unexpected columns; use the generated review file.')
        result = {}
        for row in reader:
            try:
                coords = [float(row[k]) for k in ['x_px','y_px','proposal_x_px','proposal_y_px']]
                if (None in row or row['filename'] not in names or row['filename'] in result
                        or row['decision'] not in {'accept_existing','corrected'}
                        or not np.all(np.isfinite(coords))):
                    raise ValueError('Invalid review row')
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError('Invalid TP review row.') from exc
            result[row['filename']] = row
    return result


def _same_point(left, right):
    return bool(np.allclose(left, right, atol=1e-7, rtol=0))


def launch_tp_annotation(config, annotation_path, *, count=20, review_csv=None,
                         backend='TkAgg', review_path=None, review_all=False):
    """Open the desktop seed/review window in the active Python environment.

    Seed mode: click + Enter saves missing initial positions. Review mode:
    A accepts the existing proposal; click + Enter corrects it; S moves on
    without a new decision (a flagged unresolved position stays unresolved).
    Previously accepted/manual positions retain their status when deferred.
    A is a recorded human decision, not a new manual tracking anchor.
    """
    destination = _outside_images(annotation_path, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', suffix='.json', dir=destination.parent,
                                     delete=False, encoding='utf-8') as handle:
        json.dump(config, handle); config_file = Path(handle.name)
    command = [sys.executable, '-m', 'schlieren_qml.triple_point', str(config_file),
               str(destination), '--count', str(count), '--backend', backend]
    if review_csv is not None:
        command += ['--review', str(Path(review_csv).resolve()), '--review-path',
                    str(_review_destination(destination, review_path).resolve())]
        if review_all:
            command.append('--review-all')
    try:
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent.parent)
    finally:
        config_file.unlink(missing_ok=True)
    return destination


def _annotation_window(config, annotation_path, count, review_csv, backend,
                       review_path=None, review_all=False):
    if backend not in {'TkAgg', 'QtAgg'}:
        raise ValueError('Choose the desktop backend TkAgg or QtAgg.')
    import matplotlib
    matplotlib.use(backend, force=True)
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    paths, _ = _sequence(config)
    if count < 1:
        raise ValueError('count must be positive.')
    destination = _outside_images(annotation_path, config)
    names = {p.name for p in paths}
    points = _annotations(destination, names)
    review_destination = _outside_images(_review_destination(destination, review_path), config)
    if _same_file(review_destination, destination):
        raise ValueError('Review CSV and manual annotation CSV must be different files.')
    proposed = {}; reviews = {}; review_hash = ''; tracking_rows = {}
    if review_csv is None:
        pending = [p for p in paths[:count] if p.name not in points]
    else:
        review_csv = Path(review_csv).resolve()
        if _same_file(review_destination, review_csv) or _same_file(destination, review_csv):
            raise ValueError('Review outputs must not overwrite the tracking CSV.')
        with review_csv.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        lookup = {p.name:p for p in paths}
        reviews = _reviews(review_destination, names)
        review_hash = hashlib.sha256(review_csv.read_bytes()).hexdigest()
        pending = []
        for row in rows:
            if row['filename'] not in lookup:
                raise ValueError('Review file is from another image sequence.')
            tracking_rows[row['filename']] = row
            proposal = [float(row['x_px']), float(row['y_px'])]
            reviewed = reviews.get(row['filename'])
            # Resume only decisions that still refer to this proposal. A new
            # tracking result at a different position always needs new review.
            resolved = bool(reviewed and (
                _same_point(proposal, [float(reviewed['x_px']),float(reviewed['y_px'])])
                or (reviewed['decision']=='corrected'
                    and _same_point(proposal,[float(reviewed['proposal_x_px']),float(reviewed['proposal_y_px'])])
                    and row['filename'] in points
                    and _same_point(points[row['filename']], [float(reviewed['x_px']),float(reviewed['y_px'])]))))
            if review_all or (row['accepted'] != 'True' and not resolved):
                pending.append(lookup[row['filename']]); proposed[row['filename']] = proposal
    if not pending:
        print('No missing manual annotations or unresolved reviews in this selection.'); return
    # Matplotlib normally binds S to Save Figure and Backspace to navigation.
    # Reserve review keys only while this desktop window is active; ctrl+s and
    # all unrelated shortcuts remain available, and rc_context restores them.
    reserved_keys = {'enter','backspace'} | ({'a','s'} if review_csv is not None else set())
    keymaps = {name:[key for key in matplotlib.rcParams[name] if key not in reserved_keys]
               for name in matplotlib.rcParams if name.startswith('keymap.')}
    with matplotlib.rc_context(keymaps):
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.subplots_adjust(bottom=.18, top=.85)
        state = {'index':0, 'point':None}

        def draw():
            path = pending[state['index']]
            image = read_image(path)
            ax.clear(); ax.imshow(image, cmap='gray', origin='upper')
            state['point'] = None if review_csv is not None else points.get(path.name)
            if path.name in proposed:
                x, y = proposed[path.name]; ax.plot(x, y, '+', color='#D55E00', ms=14)
            if state['point'] is not None:
                ax.plot(*state['point'], 'x', color='#0072B2', ms=12)
            instructions = ('A: keep correct proposal | click + Enter: correct | S: defer'
                            if review_csv is not None else 'Click TP, then Enter to save.')
            quality = ''
            if path.name in tracking_rows:
                row = tracking_rows[path.name]
                quality = f" | score={row['score']} | margin={row['peak_margin']}"
            ax.set(title=f"{state['index']+1}/{len(pending)}: {path.name}{quality}\n"
                         f'{instructions}\nBackspace: previous. Close: preserve saved decisions.',
                   xlabel='Original sensor x (pixels)', ylabel='Original sensor y (pixels)')
            fig.canvas.draw_idle()

        def click(event):
            if event.inaxes != ax or event.button != 1 or event.xdata is None:
                return
            if getattr(fig.canvas.manager.toolbar, 'mode', ''):
                return
            image = ax.images[0].get_array(); h, w = image.shape
            if not (0 <= event.xdata <= w-1 and 0 <= event.ydata <= h-1):
                return
            state['point'] = np.array([event.xdata, event.ydata])
            for line in list(ax.lines): line.remove()
            if pending[state['index']].name in proposed:
                ax.plot(*proposed[pending[state['index']].name], '+', color='#D55E00', ms=14)
            ax.plot(*state['point'], 'x', color='#0072B2', ms=12)
            fig.canvas.draw_idle()

        def advance():
            state['index'] += 1
            if state['index'] == len(pending): plt.close(fig)
            else: draw()

        def save_review(decision, point):
            name = pending[state['index']].name
            proposal = proposed[name]
            reviews[name] = dict(filename=name,x_px=float(point[0]),y_px=float(point[1]),
                                 decision=decision,proposal_x_px=proposal[0],proposal_y_px=proposal[1],
                                 tracks_sha256=review_hash)
            _write_csv(review_destination,_REVIEW_FIELDS,[reviews[p.name] for p in paths if p.name in reviews])

        def key(event):
            if state['index'] >= len(pending): return
            if event.key == 'backspace' and state['index'] > 0:
                state['index'] -= 1; draw()
            elif event.key == 'a' and review_csv is not None:
                name = pending[state['index']].name
                # Never silently discard a manual correction when accepting the
                # older automatic proposal. Use click+Enter for any position edit.
                if name in points and not _same_point(points[name],proposed[name]):
                    print('A manual annotation already exists here; click its correct location and press Enter.')
                    return
                save_review('accept_existing',proposed[name]); advance()
            elif event.key == 's' and review_csv is not None:
                # Move on without a new decision; a pending flag stays unresolved.
                # Existing accepted/manual/reviewed status is not revoked.
                advance()
            elif event.key == 'enter' and state['point'] is not None:
                points[pending[state['index']].name] = state['point'].copy()
                _write_csv(destination, ['filename','x_px','y_px'],
                           [{'filename':p.name,'x_px':points[p.name][0],'y_px':points[p.name][1]}
                            for p in paths if p.name in points])
                if review_csv is not None: save_review('corrected',state['point'])
                advance()

        # Retain Button references for the lifetime of this blocking window.
        buttons = []
        if review_csv is not None:
            for left,width,label,shortcut in [(.12,.24,'Keep TP (A)','a'),
                                              (.39,.25,'Save correction (Enter)','enter'),
                                              (.67,.20,'Defer (S)','s')]:
                button = Button(fig.add_axes([left,.04,width,.055]),label)
                button.on_clicked(lambda event,shortcut=shortcut:key(type('Key',(),{'key':shortcut})()))
                buttons.append(button)
        fig.canvas.mpl_connect('button_press_event', click)
        fig.canvas.mpl_connect('key_press_event', key)
        draw(); plt.show(block=True)


def apply_tp_review(tracks_csv, annotation_path, review_path=None):
    """Apply saved decisions without retracking; preserve automatic quality.

    Existing-proposal acceptance only applies at its recorded coordinate.
    Corrections must match both their recorded proposal and the manual CSV.
    Deferred/unreviewed positions remain unresolved and still block categories.
    """
    tracks_csv = Path(tracks_csv).resolve()
    manifest_path = tracks_csv.parent/'tp_tracking_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    config = manifest['config']; paths,_ = _sequence(config); names={p.name for p in paths}
    annotation_path = _outside_images(annotation_path,config)
    review_path = _outside_images(_review_destination(annotation_path,review_path),config)
    if any(_same_file(out,source) for out in [tracks_csv,manifest_path]
           for source in [annotation_path,review_path]):
        raise ValueError('Tracking outputs must not overwrite annotations or reviews.')
    points = _annotations(annotation_path,names); reviews = _reviews(review_path,names)
    with tracks_csv.open(newline='',encoding='utf-8-sig') as handle:
        rows=list(csv.DictReader(handle))
    if [r['filename'] for r in rows] != [p.name for p in paths]:
        raise ValueError('Review tracks do not match the configured image sequence.')
    changed = 0
    h,w = manifest['original_shape']
    for row in rows:
        row.setdefault('automatic_accepted',row['accepted'] if row['source']=='template' else '')
        row.setdefault('review_status','unreviewed' if row['source']=='template' else 'manual')
        decision = reviews.get(row['filename'])
        coordinate = [float(row['x_px']),float(row['y_px'])]
        if row['review_status']=='accepted_existing' and (not decision or not _same_point(
                coordinate,[float(decision['x_px']),float(decision['y_px'])])):
            row['accepted']=row['automatic_accepted'];row['review_status']='unreviewed'
        if not decision: continue
        reviewed = [float(decision['x_px']),float(decision['y_px'])]
        if not (0<=reviewed[0]<=w-1 and 0<=reviewed[1]<=h-1):
            raise ValueError('Reviewed TP coordinate is outside the original image.')
        if decision['decision']=='accept_existing' and _same_point(coordinate,reviewed):
            row['accepted']='True';row['review_status']='accepted_existing';changed+=1
        elif decision['decision']=='corrected' and row['filename'] in points:
            proposal=[float(decision['proposal_x_px']),float(decision['proposal_y_px'])]
            if (_same_point(points[row['filename']],reviewed)
                    and (_same_point(coordinate,proposal) or _same_point(coordinate,reviewed))):
                row['x_px'],row['y_px']=map(str,reviewed)
                row['accepted']='True';row['source']='manual';row['review_status']='corrected';changed+=1
    _preflight(tracks_csv.parent,[tracks_csv.name,manifest_path.name],[annotation_path,review_path],True)
    _write_csv(tracks_csv,list(rows[0]),rows)
    manifest['n_uncertain']=sum(row['accepted']!='True' for row in rows)
    manifest['n_human_accepted_existing']=sum(row['review_status']=='accepted_existing' for row in rows)
    manifest['review_csv']=str(review_path)
    manifest['review_sha256']=hashlib.sha256(review_path.read_bytes()).hexdigest() if review_path.exists() else None
    manifest['annotation_sha256']=hashlib.sha256(annotation_path.read_bytes()).hexdigest() if annotation_path.exists() else None
    _write_json(manifest_path,manifest)
    return {**manifest,'tracks_csv':str(tracks_csv),'n_review_decisions_applied':changed}


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
             min_margin=.05, max_templates=5, overwrite=False, review_path=None):
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
    review_path = _outside_images(_review_destination(annotation_path,review_path),config)
    if _same_file(review_path,annotation_path):
        raise ValueError('Review and annotation CSVs must be different files.')
    reviews = _reviews(review_path,{p.name for p in paths})
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
    _preflight(output, names, [annotation_path,review_path], overwrite)
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
        automatic_accepted = bool(accepted) if source=='template' else ''
        review_status = 'manual' if source=='manual' else 'unreviewed'
        reviewed = reviews.get(path.name)
        if reviewed and _same_point(point,[float(reviewed['x_px']),float(reviewed['y_px'])]):
            if reviewed['decision']=='accept_existing':
                accepted=True;review_status='accepted_existing'
            elif source=='manual':
                review_status='corrected'
        if accepted: anchor=point.copy()
        rows.append(dict(filename=path.name,sequence_frame_index=int(indices[i]),
                         time_s=float(indices[i]/config.get('fps',51032.)),x_px=float(point[0]),y_px=float(point[1]),
                         score=score,peak_margin=margin,accepted=bool(accepted),source=source,
                         automatic_accepted=automatic_accepted,review_status=review_status))
        if (i+1)%1000==0: print(f'TP tracking: {i+1:,}/{len(paths):,}',flush=True)
    manifest={'method':'fixed-template normalized cross correlation; exploratory TP measurement',
              'coordinates':'original sensor pixels; y increases downward', 'config':config,
              'original_shape':[h,w],
              'annotation_sha256':hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
              'patch_size':patch_size,'search_radius':search_radius,'search_roi':roi,
              'min_score':min_score,'min_margin':min_margin,'manual_count':len(seeds),
              'template_filenames':[seeds[j].name for j in np.linspace(0,len(seeds)-1,min(max_templates,len(seeds)),dtype=int)],
              'n_frames':len(rows),'n_uncertain':sum(not r['accepted'] for r in rows),
              'review_csv':str(review_path),
              'review_sha256':hashlib.sha256(review_path.read_bytes()).hexdigest() if review_path.exists() else None,
              'n_human_accepted_existing':sum(r['review_status']=='accepted_existing' for r in rows),
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
                'tracking_review': {
                    'review_csv':manifest.get('review_csv'),
                    'review_sha256':manifest.get('review_sha256'),
                    'annotation_sha256':manifest.get('annotation_sha256'),
                    'n_manual':sum(r['source']=='manual' for r in rows),
                    'n_human_accepted_existing':sum(r.get('review_status')=='accepted_existing' for r in rows),
                    'n_corrected':sum(r.get('review_status')=='corrected' for r in rows),
                    'n_unreviewed_automatic':sum(r['source']=='template' and r.get('review_status','unreviewed')=='unreviewed' for r in rows),
                    'n_original_automatic_flags':sum(r.get('automatic_accepted')=='False' for r in rows),
                },
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
    parser.add_argument('--review-path');parser.add_argument('--review-all',action='store_true')
    args=parser.parse_args()
    _annotation_window(json.loads(Path(args.config).read_text(encoding='utf-8-sig')),
                       args.annotations,args.count,args.review,args.backend,args.review_path,args.review_all)
