"""Shared, provenance-checked data access for the conference study."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime
import numpy as np

CLASSES = np.array(['candidate_1', 'candidate_2', 'candidate_3'])

def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        result = list(reader)
    if not fields or len(fields) != len(set(fields)) or any(None in r for r in result):
        raise ValueError(f'Malformed/duplicate CSV columns: {path}')
    return result

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def write_rows(path, values):
    from .triple_point import _write_csv
    if values:
        _write_csv(path, list(values[0]), values)

def category(y, definition):
    y = np.asarray(y)
    return np.where(y > definition['lower_boundary_y_px'], 'candidate_1',
        np.where(y < definition['upper_boundary_y_px'], 'candidate_3', 'candidate_2'))

def load_study(results_directory, states_csv, *, include_test=False):
    results = Path(results_directory).expanduser().resolve()
    states_path = Path(states_csv).expanduser().resolve()
    paths = dict(coefficients=results/'modal_coefficients.csv', states=states_path,
        config=results/'config_used.json', definition=states_path.parent/'tp_category_definition.json',
        model_definition=results/'tp_category_definition.json', pod=results/'pod_model.npz')
    config = json.loads(paths['config'].read_text())
    definition = json.loads(paths['definition'].read_text())
    model_definition = json.loads(paths['model_definition'].read_text())
    if definition.get('fit_partition') != 'train' or definition.get('tracks_sha256') != model_definition.get('tracks_sha256'):
        raise ValueError('The saved POD experiment and TP category definitions differ.')
    if not definition.get('tracks_sha256'):
        raise ValueError('TP provenance is missing.')
    coefficients, states = rows(paths['coefficients']), rows(states_path)
    lookup = {r['filename']: r for r in states}
    if (len(lookup) != len(states) or len({r['filename'] for r in coefficients}) != len(coefficients)
            or set(lookup) != {r['filename'] for r in coefficients}):
        raise ValueError('Coefficient and TP filenames must match uniquely.')
    columns = sorted([k for k in coefficients[0] if k.startswith('a_') and k[2:].isdigit()],key=lambda k:int(k[2:]))
    if not columns: raise ValueError('No saved POD coefficient columns.')
    allowed = {'train', 'validation', 'test'} if include_test else {'train','validation'}
    selected = [r for r in coefficients if r['partition'] in allowed]
    xy, features, index, time, names, parts = [], [], [], [], [], []
    for r in selected:
        tp = lookup[r['filename']]
        if r['sequence_frame_index'] != tp['sequence_frame_index'] or tp['accepted'] != 'True':
            raise ValueError('Unresolved or misaligned TP measurements.')
        if not np.isclose(float(r['time_s']),float(tp['time_s']),rtol=0,atol=1e-10):
            raise ValueError('POD/TP timestamps differ.')
        point = [float(tp['x_px']), float(tp['y_px'])]
        if tp['label'] != category([point[1]],definition)[0]:
            raise ValueError('TP states disagree with frozen training boundaries.')
        features.append([float(r[k]) for k in columns]); xy.append(point)
        index.append(int(r['sequence_frame_index'])); time.append(float(r['time_s']))
        names.append(r['filename']); parts.append(r['partition'])
    X, xy, index, time, names, parts = map(np.asarray,(features,xy,index,time,names,parts))
    if X.ndim != 2 or not np.isfinite(X).all() or not np.isfinite(xy).all() or not np.isfinite(time).all():
        raise ValueError('Study measurements/features must be finite.')
    frames_path=results/'frames.json'
    if frames_path.is_file():
        h,w=json.loads(frames_path.read_text())['original_shape']
        if np.any(xy<0) or np.any(xy[:,0]>w-1) or np.any(xy[:,1]>h-1):
            raise ValueError('TP coordinates fall outside the sensor.')
    split = {k:np.flatnonzero(parts == k) for k in allowed}
    order = ['train','validation'] + (['test'] if include_test else [])
    if any(len(split[k])<2 for k in order): raise ValueError('Each evaluated partition needs at least two frames.')
    if np.any(np.diff(index)<=0) or any(index[split[a][-1]]>=index[split[b][0]] for a,b in zip(order,order[1:])):
        raise ValueError('Partitions must be disjoint and chronological.')
    y = xy[:,1]; tr = split['train']
    if len(tr)!=definition['training_frames'] or not np.allclose(
        [y[tr].mean(),y[tr].std(ddof=0)],
        [definition['mean_y_px'],definition['standard_deviation_y_px']],rtol=1e-8,atol=1e-8):
        raise ValueError('Training TP positions do not match their saved definition.')
    if definition['standard_deviation_y_px']<=0: raise ValueError('Training TP spread must be positive.')
    from .pod import load_pod
    model=load_pod(paths['pod'])
    if model.training_count!=len(tr) or model.modes.shape[0]!=X.shape[1]:
        raise ValueError('Saved POD training count/rank does not match these coefficient partitions.')
    return dict(X=X, xy=xy, y=y, index=index, time=time, names=names, parts=parts, split=split,
        config=config, definition=definition, labels=category(y,definition), paths=paths,
        hashes={k:sha(p) for k,p in paths.items()}, results=results, states_path=states_path)

def check_output(path, data, *, empty=True):
    raw = Path(path).expanduser()
    if raw.is_symlink(): raise ValueError('Study output must not be a symlink.')
    path = raw.resolve()
    for source in [data['results'],data['states_path'].parent,Path(data['config']['image_directory']).resolve()]:
        if path==source or source in path.parents or path in source.parents:
            raise ValueError('Keep study output separate from source images and saved results.')
    if empty and path.exists() and any(path.iterdir()): raise FileExistsError('Choose a new empty study output directory.')
    path.mkdir(parents=True,exist_ok=True)
    return path

def new_study_directory(output_root):
    path=Path(output_root)/datetime.now().strftime('study_%Y%m%d_%H%M%S_%f')
    return path.resolve()

def save_figure(fig, output, name, dpi=300):
    import matplotlib.pyplot as plt
    with plt.rc_context({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'}):
        for ext in ['png','pdf','svg']:
            fig.savefig(Path(output)/f'{name}.{ext}',dpi=dpi,bbox_inches='tight')
    plt.close(fig)

def correlations(X, y):
    from scipy.stats import spearmanr
    result=[]
    for j in range(X.shape[1]):
        valid=np.std(X[:,j])>0 and np.std(y)>0
        result.append(dict(mode=j+1,pearson=float(np.corrcoef(X[:,j],y)[0,1]) if valid else None,
                           spearman=float(spearmanr(X[:,j],y).statistic) if valid else None))
    return result

def mode_order(X, y):
    r=correlations(X,y)
    return np.argsort([-abs(v['pearson']) if v['pearson'] is not None else np.inf for v in r],kind='stable')
