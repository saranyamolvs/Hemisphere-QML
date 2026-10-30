"""Validation-only fixed-dimension experiments, learning curves, and frozen test evaluation."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import platform
import importlib.metadata
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .paper_data import (load_study, check_output, write_json, write_rows, sha,
                         save_figure, correlations, mode_order, CLASSES)
from .classify import fit_validate_classifiers, _metrics
from .quantum import feature_states, fidelity_kernel


def _positive_integer(value,name,minimum=1):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return value

def _angle_scaler(X):
    scaler=MinMaxScaler(feature_range=(0.,np.pi)).fit(X)
    return scaler,scaler.data_max_==scaler.data_min_

def _angles(scaler,constant,X):
    raw=scaler.transform(X);raw[:,constant]=0.
    return np.clip(raw,0.,np.pi),float(np.mean(np.any((raw<0)|(raw>np.pi),axis=1)))

def _boundary_rows(data,indices,predictions,partition,margin):
    y=data['y'][indices];definition=data['definition']
    distance=np.minimum(abs(y-definition['upper_boundary_y_px']),abs(y-definition['lower_boundary_y_px']))
    result=[]
    for local,i in enumerate(indices):
        record=dict(filename=str(data['names'][i]),sequence_frame_index=int(data['index'][i]),
            partition=partition,time_s=float(data['time'][i]),tp_y_px=float(y[local]),
            true_category=str(data['labels'][i]),distance_to_boundary_px=float(distance[local]),
            outside_sensitivity_band=bool(distance[local]>margin))
        for name,predicted in predictions.items():
            record[f'{name}_prediction']=str(predicted[local])
            record[f'{name}_correct']=bool(predicted[local]==data['labels'][i])
        result.append(record)
    bins=[]
    edges=[0.,1.,2.,4.,8.,16.,np.inf]
    for low,high in zip(edges[:-1],edges[1:]):
        mask=(distance>=low)&(distance<high)
        label=f'{low:g}–{high:g}' if np.isfinite(high) else f'>={low:g}'
        record=dict(distance_bin_px=label,n=int(mask.sum()))
        for name,predicted in predictions.items():
            record[f'{name}_error_fraction']=float(np.mean(predicted[mask]!=data['labels'][indices][mask])) if mask.any() else None
        bins.append(record)
    retained=distance>margin
    sensitivity=dict(margin_px=margin,n_total=len(indices),n_retained=int(retained.sum()),
        coverage_fraction=float(retained.mean()),metrics={})
    for name,predicted in predictions.items():
        if retained.any():sensitivity['metrics'][name]=_metrics(data['labels'][indices][retained],predicted[retained],CLASSES)
    return result,bins,sensitivity

def _predict_fitted(fitted,angles):
    predictions={};timing={}
    start=perf_counter();predictions['rbf_svm']=fitted['models']['rbf_svm'].predict(angles)
    timing['rbf_prediction_s']=perf_counter()-start
    p=fitted['selected_hyperparameters']['qsvm'];start=perf_counter()
    states=feature_states(angles*p['angle_scale'],reps=p['reps'],entanglement=p['entanglement'])
    kernel=fidelity_kernel(states,fitted['train_states'])
    predictions['qsvm']=fitted['models']['qsvm'].predict(kernel)
    timing['quantum_states_kernel_prediction_s']=perf_counter()-start
    return predictions,timing

def _margin(data,measurement_report,assumed_margin_px):
    if not np.isfinite(assumed_margin_px) or assumed_margin_px<0:raise ValueError('Assumed margin must be finite and nonnegative.')
    if measurement_report is None:
        return float(assumed_margin_px),dict(source='assumed sensitivity width; not measured uncertainty')
    path=Path(measurement_report)
    report=json.loads(path.read_text())
    if report.get('status')!='complete':
        return float(assumed_margin_px),dict(source='repeat measurements pending; assumed sensitivity width',report_sha256=sha(path))
    if report['source_hashes']!=data['hashes']:raise ValueError('Uncertainty report is from a different experiment.')
    margin=float(report['boundary_sensitivity_margin_px'])
    if not np.isfinite(margin) or margin<0:raise ValueError('Invalid measured sensitivity width.')
    return margin,dict(source=report['margin_method'],report_sha256=sha(path))

def _crop_features(data,output,name,roi,bin_factor,rank,seed):
    from .data import read_image,preprocess_image
    from .triple_point import _sequence
    from .pod import fit_pod
    paths,_=_sequence(data['config']);lookup={p.name:p for p in paths}
    first=preprocess_image(read_image(lookup[str(data['names'][0])]),roi,bin_factor)
    rank=min(rank,len(data['split']['train'])-1,first.size)
    cache_path=output/'images_train_validation.npy'
    cache=np.lib.format.open_memmap(cache_path,mode='w+',dtype='float32',shape=(len(data['names']),first.size))
    start=perf_counter()
    try:
        for i,filename in enumerate(data['names']):
            image=preprocess_image(read_image(lookup[str(filename)]),roi,bin_factor)
            if image.shape!=first.shape:raise ValueError('Crop geometry changed.')
            cache[i]=image.ravel()
            if (i+1)%1000==0:print(f'{name}: crop {i+1:,}/{len(cache):,}',flush=True)
        cache.flush();preprocessing=perf_counter()-start;start=perf_counter()
        model=fit_pod(cache,data['split']['train'],rank=rank,seed=seed)
        fit_seconds=perf_counter()-start;start=perf_counter()
        X=model.transform(cache);projection=perf_counter()-start
        model.save(output/'pod_model.npz')
    finally:
        cache.flush();cache._mmap.close();del cache
        cache_path.unlink(missing_ok=True)
    x0,y0,x1,y1=roi
    coverage=(data['xy'][:,0]>=x0)&(data['xy'][:,0]<x1)&(data['xy'][:,1]>=y0)&(data['xy'][:,1]<y1)
    return X,dict(name=name,roi=roi,bin_factor=bin_factor,rank=rank,
        preprocessing_s=preprocessing,pod_fit_s=fit_seconds,projection_s=projection,
        model_sha256=sha(output/'pod_model.npz'),
        tp_inside_crop_fraction={p:float(coverage[data['split'][p]].mean()) for p in ['train','validation']},
        source='new training-only POD; no test images read')

def run_validation_study(results_directory,states_csv,output_directory,*,dimension=4,
        crops=None,crop_bin_factor=2,crop_rank=24,train_sizes=(64,128,256,512),
        seeds=(11,23,37),max_validation=512,final_train_size=512,
        classifier=None,measurement_report=None,assumed_margin_px=2.,dpi=300):
    """Select representations and hyperparameters without evaluating test values.

    Curves vary classifier training samples. Unlabeled POD uses all training
    images; TP category boundaries use all training TP measurements. Consequently
    these curves are NOT a claim about total annotation cost or independent runs.
    Each ranked feature set/scaler uses only its own sampled training rows.
    """
    data=load_study(results_directory,states_csv)
    dimension=_positive_integer(dimension,'dimension')
    if dimension>8:raise ValueError('Use at most 8 simulated qubits for the paper study.')
    _positive_integer(crop_rank,'crop_rank');_positive_integer(max_validation,'max_validation')
    sizes=list(dict.fromkeys(_positive_integer(v,'train size',3) for v in train_sizes))
    seed_values=list(dict.fromkeys(_positive_integer(v,'seed',0) for v in seeds))
    if not sizes or not seed_values:raise ValueError('Provide training sizes and seeds.')
    if final_train_size not in sizes:raise ValueError('final_train_size must be one requested learning-curve size.')
    tr=data['split']['train'];va=data['split']['validation']
    if max(sizes)>len(tr):raise ValueError('A requested classifier sample count exceeds training rows; reduce train_sizes.')
    if dimension>data['X'].shape[1] or crop_rank<dimension:raise ValueError('POD rank must cover the fixed feature dimension.')
    output=check_output(output_directory,data)
    va=va[np.linspace(0,len(va)-1,min(max_validation,len(va)),dtype=int)]
    margin,margin_info=_margin(data,measurement_report,assumed_margin_px)
    grid=dict(classifier or data['config'].get('classifier',{}))
    feature_sources={'saved_POD':data['X']};source_metadata={'saved_POD':dict(
        name='saved_POD',source='Existing training POD reused; historical preprocessing/POD cost not measured here.',
        roi=data['config'].get('roi'),bin_factor=data['config'].get('bin_factor',1),rank=data['X'].shape[1])}
    for number,(name,roi) in enumerate((crops or {}).items(),start=1):
        key=f'crop_{number:02d}';folder=output/key;folder.mkdir()
        X,metadata=_crop_features(data,folder,str(name),roi,crop_bin_factor,crop_rank,42)
        feature_sources[key]=X;source_metadata[key]=metadata
    write_json(output/'feature_sources.json',source_metadata)
    curves=[];search=[];validation_predictions=[];regression=[];mode_records=[];freeze_candidates={};sample_records=[]
    for source,X in feature_sources.items():
        source_correlations=correlations(X[tr],data['y'][tr])
        mode_records += [dict(source=source,**r) for r in source_correlations]
        for ordering in ['leading','TP_ranked']:
            representation=f'{source}__{ordering}'
            for seed in seed_values:
                permutation=np.random.default_rng(seed).permutation(tr)
                for size in sizes:
                    fit=np.sort(permutation[:size])
                    if len(np.unique(data['labels'][fit]))<2:
                        raise ValueError(f'{representation}, size {size}, seed {seed}: fewer than 2 training classes.')
                    columns=np.arange(dimension) if ordering=='leading' else mode_order(X[fit],data['y'][fit])[:dimension]
                    sample_records += [dict(representation=representation,seed=seed,n_train=size,
                        filename=str(data['names'][i]),sequence_frame_index=int(data['index'][i]),
                        true_category=str(data['labels'][i])) for i in fit]
                    started=perf_counter();scaler,constant=_angle_scaler(X[fit][:,columns])
                    angles,_=_angles(scaler,constant,X[fit][:,columns])
                    validation_angles,clipping=_angles(scaler,constant,X[va][:,columns])
                    scaling_seconds=perf_counter()-started
                    fitted=fit_validate_classifiers(angles,data['labels'][fit],validation_angles,data['labels'][va],grid)
                    predicted,timing=_predict_fitted(fitted,validation_angles)
                    for name in ['rbf_svm','qsvm']:
                        metrics=_metrics(data['labels'][va],predicted[name],CLASSES)
                        curves.append(dict(representation=representation,source=source,ordering=ordering,model=name,
                            seed=seed,n_train=size,n_validation=len(va),dimension=dimension,
                            modes=';'.join(str(int(j)+1) for j in columns),
                            accuracy=metrics['accuracy'],balanced_accuracy=metrics['balanced_accuracy'],macro_f1=metrics['macro_f1'],
                            class_support=json.dumps(metrics['class_support']),
                            training_accuracy=fitted['training'][name]['accuracy'],
                            validation_clipped_fraction=clipping,scaling_s=scaling_seconds,
                            validation_search_s=fitted['timing_seconds']['rbf_validation_search' if name=='rbf_svm' else 'quantum_states_kernels_validation_search'],
                            validation_prediction_s=timing['rbf_prediction_s' if name=='rbf_svm' else 'quantum_states_kernel_prediction_s'],
                            kernel_entries=size**2+size*len(va) if name=='qsvm' else None,
                            statevector_dimension=2**dimension if name=='qsvm' else None))
                        for r in fitted['search'][name]:
                            search.append(dict(representation=representation,model=name,seed=seed,n_train=size,
                                parameters=json.dumps({k:v for k,v in r.items() if k in ['C','gamma','reps','angle_scale']}),
                                balanced_accuracy=r['balanced_accuracy'],macro_f1=r['macro_f1']))
                    # Reference continuous-position diagnostic, same columns and rows.
                    regression_model=make_pipeline(StandardScaler(),Ridge(alpha=1.))
                    regression_model.fit(X[fit][:,columns],data['y'][fit]);yp=regression_model.predict(X[va][:,columns])
                    regression.append(dict(representation=representation,seed=seed,n_train=size,
                        ridge_alpha=1.,mae_px=float(mean_absolute_error(data['y'][va],yp)),
                        rmse_px=float(np.sqrt(mean_squared_error(data['y'][va],yp))),
                        r2=float(r2_score(data['y'][va],yp)) if np.std(data['y'][va])>0 else None))
                    for j,i in enumerate(va):
                        validation_predictions.append(dict(representation=representation,seed=seed,n_train=size,
                            filename=str(data['names'][i]),sequence_frame_index=int(data['index'][i]),
                            true_category=str(data['labels'][i]),tp_y_px=float(data['y'][i]),
                            rbf_svm_prediction=str(predicted['rbf_svm'][j]),qsvm_prediction=str(predicted['qsvm'][j])))
                    if size==final_train_size and seed==seed_values[0]:
                        path=output/f'{representation}_frozen.npz'
                        np.savez_compressed(path,train_angles=angles,train_labels=data['labels'][fit],
                            train_indices=data['index'][fit],columns=columns,scale=scaler.scale_,offset=scaler.min_,constant=constant)
                        freeze_candidates[representation]=dict(source=source,ordering=ordering,columns=columns.tolist(),
                            recipe=path.name,recipe_sha256=sha(path),parameters=fitted['selected_hyperparameters'],
                            train_size=size,seed=seed,dimension=dimension,validation_predictions=predicted)
                    print(f'{representation}: n={size}, seed={seed} complete',flush=True)
    comparison=[]
    for representation in freeze_candidates:
        for name in ['rbf_svm','qsvm']:
            subset=[r for r in curves if r['representation']==representation and r['model']==name and r['n_train']==final_train_size]
            values=[r['balanced_accuracy'] for r in subset]
            comparison.append(dict(representation=representation,model=name,n_train=final_train_size,
                n_repeats=len(values),mean_balanced_accuracy=float(np.mean(values)),
                sd_across_training_subsamples=float(np.std(values,ddof=1)) if len(values)>1 else 0.))
    scores={r:np.mean([v['mean_balanced_accuracy'] for v in comparison if v['representation']==r]) for r in freeze_candidates}
    selected=max(scores,key=scores.get)
    frozen=freeze_candidates[selected];predicted=frozen.pop('validation_predictions')
    for other in freeze_candidates.values():other.pop('validation_predictions',None)
    boundary,bins,sensitivity=_boundary_rows(data,va,predicted,'validation',margin)
    baseline=np.full(len(va),max(CLASSES,key=lambda k:np.sum(data['labels'][tr]==k)))
    selection=dict(status='frozen',representation=selected,**frozen,source_hashes=data['hashes'],
        paths={k:str(v) for k,v in data['paths'].items()},results_directory=str(data['results']),states_csv=str(data['states_path']),
        sensitivity_margin_px=margin,margin_provenance=margin_info,class_weight=grid.get('class_weight','balanced'),
        selection_rule='Highest mean of both models validation balanced accuracy across specified seeds at final_train_size; first representation breaks ties. Freeze first specified seed, never best seed.',
        class_boundary_definition=data['definition'],sources=source_metadata,
        learning_curve_scope='Classifier sample-size curves; POD and category boundaries use all training measurements/images. Not total label-efficiency curves.',
        validation_baseline=_metrics(data['labels'][va],baseline,CLASSES),
        continuous_position_baseline=dict(validation_rmse_px=float(np.sqrt(np.mean((data['y'][va]-data['y'][tr].mean())**2)))),
        settings=dict(train_sizes=sizes,seeds=seed_values,max_validation=max_validation,grid=grid),
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','matplotlib']},
        platform=platform.platform(),quantum_method='Exact noiseless classical statevector simulation; timing is not quantum hardware performance.')
    write_rows(output/'learning_curves.csv',curves);write_rows(output/'validation_search.csv',search)
    write_rows(output/'classifier_training_samples.csv',sample_records)
    write_rows(output/'fixed_dimension_comparison.csv',comparison)
    write_rows(output/'validation_predictions_all.csv',validation_predictions)
    write_rows(output/'pod_tp_correlations.csv',mode_records);write_rows(output/'continuous_tp_regression.csv',regression)
    write_rows(output/'validation_boundary_predictions.csv',boundary);write_rows(output/'validation_boundary_bins.csv',bins)
    write_json(output/'validation_boundary_sensitivity.json',sensitivity)
    write_json(output/'selection.json',selection)
    (output/'selection.sha256').write_text(sha(output/'selection.json'))
    _validation_plots(output,curves,comparison,mode_records,regression,bins,dpi)
    return dict(output_directory=str(output),selected_representation=selected,dimension=dimension,
        selection_json=str(output/'selection.json'),n_validation=len(va),sensitivity_margin_px=margin,
        message='Validation study only; final test not evaluated.')

def _boundary_plot(output,bins,partition,dpi):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(8,5),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
    x=np.arange(len(bins))
    for model,marker,color in [('rbf_svm','o','#0072B2'),('qsvm','s','#D55E00')]:
        axes[0].plot(x,[r[f'{model}_error_fraction'] if r[f'{model}_error_fraction'] is not None else np.nan for r in bins],
                     marker=marker,color=color,label=model)
    axes[0].set(ylabel='Misclassified fraction',ylim=(0,1),title=f'{partition.capitalize()} error versus TP boundary distance')
    axes[0].legend(frameon=False);axes[1].bar(x,[r['n'] for r in bins],color='0.65')
    axes[1].set(xticks=x,xticklabels=[r['distance_bin_px'] for r in bins],xlabel='Distance to nearest boundary (sensor px)',ylabel='Frames')
    save_figure(fig,output,f'{partition}_boundary_errors',dpi)

def _validation_plots(output,curves,comparison,corr,regression,bins,dpi):
    import matplotlib.pyplot as plt
    representations=list(dict.fromkeys(r['representation'] for r in curves))
    fig,axes=plt.subplots(len(representations),1,figsize=(8,2.8*len(representations)),squeeze=False,layout='constrained')
    for ax,representation in zip(axes.flat,representations):
        for model,color in [('rbf_svm','#0072B2'),('qsvm','#D55E00')]:
            rows=[r for r in curves if r['representation']==representation and r['model']==model]
            sizes=sorted({r['n_train'] for r in rows});mean=[];sd=[]
            for size in sizes:
                values=[r['balanced_accuracy'] for r in rows if r['n_train']==size]
                mean.append(np.mean(values));sd.append(np.std(values,ddof=1) if len(values)>1 else 0)
            ax.errorbar(sizes,mean,yerr=sd,marker='o',capsize=3,color=color,label=model)
        ax.set(title=representation,xlabel='Classifier training samples',ylabel='Validation balanced accuracy',ylim=(0,1.03))
        ax.legend(frameon=False)
    fig.suptitle('Matched classifier learning curves; bars show training-subsample SD, not confidence intervals')
    save_figure(fig,output,'02_validation_learning_curves',dpi)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(len(representations))
    for model,offset,color in [('rbf_svm',-.18,'#0072B2'),('qsvm',.18,'#D55E00')]:
        values=[next(r for r in comparison if r['model']==model and r['representation']==rep) for rep in representations]
        ax.bar(x+offset,[r['mean_balanced_accuracy'] for r in values],width=.36,
               yerr=[r['sd_across_training_subsamples'] for r in values],color=color,label=model,capsize=3)
    ax.set(xticks=x,xticklabels=representations,ylabel='Validation balanced accuracy',ylim=(0,1.03),title='Fixed feature dimension; mean ± training-subsample SD')
    ax.tick_params(axis='x',rotation=20);ax.legend(frameon=False)
    save_figure(fig,output,'03_fixed_dimension_comparison',dpi)
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    for source in dict.fromkeys(r['source'] for r in corr):
        subset=[r for r in corr if r['source']==source]
        ax.plot([r['mode'] for r in subset],[abs(r['pearson']) if r['pearson'] is not None else np.nan for r in subset],marker='.',label=source)
    ax.set(xlabel='POD mode',ylabel='Absolute training Pearson correlation with TP y',ylim=(0,1.03));ax.legend(frameon=False)
    save_figure(fig,output,'04_pod_tp_relationship',dpi)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for rep in representations:
        rows=[r for r in regression if r['representation']==rep];sizes=sorted({r['n_train'] for r in rows})
        ax.plot(sizes,[np.mean([r['rmse_px'] for r in rows if r['n_train']==size]) for size in sizes],marker='o',label=rep)
    ax.set(xlabel='Regression training samples',ylabel='Validation TP-y RMSE (px)',title='Continuous TP representation check: Ridge alpha=1')
    ax.legend(frameon=False,fontsize=8);save_figure(fig,output,'05_continuous_tp_prediction',dpi)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for model,color in [('rbf_svm','#0072B2'),('qsvm','#D55E00')]:
        for ax,column,title in [(axes[0],'validation_search_s','Validation hyperparameter-search wall time'),
                                (axes[1],'validation_prediction_s','Selected-model validation prediction wall time')]:
            sizes=sorted({r['n_train'] for r in curves})
            values=[np.mean([r[column] for r in curves if r['model']==model and r['n_train']==size]) for size in sizes]
            ax.plot(sizes,values,marker='o',color=color,label=model)
            ax.set(xlabel='Classifier training samples',ylabel='Wall time (s)',title=title)
            ax.legend(frameon=False)
    save_figure(fig,output,'06_classical_simulator_cost',dpi)
    _boundary_plot(output,bins,'validation',dpi)

def _paired_block_difference(truth,first,second,indices,*,block_frames=256,n_bootstrap=500,seed=42):
    _positive_integer(block_frames,'block_frames');_positive_integer(n_bootstrap,'n_bootstrap')
    groups=[np.flatnonzero(indices//block_frames==key) for key in np.unique(indices//block_frames)]
    if len(groups)<5:return dict(status='insufficient_blocks',n_blocks=len(groups),block_frames=block_frames)
    rng=np.random.default_rng(seed);values=[]
    support=set(truth)
    for _ in range(n_bootstrap):
        sample=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        if set(truth[sample])!=support:continue
        values.append(_metrics(truth[sample],second[sample],CLASSES)['balanced_accuracy']-
                      _metrics(truth[sample],first[sample],CLASSES)['balanced_accuracy'])
    if len(values)<max(20,n_bootstrap//2):return dict(status='insufficient_class_complete_replicates',n_valid=len(values))
    return dict(status='complete',n_blocks=len(groups),block_frames=block_frames,n_valid=len(values),
        metric='QSVM minus RBF balanced accuracy',percentile_95_interval=np.percentile(values,[2.5,97.5]).tolist(),
        limitation='Paired temporal-block sensitivity interval; validity depends on block length and stationarity. Not evidence of quantum computational advantage.')

def evaluate_frozen_test(study_directory,output_directory,*,max_test=256,
        test_usage='previously_inspected',block_frames=256,n_bootstrap=500,dpi=300):
    """Evaluate a frozen recipe once, without selecting anything from test."""
    from .pod import load_pod
    from .data import read_image,preprocess_image
    from .triple_point import _sequence
    study=Path(study_directory).resolve();selection_path=study/'selection.json'
    if sha(selection_path)!=(study/'selection.sha256').read_text().strip():raise ValueError('Frozen selection changed.')
    selection=json.loads(selection_path.read_text())
    if test_usage not in ['previously_inspected','unseen_record_or_holdout']:
        raise ValueError('Explicitly describe test usage as previously_inspected or unseen_record_or_holdout.')
    data=load_study(selection['results_directory'],selection['states_csv'],include_test=True)
    if data['hashes']!=selection['source_hashes']:raise ValueError('Source experiment changed since validation selection.')
    recipe_path=study/selection['recipe']
    if sha(recipe_path)!=selection['recipe_sha256']:raise ValueError('Frozen training recipe changed.')
    _positive_integer(max_test,'max_test')
    output=check_output(output_directory,data)
    if output==study or output in study.parents:raise ValueError('Test output must not replace the frozen study.')
    recipe=np.load(recipe_path,allow_pickle=False)
    te=data['split']['test'];te=te[np.linspace(0,len(te)-1,min(max_test,len(te)),dtype=int)]
    source=selection['source'];start=perf_counter()
    if source=='saved_POD':X=data['X'][te]
    else:
        paths,_=_sequence(data['config']);lookup={p.name:p for p in paths}
        info=selection['sources'][source]
        if sha(study/source/'pod_model.npz')!=info['model_sha256']:raise ValueError('Frozen crop POD model changed.')
        model=load_pod(study/source/'pod_model.npz');values=[]
        for i in te:
            image=preprocess_image(read_image(lookup[str(data['names'][i])]),info['roi'],info['bin_factor'])
            values.append(model.transform(image.reshape(1,-1))[0])
        X=np.asarray(values)
    feature_seconds=perf_counter()-start
    X=X[:,recipe['columns']]
    raw=X*recipe['scale']+recipe['offset'];raw[:,recipe['constant']]=0
    angles=np.clip(raw,0,np.pi);clipping=float(np.mean(np.any((raw<0)|(raw>np.pi),axis=1)))
    train_angles=recipe['train_angles'];labels=recipe['train_labels'];parameters=selection['parameters']
    start=perf_counter();rbf=SVC(kernel='rbf',class_weight=selection['class_weight'],**parameters['rbf_svm'])
    rbf.fit(train_angles,labels);rbf_fit=perf_counter()-start
    start=perf_counter();first=rbf.predict(angles);rbf_predict=perf_counter()-start
    p=parameters['qsvm'];start=perf_counter()
    train_states=feature_states(train_angles*p['angle_scale'],reps=p['reps'],entanglement=p['entanglement'])
    K=fidelity_kernel(train_states)
    q=SVC(kernel='precomputed',C=p['C'],class_weight=selection['class_weight']);q.fit(K,labels)
    quantum_fit=perf_counter()-start;start=perf_counter()
    test_states=feature_states(angles*p['angle_scale'],reps=p['reps'],entanglement=p['entanglement'])
    K_test=fidelity_kernel(test_states,train_states);second=q.predict(K_test)
    quantum_predict=perf_counter()-start
    predictions=dict(rbf_svm=first,qsvm=second)
    boundary,bins,sensitivity=_boundary_rows(data,te,predictions,'test',selection['sensitivity_margin_px'])
    majority=max(CLASSES,key=lambda k:np.sum(labels==k))
    metrics={name:_metrics(data['labels'][te],pred,CLASSES) for name,pred in predictions.items()}
    metrics['majority']=_metrics(data['labels'][te],np.full(len(te),majority),CLASSES)
    report=dict(test_usage=test_usage,metrics=metrics,representation=selection['representation'],
        selection_sha256=sha(selection_path),source_hashes=data['hashes'],n_test=len(te),
        test_clipped_fraction=clipping,secondary_boundary_sensitivity=sensitivity,
        timing_seconds=dict(test_feature_projection_or_table_extraction=feature_seconds,
            rbf_selected_fit=rbf_fit,rbf_prediction=rbf_predict,
            quantum_training_states_kernel_and_selected_fit=quantum_fit,
            quantum_test_states_kernel_and_prediction=quantum_predict),
        quantum_dense_array_bytes=int(train_states.nbytes+test_states.nbytes+K.nbytes+K_test.nbytes),
        memory_note='Array storage only, not peak process memory; simulation wall time, not quantum hardware cost.',
        paired_temporal_block_difference=_paired_block_difference(data['labels'][te],first,second,data['index'][te],
            block_frames=block_frames,n_bootstrap=n_bootstrap),
        limitations='Previously inspected test results are retrospective, not fresh generalization evidence. Sampled high-speed frames can remain temporally dependent.')
    write_rows(output/'test_predictions.csv',boundary);write_rows(output/'test_boundary_bins.csv',bins)
    write_json(output/'final_evaluation.json',report);_boundary_plot(output,bins,'test',dpi)
    _final_confusion(data['labels'][te],predictions,output,dpi,test_usage)
    return dict(output_directory=str(output),report_json=str(output/'final_evaluation.json'),**report)

def _final_confusion(truth,predictions,output,dpi,test_usage):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    source=[]
    for ax,(name,pred) in zip(axes,predictions.items()):
        cm=confusion_matrix(truth,pred,labels=CLASSES);support=cm.sum(axis=1)
        values=np.divide(cm,support[:,None],out=np.zeros_like(cm,dtype=float),where=support[:,None]>0)
        im=ax.imshow(values,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):
                label=f'{values[i,j]:.1%}\n(n={cm[i,j]})' if support[i] else 'No support'
                ax.text(j,i,label,ha='center',va='center',fontsize=9,color='white' if values[i,j]>.5 else 'black')
                source.append(dict(model=name,true_category=str(CLASSES[i]),predicted_category=str(CLASSES[j]),count=int(cm[i,j]),row_support=int(support[i])))
        metric=_metrics(truth,pred,CLASSES)
        ax.set(xticks=range(3),yticks=range(3),xticklabels=['C1','C2','C3'],yticklabels=['C1','C2','C3'],
               xlabel='Predicted TP category',ylabel='Reference TP category',title=f"{name}\nBalanced accuracy {metric['balanced_accuracy']:.1%}")
    fig.colorbar(im,ax=axes,label='Fraction within reference category',shrink=.8)
    fig.suptitle(f'N={len(truth)} test samples | {test_usage}')
    write_rows(output/'test_confusion_counts.csv',source)
    save_figure(fig,output,'07_frozen_test_confusion',dpi)
