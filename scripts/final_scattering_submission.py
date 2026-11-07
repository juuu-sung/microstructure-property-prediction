"""User-run fixed486 full500 TabPFN context and separate50:50 submission.

Built with PriorLabs-TabPFN. prepare/validate only audit saved inputs.
check uses synthetic500/1000 arrays. extract adds test scattering201 on CPU.
predict fits one500-row context and predicts1000 queries; no new selection.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import os
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
if __package__:
    from . import evaluate_scattering_tabpfn_confirmation as confirmation
    from . import final_tabpfn_submission as previous
else:
    import evaluate_scattering_tabpfn_confirmation as confirmation
    import final_tabpfn_submission as previous

t = confirmation.t
sf = confirmation.experiment.sf
original = previous.original
ROOT = t.ROOT
OUT = ROOT/'reports/final_scattering_blend'
ANCHORS = 16
SEED = previous.SEED+9
REP = confirmation.experiment.REP
PREVIOUS_SHA = '7548c628fc54df55ee90a8407bd6a5aa5386704a68c6cca25797fc7545982dcf'
PROTOCOL = {
    'version':1, 'attribution':'Built with PriorLabs-TabPFN', 'representation':REP,
    'features':'Fixed existing285 + archived scattering201;486 columns, no selection, clipping or normalization changes.',
    'train':'All500 labeled rows; reuse verified486 cache; TabPFN preprocessing fit only on context.',
    'test':'Reuse existing285 test cache; extract same201 on native256 images. All1000 queries in sample order, one predict call.',
    'scattering':'Same Kymatio0.3.0 J5/L8/order2 CPU float64 descriptor;2 threads; no image augmentation.',
    'anchor':{'choice':'first16 train.csv IDs; independent of targets','count':ANCHORS,'atol':1e-12,'rtol':1e-10},
    'predictor':'Same local TabPFN-v2,8 estimators,FP32,mean output,offline; exact archived CUDA runtime/config.',
    'full_context_seed':SEED, 'context_fits':1, 'gradient_training':False,
    'blend':{'base_hand185_resnet_dino':.5,'tabpfn486':.5},
    'base':'Original hand18525%+originalResNet25%+DINO_D450% predictions. Never blend previous TabPFN285 ensemble into486.',
    'effective_weights':{'hand185':.125,'originalResNet':.125,'DINO_D4':.25,'TabPFN486':.5},
    'previous_submission_sha256':PREVIOUS_SHA,'base_submission_sha256':previous.BASE_SHA,
    'output':'Separate sample-ordered ID,hardness CSV; preserve both prior submissions; no calibration, rounding or upload.',
    'resume':'Per-image test cells, test cache and single final inference cell verified; changed bindings fail.',
}


def protect_destination(folder):
    resolved = Path(folder).resolve()
    for protected in (previous.BASE,previous.OUT,ROOT/'open'):
        protected = protected.resolve()
        if resolved == protected or resolved.is_relative_to(protected) or protected.is_relative_to(resolved):
            raise ValueError('Output overlaps protected inputs/previous submission: '+str(folder))


def forbidden(*args):
    raise ValueError('Required archived result missing; no implicit fitting')


def previous_state(data, train486):
    old_data, features, bp, old_plan = previous.validate()
    np.testing.assert_array_equal(data.ids,old_data.ids)
    np.testing.assert_array_equal(data.y,old_data.y)
    np.testing.assert_array_equal(train486[:,:285],features['morphology285'])
    selection = previous.read_selection(previous.OUT,old_data,features,old_plan)
    if selection['representation'] != 'morphology285': raise ValueError('Expected submitted285 model')
    test285 = previous.read_test_cache(previous.OUT,old_plan,selection)
    test185,base = previous.baseline_test(bp)
    np.testing.assert_array_equal(test285[:,:185],test185)
    binding = {'plan':t.digest(old_plan['frozen']),'stage':'full500_test_prediction','runtime':selection['binding']['runtime'],
        'selection_sha256':t.common.sha256_file(previous.OUT/'selection/selection.json'),
        'test_features_sha256':t.common.sha256_file(previous.OUT/'test_features/features.npz')}
    with contextlib.redirect_stdout(io.StringIO()):
        p285 = t.run_cell(previous.OUT/'inference',binding,'morphology285',np.r_[features['morphology285'],test285],
                          data.y,np.arange(500),np.arange(500,1500),SEED,forbidden)
    output = previous.OUT/'submission'
    done = t.read_json(output/'completed.json')
    expected = {**binding,'inference_sha256':t.common.sha256_file(previous.OUT/'inference/completed.json'),
                'original_submission_sha256':previous.BASE_SHA,'representation':'morphology285'}
    if done['binding'] != expected or set(done['files']) != {'submission.csv','component_predictions.csv','validation.json'}:
        raise ValueError('Previous submission manifest differs')
    previous.check_refs([{'path':output/n,'sha256':sha} for n,sha in done['files'].items()])
    path = output/'submission.csv'
    if t.common.sha256_file(path) != PREVIOUS_SHA: raise ValueError('Previous submitted CSV changed')
    original.verify_submission(path,bp['test']['ids'],.5*(base+p285))
    old_values = t.core.finite_column(t.core.read_rows(path,{'ID','hardness'}),'hardness',path)
    metadata = t.read_json(previous.OUT/'inference/completed.json')['result']['metadata']
    return test285,base,old_values,bp,old_plan,metadata


def load_inputs():
    data,x,refs,cp = confirmation.validate()
    summaries = [confirmation.read_completed(confirmation.OUT,data,x,refs[s],cp,s) for s in confirmation.SEEDS]
    result = {**confirmation.aggregate(summaries),'protocol_sha256':t.digest(cp['frozen'])}
    review = t.read_json(confirmation.OUT/'review_results.json')
    if result != t.read_json(confirmation.OUT/'confirmation_summary.json') or review['comparison'] != result or not result['all_seeds_improve_both']:
        raise ValueError('Reviewed confirmation evidence changed')
    tx,base,old,bp,old_plan,metadata = previous_state(data,x)
    if x.shape != (500,486) or tx.shape != (1000,285) or set(data.ids)&set(bp['test']['ids']):
        raise ValueError('Invalid train/test dimensions or overlapping IDs')
    if any(r['runtime'] != metadata['runtime'] for r in refs.values()): raise ValueError('Archived runtime differs')
    prior_files = []
    for root in (previous.BASE,previous.OUT):
        prior_files += [t.file_ref(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.json','.npz','.csv')]
    frozen = {'protocol':PROTOCOL,'source':t.file_ref(__file__),'previous_source':t.file_ref(previous.__file__),
        'confirmation_plan':t.file_ref(confirmation.OUT/'protocol.json'),'confirmation_payload':t.digest(cp['frozen']),
        'confirmation_summary':t.file_ref(confirmation.OUT/'confirmation_summary.json'),
        'confirmation_review':t.file_ref(confirmation.OUT/'review_results.json'),
        'previous_plan':t.file_ref(previous.OUT/'plan.json'),'previous_payload':t.digest(old_plan['frozen']),
        'prior_files':prior_files,'train_ids':data.ids.tolist(),'target_sha256':t.common.sha256_array(data.y),
        'train486_sha256':t.common.sha256_array(x),'test285_sha256':t.common.sha256_array(tx),
        'test_binding':bp['test'],'anchor_ids':data.ids[:ANCHORS].tolist(),
        'anchor_values_sha256':t.common.sha256_array(x[:ANCHORS,285:]),
        'runtime':metadata['runtime'],'resolved_config':metadata['resolved_config'],
        'base_sha256':t.common.sha256_array(base),'previous_sha256':t.common.sha256_array(old)}
    return data,x,tx,base,old,frozen


def prepare(folder=OUT):
    protect_destination(folder)
    *_,frozen = load_inputs()
    if (folder/'plan.json').exists():
        validate(folder);print('Existing final scattering plan verified; unchanged');return
    if folder.exists() and any(folder.iterdir()):raise ValueError('Nonempty unprepared destination')
    t.write_json(folder/'plan.json',{'frozen':frozen,'created_unix':time.time(),'fits_during_preparation':0,'features_during_preparation':0})
    print(f'Prepared fixed486/full500/test1000/base50%+TabPFN48650%: {folder}; no fitting or extraction')


def validate(folder=OUT):
    data,x,tx,base,old,frozen = load_inputs()
    plan = t.read_json(folder/'plan.json')
    if plan['frozen'] != frozen or plan['fits_during_preparation'] != 0 or plan['features_during_preparation'] != 0:
        raise ValueError('Frozen final inputs/settings/source changed')
    return data,x,tx,base,old,plan


def anchor_record(folder,plan,expected,extractor=None):
    ids = plan['frozen']['anchor_ids']
    binding = {'plan':t.digest(plan['frozen']),'ids':ids,'expected_sha256':t.common.sha256_array(expected)}
    path = folder/'test_features/anchor.json'
    if path.exists():
        saved = t.read_json(path)
        if saved['binding'] != binding or saved['result_sha256'] != t.digest(saved['result']):raise ValueError('Anchor binding changed')
        actual = np.asarray(saved['result']['features'],dtype=np.float64)
    else:
        if extractor is None:raise ValueError('Train anchor incomplete; run extract first')
        actual = np.stack([extractor(sf.read_gray(ROOT/'open/train'/f'{i}.png'))[0] for i in ids])
    if actual.shape != (ANCHORS,201) or expected.shape != actual.shape or not np.isfinite(actual).all() or not np.allclose(actual,expected,rtol=1e-10,atol=1e-12):
        raise ValueError('Scattering train anchor mismatch; no test extraction allowed')
    result = {'features':actual.tolist(),'max_absolute_difference':float(np.max(np.abs(actual-expected))),'passed':True}
    if not path.exists():t.write_json(path,{'binding':binding,'result':result,'result_sha256':t.digest(result)})
    elif result != saved['result']:raise ValueError('Anchor summary differs')
    return t.file_ref(path)


def test_cell(folder,plan,item,extractor=None):
    source = ROOT/'open'/item['relative_path']
    if t.common.sha256_file(source) != item['sha256']:raise ValueError('Test image changed')
    binding = {'plan':t.digest(plan['frozen']),'image':item}
    path = folder/'test_features/cells'/f"{item['id']}.json"
    if path.exists():
        saved = t.read_json(path)
        if saved['binding'] != binding or saved['result_sha256'] != t.digest(saved['result']):raise ValueError('Test feature cell changed')
        row = np.asarray(saved['result']['features'],dtype=np.float64)
    else:
        if extractor is None:raise ValueError('Test cell missing; run extract first')
        row,detail = extractor(sf.read_gray(source))
        sf.validate_matrix(row[None,:])
        if t.common.sha256_file(source) != item['sha256']:raise ValueError('Test image changed during extraction')
        result = {'features':row.tolist(),'detail':detail}
        t.write_json(path,{'binding':binding,'result':result,'result_sha256':t.digest(result)})
    sf.validate_matrix(row[None,:])
    return row


def load_test_cache(folder,plan,train486,test285):
    path = folder/'test_features'
    if not (path/'completed.json').exists():raise FileNotFoundError('Test486 cache incomplete. Run extract successfully first.')
    binding = {'plan':t.digest(plan['frozen'])}
    done = t.read_json(path/'completed.json')
    if done['binding'] != binding or set(done['files']) != {'features.npz','metadata.json','anchor.json'}:raise ValueError('Test cache completion differs')
    previous.check_refs([{'path':path/n,'sha256':sha} for n,sha in done['files'].items()])
    arrays,meta = original.read_archive(path,'features.npz')
    ids = plan['frozen']['test_binding']['ids']
    if set(arrays) != {'ids','features','scattering'} or arrays['ids'].tolist() != ids or meta['binding'] != binding or meta['test_used_for_fit'] is not False:
        raise ValueError('Test cache IDs/schema/provenance differs')
    sf.validate_matrix(arrays['scattering'])
    if arrays['features'].shape != (len(ids),486) or arrays['scattering'].shape != (len(ids),201):raise ValueError('Test cache shape differs')
    np.testing.assert_array_equal(arrays['features'],np.c_[test285,arrays['scattering']])
    if anchor_record(folder,plan,train486[:ANCHORS,285:]) != meta['anchor']:raise ValueError('Anchor source changed')
    if set(meta['cells']) != {f'cells/{i}.json' for i in ids}:raise ValueError('Incomplete feature cell set')
    for j,item in enumerate(plan['frozen']['test_binding']['images']):
        name = f"cells/{item['id']}.json"
        if t.common.sha256_file(path/name) != meta['cells'][name]:raise ValueError('Test feature cell digest changed')
        np.testing.assert_array_equal(test_cell(folder,plan,item),arrays['scattering'][j])
    return arrays['features']


def extract_stage(folder=OUT):
    protect_destination(folder)
    _,train486,test285,_,_,plan = validate(folder)
    path = folder/'test_features'
    if (path/'completed.json').exists():
        load_test_cache(folder,plan,train486,test285);print('Test486 cache verified; no extraction');return
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        extractor = sf.Extractor()
        anchor = anchor_record(folder,plan,train486[:ANCHORS,285:],extractor)
        print('Train16 scattering anchor matched; beginning/resuming test extraction',flush=True)
        rows = []
        for j,item in enumerate(plan['frozen']['test_binding']['images']):
            rows.append(test_cell(folder,plan,item,extractor))
            if j == 0 or (j+1)%50 == 0:print(f'Test scattering: {j+1}/1000; elapsed={time.perf_counter()-started:.1f}s',flush=True)
    added = np.stack(rows); ids = plan['frozen']['test_binding']['ids']
    arrays = {'ids':np.array(ids),'features':np.c_[test285,added],'scattering':added}
    validate(folder)
    meta = {'binding':{'plan':t.digest(plan['frozen'])},'array_sha256':{n:t.common.sha256_array(v) for n,v in arrays.items()},
        'anchor':anchor,'cells':{f'cells/{i}.json':t.common.sha256_file(path/'cells'/f'{i}.json') for i in ids},
        'test_used_for_fit':False,'labels_saved':False,'seconds':time.perf_counter()-started}
    original.save_archive(path,arrays,meta)
    t.write_json(path/'completed.json',{'binding':meta['binding'],'files':{n:t.common.sha256_file(path/n) for n in ('features.npz','metadata.json','anchor.json')}})
    load_test_cache(folder,plan,train486,test285)
    print(f'Test486 ready: {path/"features.npz"}; shape=(1000,486); previous285 unchanged')


def write_submission(output,ids,base,pred,previous_values,binding):
    protect_destination(output)
    if len(set(ids)) != len(ids) or any(a.shape != (len(ids),) for a in (base,pred,previous_values)) or not np.isfinite([base,pred,previous_values]).all():
        raise ValueError('Invalid prediction arrays or IDs')
    values = .5*(base+pred)
    components = [{'ID':str(i),'base':float(base[j]),'tabpfn486':float(pred[j]),'previous_tabpfn285_blend':float(previous_values[j]),'hardness':float(values[j])} for j,i in enumerate(ids)]
    names = {'submission.csv','component_predictions.csv','validation.json'}
    if (output/'completed.json').exists():
        done = t.read_json(output/'completed.json')
        if done['binding'] != binding or set(done['files']) != names:raise ValueError('Completed submission binding differs')
        previous.check_refs([{'path':output/n,'sha256':sha} for n,sha in done['files'].items()])
        original.verify_submission(output/'submission.csv',ids,values)
        confirmation.same_rows(output/'component_predictions.csv',components)
        print('Separate scattering submission verified; unchanged');return
    output.mkdir(parents=True,exist_ok=True)
    tmp = output/'submission.csv.tmp';path = output/'submission.csv'
    tmp.write_text(original.submission_text(ids,values),encoding='utf-8',newline='')
    os.replace(tmp,path)
    original.verify_submission(path,ids,values)
    t.write_csv(output/'component_predictions.csv',components)
    delta = values-previous_values
    report = {'binding':binding,'rows':len(ids),'columns':['ID','hardness'],'sample_order_matches':True,'all_finite':True,
        'csv_round_trip_exact':True,'weights':PROTOCOL['blend'],'effective_weights':PROTOCOL['effective_weights'],
        'clipping_or_calibration':False,'uploaded':False,'prediction_min':float(values.min()),'prediction_max':float(values.max()),
        'prediction_mean':float(values.mean()),'prediction_std':float(values.std()),
        'mean_absolute_change_vs_previous':float(np.abs(delta).mean()),'max_absolute_change_vs_previous':float(np.abs(delta).max()),
        'submission_sha256':t.common.sha256_file(path),'attribution':PROTOCOL['attribution']}
    t.write_json(output/'validation.json',report)
    t.write_json(output/'completed.json',{'binding':binding,'files':{n:t.common.sha256_file(output/n) for n in sorted(names)}})
    print(f'Submission ready: {path}; rows={len(ids)}; previous retained; base50%+TabPFN48650%/IDs/finite/CSV verified')


def predict_stage(folder=OUT,device='cuda'):
    protect_destination(folder)
    data,train486,test285,base,old,plan = validate(folder)
    tx = load_test_cache(folder,plan,train486,test285)
    binding = {'plan':t.digest(plan['frozen']),'stage':'fixed486_full500_test1000','runtime':plan['frozen']['runtime'],
        'test_cache':t.file_ref(folder/'test_features/completed.json')}
    if (folder/'inference/completed.json').exists():predictor = forbidden
    else:
        predictor = t.Predictor(t.verify_weights(t.OUT),device)
        if predictor.runtime != plan['frozen']['runtime']:raise ValueError('Use exact archived CUDA runtime; no fallback')
    x = np.r_[train486,tx]
    p = t.run_cell(folder/'inference',binding,REP,x,data.y,np.arange(500),np.arange(500,1500),SEED,predictor)
    meta = t.read_json(folder/'inference/completed.json')['result']['metadata']
    if (meta['context_rows'],meta['query_rows'],meta['columns']) != (500,1000,486) or meta['runtime'] != plan['frozen']['runtime'] or meta['resolved_config'] != plan['frozen']['resolved_config']:
        raise ValueError('Final context/runtime/config differs')
    np.testing.assert_allclose(meta['context_mean'],data.y.mean(),atol=1e-9,rtol=0)
    np.testing.assert_allclose(meta['context_std'],data.y.std(),atol=1e-9,rtol=0)
    validate(folder)
    write_submission(folder/'submission',plan['frozen']['test_binding']['ids'],base,p,old,
        {**binding,'inference':t.file_ref(folder/'inference/completed.json'),
         'base_submission_sha256':previous.BASE_SHA,'previous_submission_sha256':PREVIOUS_SHA})


def check(device='cuda',folder=OUT):
    protect_destination(folder)
    predictor = t.Predictor(t.verify_weights(t.OUT),device)
    rng = np.random.default_rng(271828)
    x = rng.normal(size=(1500,486));x[:,0]=0.
    y = 190+3*x[:500,2]+rng.normal(size=500)
    p,meta = predictor(x[:500],y,x[500:],SEED)
    if p.shape != (1000,) or not np.isfinite(p).all():raise ValueError('Synthetic inference invalid')
    t.write_json(folder/'synthetic_check.json',{'competition_data_used':False,'shape':list(p.shape),'finite':True,
        'source':t.file_ref(__file__),'protocol':PROTOCOL,'metadata':meta})
    print(f'Synthetic500/1000 with486 columns passed; seconds={meta["seconds"]:.1f}; peak={meta["peak_gpu_allocated_bytes"]}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','validate','check','extract','predict','run'))
    p.add_argument('--device',default='cuda');a=p.parse_args(argv)
    if a.stage=='prepare':prepare()
    elif a.stage=='validate':validate();print('Final486 inputs/caches/prior submissions verified; no fit/extraction')
    elif a.stage=='check':check(a.device)
    else:
        if a.stage in ('extract','run'):extract_stage()
        if a.stage in ('predict','run'):predict_stage(device=a.device)


if __name__=='__main__':main()
