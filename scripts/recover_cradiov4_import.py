"""Recover already-extracted features after the known setuptools-path bug.

No models are instantiated, no inference/fitting occurs, no old file is changed.
This migration accepts exactly the audited source-only patch, not arbitrary
protocol/input/environment changes. Old32 chunks stay in the original folder.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import time
import numpy as np

if __package__:
    from . import cradiov4_experiment as experiment
else:
    import cradiov4_experiment as experiment

e=experiment.e
ROOT=experiment.ROOT
OLD_PLAN='5fce57e16870909e5eb0e01b0c0627ae87f01d6a3b1a2ce4e73757aba0a2f1bc'
PATCHED_SOURCES={
    'cradiov4_engine.py':(
        '1982aa4e6796f43a17f9ea1313929f8a175b33c165e39deb2bd2c55a03ae6a0a',
        '52eb7b4ba14b558429e13f430e409b9d58f414c7dddaca04670c4bbded08865e'),
    'run_cradiov4.ps1':(
        'cf7d23b5dae0e62e76ec065f9ed2172c48baeeb0a07ee2c8c5d625140c1352bb',
        '8f24554b8b1a8f0a26e35277f597277243a50ce018b795b0ae29b192e1f75f61')}


def verify_patch(oldplan, current):
    if e.digest(oldplan['frozen'])!=OLD_PLAN:
        raise ValueError('This recovery only supports the known original protocol')
    expected=copy.deepcopy(oldplan['frozen'])
    for name,(before,after) in PATCHED_SOURCES.items():
        ref=expected['sources'][name]
        if ref['sha256']!=before or current['sources'][name]['sha256']!=after:
            raise ValueError('Unreviewed source revision: '+name)
        ref['sha256']=after
    if current!=expected:
        raise ValueError('Something besides the two audited source fixes changed; recovery refused')


def write_once_json(path,value):
    if path.exists():
        if e.read_json(path)!=value:raise ValueError('Existing recovery file differs: '+str(path))
    else:e.write_json(path,value)


def recover(source=ROOT/'reports/cradiov4',target=ROOT/'reports/cradiov4_importfix'):
    source,target=source.resolve(),target.resolve()
    reports=(ROOT/'reports').resolve()
    if (not source.is_relative_to(reports) or not target.is_relative_to(reports)
            or source==target or source.is_relative_to(target) or target.is_relative_to(source)):
        raise ValueError('Use distinct sibling report directories inside the workspace')
    oldplan=e.read_json(source/'protocol.json')
    check=experiment.read_check(source,oldplan)
    # Validates original data, inner baselines, all source inputs, submissions and
    # environment through unchanged upstream validators before accepting caches.
    data,_,_,current=experiment.load_state()
    verify_patch(oldplan,current)
    if any((source/f'seed{s}').exists() for s in experiment.SEEDS):
        raise ValueError('Recovery is for interrupted extraction only, not completed evaluations')
    if target.exists() and not (target/'recovery_intent.json').exists():
        raise ValueError('Target already exists without this recovery intent')
    parts=[];origins={};scope=experiment.feature_binding(oldplan,check)
    names={f'{i:04d}.npz' for i in range(0,len(data.ids),16)}
    if {p.name for p in (source/'features/chunks').glob('*.npz')}!=names:
        raise ValueError('All32 original chunks must be present, with no unexpected chunks')
    for start in range(0,len(data.ids),16):
        path=source/f'features/chunks/{start:04d}.npz'
        x,_=e.npz_read(path,data.ids[start:start+16],experiment.eng.WIDTH,scope)
        parts.append(x);origins[str(path.relative_to(source))]=e.file_ref(path)
    values=np.concatenate(parts)
    if values.shape!=(500,4608):raise ValueError('Unexpected recovered matrix')
    origin={'protocol':e.file_ref(source/'protocol.json'),
            'synthetic_check':e.file_ref(source/'synthetic_check.json'),'chunks':origins}
    intent={'old_plan_sha256':OLD_PLAN,'new_plan_sha256':e.digest(current),
            'source':str(source),'target':str(target),'origins':origin,
            'sources_patch':{k:list(v) for k,v in PATCHED_SOURCES.items()},
            'feature_array_sha256':e.base.sha256_array(values),'ids_sha256':e.base.sha256_array(data.ids),
            'recovery_script':e.file_ref(__file__)}
    write_once_json(target/'recovery_intent.json',intent)
    newplan={'frozen':current,'created_unix':oldplan['created_unix'],
             'competition_fits':0,'competition_forwards':0,
             'recovered_from':{'plan':OLD_PLAN,'path':str(source)}}
    write_once_json(target/'protocol.json',newplan)
    saved_check=e.read_json(source/'synthetic_check.json')
    saved_check['binding']=experiment.bind(newplan)
    write_once_json(target/'synthetic_check.json',saved_check)
    path=target/'features/features.npz'
    if not path.exists():
        e.npz_save(path,data.ids,values,experiment.feature_binding(newplan,check),
            {'runtime':check['runtime'],'size':check['size'],'width':experiment.eng.WIDTH})
    loaded,_=experiment.load_features(target,data,newplan)
    np.testing.assert_array_equal(loaded,values)
    # Ensure all original artifacts remained byte-identical throughout recovery.
    for ref in [origin['protocol'],origin['synthetic_check'],*origins.values()]:
        if e.base.sha256_file(ref['path'])!=ref['sha256']:
            raise ValueError('Original artifact changed during recovery')
    e.write_json(target/'recovery_complete.json',{'intent_sha256':e.digest(intent),
        'features':e.file_ref(path),'shape':list(loaded.shape),'array_identical':True,
        'original_artifacts_unchanged':True,'new_model_forwards':0,'new_regression_fits':0,
        'verified_unix':time.time()})
    print(f'Recovered {loaded.shape}: {path}')
    print('Original32 chunks preserved, arrays identical. No model forward, regression fit or submission change.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'reports/cradiov4')
    p.add_argument('--output-dir',type=Path,default=ROOT/'reports/cradiov4_importfix')
    a=p.parse_args();recover(a.source,a.output_dir)
