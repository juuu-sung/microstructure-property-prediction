"""Audit/resume existing Mitra cells, adding RAM waits without rebinding old results."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import mitra_v2_experiment as exp
import mitra_v2_memory_guard as guard

ROOT=exp.ROOT
RECOVERY=ROOT/'reports/mitra_v2_memory_recovery'
ADDENDUM=RECOVERY/'runtime_addendum.json'


def forbidden():raise RuntimeError('Read-only audit must not train')


def expected_addendum():
    files=[Path(__file__),ROOT/'scripts/mitra_v2_memory_guard.py',
           ROOT/'scripts/mitra_v2_resume_worker.py',ROOT/'scripts/resume_mitra_v2.ps1']
    return {'version':1,'policy':guard.POLICY,'sources':{p.name:exp.e.file_ref(p) for p in files},
        'original_protocol':exp.e.file_ref(exp.OUT/'protocol.json'),
        'purpose':'Wait before worker startup and child RAM guards; preserve original2.0 AG ratio and all numerical settings.',
        'compatibility':'Old cells retain their original bindings. New worker results include this runtime addendum. Original code/checks/cache hashes are not rewritten.',
        'incomplete_child':'The incomplete outer cell restarts all8children; a lone saved child has no complete worker result/RNG provenance.'}


def addendum(create=False):
    expected=expected_addendum()
    if ADDENDUM.exists():
        if exp.e.read_json(ADDENDUM)!=expected:raise ValueError('Recovery addendum changed')
    elif create:exp.e.write_json(ADDENDUM,expected)
    else:raise FileNotFoundError('Recovery addendum missing')
    return expected


def audit(state):
    data,x,refs,plan=state
    completed=[];pending=[];tabular=[];files={}
    for seed in exp.SEEDS:
        binding={'plan':exp.e.digest(plan['frozen']),'seed':seed}
        ref=refs[seed]
        for fold in range(5):
            sub=exp.OUT/f'seed{seed}/fold{fold}'
            ti=np.flatnonzero(ref['folds']!=fold);vi=np.flatnonzero(ref['folds']==fold)
            fit_seed=seed+1009*fold+9
            scope={**binding,'fold':fold,'train_indices':ti.tolist(),'query_indices':vi.tolist(),'fit_seed':fit_seed}
            if not (sub/'projection.json').exists():
                pending.append(f'seed{seed}/fold{fold}');continue
            exp.compress_cell(sub,scope,x[ti],data.y[ti],x[vi],False)
            bound={**scope,'projection':exp.e.file_ref(sub/'projection.npz'),
                'metadata':exp.e.file_ref(sub/'projection.json')}
            for name in ('projection.npz','projection.json'):
                files[str(sub/name)]=exp.assets.sha(sub/name)
            if (sub/'tabpfn256.json').exists():
                tr=exp.e.cell(sub/'tabpfn256.json',bound,forbidden,False)
                exp.eng.finite(tr['predictions'],(100,))
                if tr['seed']!=fit_seed or tr['input_sha256']!=exp.assets.sha(sub/'projection.npz'):
                    raise ValueError('TabPFN cell provenance differs')
                if exp.e.read_json(sub/'tabpfn_result.json')!=tr:raise ValueError('TabPFN raw result differs')
                tabular.append(f'seed{seed}/fold{fold}')
                for name in ('tabpfn256.json','tabpfn_result.json'):files[str(sub/name)]=exp.assets.sha(sub/name)
            if not (sub/'mitra256.json').exists():
                pending.append(f'seed{seed}/fold{fold}');continue
            mr=exp.e.cell(sub/'mitra256.json',bound,forbidden,False)
            exp.worker.validate_audits(mr['audit'],50,mr['bag_members'])
            if (mr['synthetic'] or mr['steps_tested']!=50 or mr['recipe']!=exp.worker.RECIPE
                or mr['input_sha256']!=exp.assets.sha(sub/'projection.npz') or mr['seed']!=fit_seed
                or mr['environment']!=plan['frozen']['mitra_environment']
                or mr['worker_sha256']!=exp.assets.sha(exp.worker.__file__) or mr['assets']!=exp.assets.spec()):
                raise ValueError('Mitra cell provenance differs')
            if 'recovery_runtime' in mr:
                if mr['recovery_runtime']['addendum_sha256']!=exp.assets.sha(ADDENDUM):
                    raise ValueError('Recovered result runtime addendum differs')
            exp.eng.finite(mr['predictions'],(100,))
            if exp.e.read_json(sub/'worker_result.json')!=mr:raise ValueError('Mitra raw result differs')
            completed.append(f'seed{seed}/fold{fold}')
            for name in ('mitra256.json','worker_result.json'):files[str(sub/name)]=exp.assets.sha(sub/name)
    return {'completed_mitra':completed,'completed_tabpfn':tabular,'pending_mitra':pending,
            'preserved_files':files,'competition_fits_executed':0,'competition_predictions_executed':0}


def prepare_recovery():
    state=exp.validate(exp.OUT)
    addendum(create=True)
    result=audit(state)
    # Each inspection is its own receipt, never relabel an old cache.
    path=RECOVERY/f'audit_{time.time_ns()}.json'
    exp.e.write_json(path,result)
    print('Verified reusable Mitra cells: '+str(result['completed_mitra']),flush=True)
    print('Verified reusable TabPFN cells: '+str(result['completed_tabpfn']),flush=True)
    print('Pending Mitra cells: '+str(result['pending_mitra']),flush=True)
    return state,result


def run_worker(input_path,output_path,seed):
    addendum()
    log=output_path.with_suffix('.log')
    if log.exists():
        old=RECOVERY/'previous_logs'/f'{output_path.parent.parent.name}_{output_path.parent.name}_{exp.assets.sha(log)[:16]}.log'
        old.parent.mkdir(parents=True,exist_ok=True)
        if not old.exists():shutil.copy2(log,old)
    print(f'Mitra RAM-aware run; progress log: {log}',flush=True)
    env=dict(os.environ);env['PYTHONIOENCODING']='utf-8'
    args=[str(exp.PYTHON),str(ROOT/'scripts/mitra_v2_resume_worker.py'),
          '--input',str(input_path),'--output',str(output_path),'--seed',str(seed),'--addendum',str(ADDENDUM)]
    children=0
    with log.open('w',encoding='utf-8') as f:
        with subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1) as process:
            for line in process.stdout:
                f.write(line);f.flush()
                if line.startswith('[RAM '):print(line.rstrip(),flush=True)
                elif line.startswith('[experiment-audit] '):
                    item=json.loads(line.split('] ',1)[1])
                    if item.get('completed'):
                        children+=1
                        print(f'Mitra child{children}/8 completed: {item["optimizer_steps"]}steps, {item["seconds"]:.0f}s',flush=True)
            returncode=process.wait()
    if returncode:
        tail='\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-25:])
        raise RuntimeError(f'Mitra stopped; completed outer cells preserved. See{log}\n{tail}')
    result=exp.e.read_json(output_path)
    if result['recovery_runtime']['addendum_sha256']!=exp.assets.sha(ADDENDUM):
        raise ValueError('Recovery output provenance differs')
    return result


def memory_status():
    code="import json,psutil; m=psutil.virtual_memory(); print(json.dumps({'available_bytes':m.available,'total_bytes':m.total}))"
    r=subprocess.run([str(exp.PYTHON),'-c',code],capture_output=True,text=True,check=True)
    m=json.loads(r.stdout)
    print(f'System RAM available={m["available_bytes"]/guard.GIB:.2f} GiB; '
          f'new Mitra worker waits for{guard.POLICY["startup_available_gib"]:.1f} GiB. '
          'Close memory-heavy applications before resuming.',flush=True)
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('check','run'))
    a=p.parse_args()
    state,before=prepare_recovery();memory_status()
    if a.command=='check':
        print('Read-only recovery check complete; no training or prediction.');return
    exp.run_worker=run_worker
    for seed in exp.SEEDS:
        exp.evaluate(exp.OUT,seed,True,state)
        # Catch mid-run changes before moving on to the next seed.
        state=exp.validate(exp.OUT);addendum()
    exp.compare(exp.OUT)
    for path,digest in before['preserved_files'].items():
        if exp.assets.sha(path)!=digest:raise ValueError('Previously completed file changed: '+path)
    print('Recovery complete; existing cells verified unchanged.')


if __name__=='__main__':main()
