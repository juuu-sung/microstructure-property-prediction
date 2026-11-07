"""Fixed raw486 Mitra-v2 versus archived TabPFN486; user-run competition fits."""
from __future__ import annotations
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import mitra_v2_experiment as old
import mitra_v2_raw486_worker as worker
import mitra_v2_raw486_runtime as memory
import mitra_v2_memory_guard as guard

e, core, diagnostics, assets = old.e, old.core, old.diagnostics, old.assets
ROOT, PYTHON, SEEDS = old.ROOT, old.PYTHON, old.SEEDS
OUT = ROOT/'reports/mitra_v2_raw486'
CHECK = ROOT/'reports/mitra_v2_raw486_runtime/synthetic_check.json'
MATH = CHECK.with_name('math_check.json')
MATH_SCRIPT = ROOT/'scripts/check_mitra_raw486_math.py'
PRIMARY = 'mitra_raw486_replaced_blend'
PROTOCOL = {
    'version': 1, 'seeds': list(SEEDS), 'primary': PRIMARY,
    'input': 'Identical existing raw486 columns/values. No PCA, SVD, feature selection, external standardization or image changes. Identity AutoGluon feature generator. Official Mitra bag-local preprocessing (including constant-column handling) retained.',
    'control': 'Archived, validated TabPFN486 OOF from identical outer splits. No TabPFN refit. Current = .25B+.25TabPFN486+.25Large518+.25C_RADIO; B=.25hand185+.25ResNet+.5DINO_D4.',
    'candidate': '.25B+.25Mitra_raw486+.25Large518+.25C_RADIO. Replace only TabPFN25%.',
    'mitra': worker.RECIPE,
    'memory': memory.POLICY, 'ram_wait': guard.POLICY,
    'validation': 'seed42/2026;5 fixed outer400/100.8 sequential internal350/50 bags,50 optimizer steps each; checkpoint selected internally. Official all400 support at outer prediction. Query labels never sent to worker.',
    'selection': 'No new hyperparameter/mixture/model/dimension search. Seed=seed+1009*fold+9. Full steps and8members required; no context/time/bag salvage.',
    'check': 'Synthetic400x486,query100x486,8bags x2steps plus full prediction; no competition fitting. Memory-only changes independently checked for output/gradient agreement on a small synthetic network.',
    'diagnostics': 'Standalone,common16,remaining484,low10/high10/middle80,all noise quartiles,all folds,newworst20,signs,residual correlation.',
    'priority': 'Both seeds must improve current; mean OOF RMSE gain>=0.30 prioritizes expansion, not automatic adoption. Review common16 and remaining484/noise separately.',
    'scope': 'User runs80 child models. Train500 only. No test/final500/submission changes. Existing SVD256 results preserved.',
    'limits': 'Same repeatedly studied500 and reused folds, not independent holdout. Synthetic2step feasibility does not guarantee50step completion or improvement.',
    'sources': ['https://huggingface.co/autogluon/mitra-finetune', 'https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu'],
}


def validate_worker(r, *, synthetic, input_path=None, seed=None, environment=None):
    worker.validate_audits(r['audit'], 2 if synthetic else 50, r['bag_members'])
    for a in r['audit']:
        if (a['configured_epochs'] != (2 if synthetic else 50) or a['warmup_steps']!=10
            or a['weight_decay']!=.3 or a['max_samples_support_before']!=20480):
            raise ValueError('Observed training schedule differs from fixed recipe')
    if (r['synthetic'] != synthetic or r['steps_tested'] != (2 if synthetic else 50)
        or r['recipe'] != worker.RECIPE or r['worker_sha256'] != assets.sha(worker.__file__)
        or r['assets'] != assets.spec() or not r['raw486_retained']
        or r['memory_runtime'] != {'policy':memory.POLICY,'sha256':assets.sha(memory.__file__)}
        or r['memory_guard']['policy'] != guard.POLICY
        or r['math_check_sha256'] != assets.sha(MATH)):
        raise ValueError('Raw486 worker provenance/recipe changed')
    if r['peak_allocated_bytes'] > r['total_vram']:
        raise ValueError('Physical GPU capacity exceeded')
    if environment is not None and r['environment'] != environment:
        raise ValueError('Mitra environment changed')
    if seed is not None and r['seed'] != seed:
        raise ValueError('Worker seed changed')
    if r['competition_fits'] != (0 if synthetic else 8) or r['competition_queries'] != (0 if synthetic else 100):
        raise ValueError('Worker task scope changed')
    if r['input_sha256'] != (None if synthetic else assets.sha(input_path)):
        raise ValueError('Worker input hash changed')
    return old.eng.finite(r['predictions'], (100,))


def run_worker(input_path, output_path, seed, synthetic=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log = output_path.with_suffix('.log')
    if log.exists():
        shutil.copy2(log, log.with_name(log.stem+f'.previous_{time.time_ns()}.log'))
    args = [str(PYTHON), str(Path(worker.__file__)), 'check' if synthetic else 'fit',
            '--output', str(output_path), '--seed', str(seed)]
    if not synthetic: args += ['--input',str(input_path)]
    print(f'Raw486 Mitra: {"synthetic check" if synthetic else "8bags x50steps"}; log={log}', flush=True)
    env = dict(os.environ); env['PYTHONIOENCODING'] = 'utf-8'
    children = 0
    with log.open('w', encoding='utf-8') as f:
        with subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            for line in process.stdout:
                f.write(line); f.flush()
                if line.startswith(('[RAM ','[raw486 step]')): print(line.rstrip(), flush=True)
                elif line.startswith('[experiment-audit] '):
                    a = json.loads(line.split('] ',1)[1])
                    if a.get('completed'):
                        children += 1
                        print(f'Child{children}/8: {a["optimizer_steps"]}steps, {a["seconds"]:.0f}s, width={a["width"]}',flush=True)
            code = process.wait()
    if code:
        tail = '\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-20:])
        raise RuntimeError(f'Raw486 stopped; completed outer cells preserved. See {log}\n{tail}')
    r = e.read_json(output_path)
    validate_worker(r,synthetic=synthetic,input_path=input_path,seed=seed)
    return r


def check():
    assets.validate()
    if not MATH.exists():
        subprocess.run([str(PYTHON),str(MATH_SCRIPT)],cwd=ROOT,check=True)
    validate_math()
    if CHECK.exists():
        r = e.read_json(CHECK)
        validate_worker(r,synthetic=True,seed=314159,environment=old.worker_env())
        print('Existing raw486 synthetic capacity check verified; no rerun.')
    else:
        r = run_worker(None,CHECK,314159,True)
    print(f'Check passed: {r["gpu"]}, peak allocated={r["peak_allocated_bytes"]/2**30:.3f} GiB, {r["seconds"]:.1f}s; no competition fits.')


def validate_math():
    r=e.read_json(MATH)
    if (not r['passed'] or r['competition_fits']!=0 or r['policy']!=memory.POLICY
        or r['runtime_sha256']!=assets.sha(memory.__file__) or r['test_sha256']!=assets.sha(MATH_SCRIPT)):
        raise ValueError('Small-network output/gradient check stale')


def load_state():
    data,x,refs,upstream = old.load_state()
    validate_math()
    r = e.read_json(CHECK)
    validate_worker(r,synthetic=True,seed=314159,environment=upstream['mitra_environment'])
    sources = [Path(__file__),Path(worker.__file__),Path(memory.__file__),Path(guard.__file__),
               ROOT/'scripts/run_mitra_v2_raw486.ps1',MATH_SCRIPT]
    frozen = {'protocol':PROTOCOL,'upstream':upstream,'common_failure_ids':upstream['common_failure_ids'],
              'source_files':{p.name:e.file_ref(p) for p in sources},'synthetic_check':e.file_ref(CHECK),
              'math_check':e.file_ref(MATH)}
    return data,x,refs,frozen


def prepare(folder=OUT):
    data,_,refs,frozen = load_state()
    path = folder/'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen'] != frozen: raise ValueError('Prepared raw486 experiment changed; preserve old output')
    else:
        if folder.exists() and any(folder.iterdir()): raise ValueError('Nonempty unprepared raw486 output')
        e.write_json(path,{'frozen':frozen,'created_unix':time.time(),'competition_fits':0})
    for seed in SEEDS:
        print(f'Prepared seed{seed}; current={core.metrics(data.y,refs[seed]["current"])["rmse"]:.6f}; raw486 passed unchanged; no competition fits.')


def validate(folder=OUT):
    data,x,refs,frozen = load_state(); plan=e.read_json(folder/'protocol.json')
    if plan['frozen'] != frozen: raise ValueError('Frozen source/data/runtime changed; no reuse')
    return data,x,refs,plan


def raw_cell(folder,binding,x,y,q,allow_fit):
    # Persist exact float64 source values; no transform or query labels.
    x,y,q=[old.eng.finite(a,shape) for a,shape in ((x,(400,486)),(y,(400,)),(q,(100,486)))]
    path=folder/'raw486.npz'; meta=folder/'raw486.json'
    scope={**binding,'x':e.base.sha256_array(x),'y':e.base.sha256_array(y),'q':e.base.sha256_array(q)}
    if path.exists() and meta.exists():
        m=e.read_json(meta)
        if m['binding']!=scope or m['file']!=e.file_ref(path): raise ValueError('Raw486 payload changed')
        with np.load(path,allow_pickle=False) as a:
            if set(a.files)!={'train_x','train_y','query_x'}: raise ValueError('Unexpected payload schema')
            for name,v in (('train_x',x),('train_y',y),('query_x',q)): np.testing.assert_array_equal(a[name],v)
        return m
    if not allow_fit: raise FileNotFoundError('Missing raw486 payload; implicit work forbidden')
    folder.mkdir(parents=True,exist_ok=True)
    temp=folder/'raw486.pending.npz'; np.savez_compressed(temp,train_x=x,train_y=y,query_x=q); temp.replace(path)
    m={'binding':scope,'file':e.file_ref(path),'transformation':'none','width':486}
    e.write_json(meta,m); return m


def collect(folder,data,x,ref,binding,seed,allow_fit,environment):
    predictions=np.full(500,np.nan); audits=[]
    for fold in range(5):
        ti=np.flatnonzero(ref['folds']!=fold); vi=np.flatnonzero(ref['folds']==fold)
        if (len(ti),len(vi))!=(400,100): raise ValueError('Unexpected fold size')
        sub=folder/f'fold{fold}'; fit_seed=seed+1009*fold+9
        scope={**binding,'fold':fold,'train_indices':ti.tolist(),'query_indices':vi.tolist(),'fit_seed':fit_seed}
        raw_cell(sub,scope,x[ti],data.y[ti],x[vi],allow_fit)
        inp=sub/'raw486.npz'; out=sub/'worker_result.json'
        bound={**scope,'payload':e.file_ref(inp),'metadata':e.file_ref(sub/'raw486.json')}
        r=e.cell(sub/'mitra486.json',bound,lambda:run_worker(inp,out,fit_seed),allow_fit)
        predictions[vi]=validate_worker(r,synthetic=False,input_path=inp,seed=fit_seed,environment=environment)
        if e.read_json(out)!=r: raise ValueError('Raw worker result differs from bound cache')
        audits.append({'fold':fold,'audit':r['audit'],'seconds':r['seconds'],'peak_allocated_bytes':r['peak_allocated_bytes']})
        print(f'seed{seed} outer{fold}: raw486,8bags x50steps verified; outer labels unused',flush=True)
    current=.25*(ref['base']+ref['large']+ref['radio']+ref['tab486'])
    np.testing.assert_allclose(current,ref['current'],rtol=0,atol=1e-10)
    return {'current':ref['current'],'tabpfn486':ref['tab486'],'mitra_raw486':predictions,
            PRIMARY:ref['current']+.25*(predictions-ref['tab486'])},audits


def results(data,ref,p,audits,plan,seed,binding):
    mapped={diagnostics.PRIMARY if n==PRIMARY else n:v for n,v in p.items()}
    s,files=diagnostics.result_data(data,ref,mapped,audits,plan,seed)
    for key in ('metrics','diagnosis'): s[key][PRIMARY]=s[key].pop(diagnostics.PRIMARY)
    for rows in files.values():
        for row in rows:
            for k in list(row):
                new=k.replace(diagnostics.PRIMARY,PRIMARY)
                if k!=new: row[new]=row.pop(k)
            if row.get('model')==diagnostics.PRIMARY: row['model']=PRIMARY
    s.update(binding=binding,primary=PRIMARY,selection=audits,tabpfn486_anchor_matched=True)
    s['metrics']['mitra_raw486']['gain_vs_tabpfn486']=s['metrics']['tabpfn486']['rmse']-s['metrics']['mitra_raw486']['rmse']
    return s,files


def report(s):
    lines=[f'Uncompressed486 Mitra-v2 seed{s["seed"]}; replace only current TabPFN25%. Gain>0 means improvement.']
    for n,m in s['metrics'].items(): lines.append(f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}')
    lines.append(f'Standalone Mitra gain vs TabPFN486={s["metrics"]["mitra_raw486"]["gain_vs_tabpfn486"]:+.6f}')
    for g in ('common16','remaining484','noiseQ1','noiseQ2','noiseQ3','noiseQ4','low10','high10','middle80'):
        a=s['diagnosis']['current']['groups'][g]; b=s['diagnosis'][PRIMARY]['groups'][g]
        lines.append(f'{g}: {a["rmse"]:.6f} -> {b["rmse"]:.6f}; gain={a["rmse"]-b["rmse"]:+.6f}')
    d=s['diagnosis'][PRIMARY]
    lines.extend([f'Common16 improved={d["common16_improved"]}/16; in worst20={d["common16_in_worst20"]}/16; same sign={d["common16_same_error_sign"]}/16.',
                  f'New worst20={d["new_worst20_ids"]}',PROTOCOL['priority'],PROTOCOL['limits']])
    return '\n'.join(lines)+'\n'


def evaluate(folder=OUT,seed=42,allow_fit=True,state=None):
    data,x,refs,plan=state or validate(folder);sub=folder/f'seed{seed}'
    binding={'plan':e.digest(plan['frozen']),'seed':seed};complete=sub/'completed_evaluation.json'
    if complete.exists():
        receipt=e.read_json(complete)
        if receipt['binding']!=binding: raise ValueError('Completion scope changed')
        for name,h in receipt['files'].items():
            if assets.sha(sub/name)!=h: raise ValueError('Completed file changed: '+name)
    p,audits=collect(sub,data,x,refs[seed],binding,seed,allow_fit and not complete.exists(),plan['frozen']['upstream']['mitra_environment'])
    s,files=results(data,refs[seed],p,audits,plan,seed,binding)
    if complete.exists():
        if e.read_json(sub/'summary.json')!=s or (sub/'report.txt').read_text(encoding='utf-8')!=report(s): raise ValueError('Reconstructed report differs')
        for name,rows in files.items():
            expected=list(csv.DictReader(io.StringIO(core.csv_text(rows))))
            with (sub/name).open(encoding='utf-8',newline='') as f:
                if list(csv.DictReader(f))!=expected: raise ValueError('Reconstructed CSV differs: '+name)
        print(f'Completed raw486 seed{seed} verified; no refitting.');return s
    if not allow_fit: raise FileNotFoundError('Incomplete; no implicit fitting')
    for name,rows in files.items(): e.write_csv(sub/name,rows)
    e.write_json(sub/'summary.json',s);(sub/'report.txt').write_text(report(s),encoding='utf-8')
    names=list(files)+['summary.json','report.txt']+[f'fold{f}/{n}' for f in range(5) for n in ('raw486.npz','raw486.json','mitra486.json','worker_result.json')]
    e.write_json(complete,{'binding':binding,'files':{n:assets.sha(sub/n) for n in names}})
    print(report(s)); print('Report: '+str(sub/'report.txt'));return s


def compare(folder=OUT):
    state=validate(folder);summaries=[evaluate(folder,s,False,state) for s in SEEDS]
    rows=[{'seed':s['seed'],'current_rmse':s['metrics']['current']['rmse'],
           'candidate_rmse':s['metrics'][PRIMARY]['rmse'],'gain_vs_current':s['metrics'][PRIMARY]['gain_vs_current'],
           'standalone_gain_vs_tabpfn486':s['metrics']['mitra_raw486']['gain_vs_tabpfn486'],
           'diagnosis':s['diagnosis'][PRIMARY]} for s in summaries]
    mean=float(np.mean([r['gain_vs_current'] for r in rows]));good=all(r['gain_vs_current']>0 for r in rows)
    result={'primary':PRIMARY,'per_seed':rows,'mean_gain_vs_current':mean,'all_seeds_improve':good,
            'passes_expansion_priority':good and mean>=.3,'automatic_adoption':False,'independent_holdout':False,
            'additional_review_required':PROTOCOL['diagnostics'],'protocol_sha256':e.digest(state[3]['frozen']),
            'aggregation':'Mean per-seed OOF RMSE differences; same500, no prediction averaging or independent inference.'}
    e.write_json(folder/'comparison.json',result); print(e.base.canonical_json(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('check','prepare','evaluate','compare'))
    p.add_argument('--seed',type=int,choices=SEEDS,default=42);a=p.parse_args()
    if a.command=='check':check()
    elif a.command=='prepare':prepare()
    elif a.command=='evaluate':evaluate(seed=a.seed)
    else:compare()


if __name__=='__main__':main()
