"""User-run Mitra-v2 versus raw486/SVD256 TabPFN; replace only current25%."""
from __future__ import annotations
import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

if __package__:
    from . import cradiov4_experiment as prior
    from . import mitra_v2_engine as eng
    from . import mitra_v2_worker as worker
else:
    import cradiov4_experiment as prior
    import mitra_v2_engine as eng
    import mitra_v2_worker as worker

e, core, diagnostics = prior.e, prior.core, prior.diagnostics
residual, assets = prior.residual, worker.assets
t = residual.t
ROOT = e.ROOT
SOURCE = ROOT/'reports/cradiov4_importfix'
OUT = ROOT/'reports/mitra_v2'
RUNTIME = ROOT/'reports/mitra_v2_runtime'
PYTHON = ROOT/'.venv-mitra-v2/Scripts/python.exe'
SEEDS = (42,2026)
PRIMARY = 'mitra_replaced_blend'
CONTROL = 'tabpfn_svd256_control_blend'
PROTOCOL = {
    'version':1,'seeds':list(SEEDS),'primary':PRIMARY,
    'current':'.25B+.25TabPFN486+.25Large518+.25C_RADIO; B=.25hand185+.25ResNet+.5DINO_D4. Equals submitted S:L:C=50:25:25.',
    'candidate':'.25B+.25Mitra_SVD256+.25Large518+.25C_RADIO; replace ONLY TabPFN25%.',
    'controls':'Verified archived raw486 TabPFN=current; new SVD256 TabPFN with identical original V2 settings and seeds. No raw486 refit.',
    'preprocessing':'Per outer400 only: mean imputation -> StandardScaler -> centered PCA256, full SVD, no whitening, random_state0. Both models receive identical arrays. Official Mitra calls this SVD. No query fit, y-based compression or dimension search.',
    'mitra':worker.RECIPE,
    'bagging':'8 sequential internal350/50 splits of outer400. Checkpoint selection uses only internal50. Each child predicts outer100 with all400 support after selection, official heldout-in-support default. No outer labels passed. Full50 optimizer steps and8 output members required; no silent time/OOM salvage.',
    'internal_preprocessing_limit':'Shared compression fits outer400, including each bag internal validation features but no labels; this matches official pre-bag compression. Internal bag metrics are checkpoint selectors, not unbiased performance estimates. Outer100 excluded entirely from fitted preprocessing and labels.',
    'tabpfn':'Existing isolated main env, official V2 local checkpoint,8estimators,FP32, originalPredictor. No weight update. Seed=seed+1009*fold+9.',
    'mitra_seed':'seed+1009*fold+9; one fresh process per outer cell. Official8bag seeds. BF16/SDPA,no TF32.',
    'selection':'No additional model/lr/step/mixture search. Outer5folds fixed42/2026,100each. No nested3fold search needed for fixed candidates;8bag validation selects checkpoints internally.',
    'diagnostics':'Standalone,common16,remaining484,low10/high10/middle80,noiseQ1..4,folds,newworst20,signs,correlation. Both compression and model comparisons.',
    'priority':'Both seeds beat current and SVD256 control; mean gain>=0.30 prioritizes expansion only. Group tradeoffs reviewed separately; no automatic adoption.',
    'scope':'User runs10 PCA fits,10 TabPFN contexts and80 Mitra child fits. Train500 only; no test/final500/submission/image encoder work. Existing submissions preserved.',
    'limits':'Repeatedly studied500/reused splits, not independent holdout. Synthetic2step test checks geometry/Adam memory, not50step numerical convergence or competition improvement.',
    'sources':['https://huggingface.co/autogluon/mitra-regressor-2','https://huggingface.co/autogluon/mitra-finetune'],
    'attribution':'AutoGluon Mitra-v2/official mitra-finetune, Apache2.0; controls built with PriorLabs-TabPFN.'}


def worker_env():
    r = subprocess.run([str(PYTHON),str(ROOT/'scripts/mitra_v2_worker.py'),'environment'],
        check=True, capture_output=True, text=True, encoding='utf-8', cwd=ROOT)
    return json.loads(r.stdout)


def load_state():
    assets.validate()
    with contextlib.redirect_stdout(io.StringIO()):
        data,_,old,cp = prior.validate(SOURCE)
        rd,raw,rr,rp = residual.validate()
        for seed in SEEDS:
            prior.evaluate(SOURCE,seed,False)
    np.testing.assert_array_equal(rd.ids,data.ids)
    np.testing.assert_array_equal(rd.y,data.y)
    x = eng.finite(raw['tabpfn486'], (500,486))
    refs,files = {},{}
    for seed in SEEDS:
        sub = SOURCE/f'seed{seed}'
        with (sub/'oof.csv').open(encoding='utf-8',newline='') as f:
            rows = list(csv.DictReader(f))
        if [r['ID'] for r in rows] != data.ids.tolist():raise ValueError('OOF ID mismatch')
        np.testing.assert_array_equal([float(r['hardness']) for r in rows],data.y)
        np.testing.assert_array_equal([int(r['fold']) for r in rows],rr[seed]['folds'])
        comp = rr[seed]['components']
        base = .25*comp[:,0]+.25*comp[:,1]+.5*comp[:,2]
        tab = comp[:,3]
        np.testing.assert_allclose(.5*base+.5*tab,old[seed]['scattering'],rtol=0,atol=1e-10)
        radio = eng.finite([float(r['radio_ridge']) for r in rows],(500,))
        current = eng.finite([float(r['radio_complementary_blend']) for r in rows],(500,))
        refs[seed] = {'folds':rr[seed]['folds'],'base':base,'tab486':tab,'large':old[seed]['large'],
                     'radio':radio,'current':current}
        np.testing.assert_allclose(eng.compose(base,old[seed]['large'],radio,tab,tab,tab)['current'],current,rtol=0,atol=1e-10)
        files[str(seed)] = {n:e.file_ref(sub/n) for n in ('completed_evaluation.json','oof.csv','summary.json')}
    sources = [Path(__file__),Path(eng.__file__),Path(worker.__file__),Path(assets.__file__),assets.LOCK,
        ROOT/'scripts/run_mitra_v2.ps1',ROOT/'scripts/setup_mitra_v2.ps1',ROOT/'scripts/mitra_v2_requirements.lock.txt',
        ROOT/'scripts/mitra_v2_tabpfn_worker.py',
        Path(prior.__file__),Path(residual.__file__),Path(diagnostics.__file__)]
    frozen = {'protocol':PROTOCOL,'assets':assets.spec(),'source_files':{p.name:e.file_ref(p) for p in sources},
        'main_environment':e.environment(),'mitra_environment':worker_env(),
        'upstream':{str(p):e.file_ref(p) for p in (SOURCE/'protocol.json',residual.OUT/'protocol.json')},
        'reference_files':files,'training_inventory':cp['frozen']['training_inventory'],
        'common_failure_ids':cp['frozen']['common_failure_ids'],
        'arrays':{n:e.base.sha256_array(v) for n,v in {'x486':x,'y':data.y,'ids':data.ids,'noise':data.noise}.items()},
        'tabpfn_weights':e.file_ref(t.verify_weights(t.OUT)),
        'submissions':{n:e.file_ref(ROOT/f'reports/{n}/submission/submission.csv') for n in ('final_scattering_blend','final_large518_blend','final_cradiov4_blend')}}
    return data,x,refs,frozen


def checked_runtime(frozen):
    p = RUNTIME/'synthetic_check.json'
    r = e.read_json(p)
    worker.validate_audits(r['audit'],2,r['bag_members'])
    if (not r['synthetic'] or r['competition_fits'] != 0 or r['steps_tested'] != 2
        or r['worker_sha256'] != assets.sha(worker.__file__) or r['assets'] != frozen['assets']
        or r['environment'] != frozen['mitra_environment'] or r['recipe'] != worker.RECIPE):
        raise ValueError('Synthetic check stale or not applicable')
    return e.file_ref(p)


def prepare(folder=OUT):
    data,_,refs,frozen = load_state()
    frozen['synthetic_check'] = checked_runtime(frozen)
    path = folder/'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen'] != frozen:raise ValueError('Prepared experiment changed; preserve old directory')
    else:
        if folder.exists() and any(folder.iterdir()):raise ValueError('Nonempty unprepared output directory')
        e.write_json(path,{'frozen':frozen,'created_unix':time.time(),'competition_fits':0})
    print('Prepared Mitra-v2 / TabPFN486 / TabPFN_SVD256; TabPFN25% replacement only. No competition fits.')
    for seed in SEEDS:print(f'Current seed{seed}: {core.metrics(data.y,refs[seed]["current"])["rmse"]:.6f} anchor validated')


def validate(folder):
    data,x,refs,frozen = load_state()
    frozen['synthetic_check'] = checked_runtime(frozen)
    plan = e.read_json(folder/'protocol.json')
    if plan['frozen'] != frozen:raise ValueError('Frozen code/input/environment changed; no reuse')
    return data,x,refs,plan


def compress_cell(folder,binding,x,y,q,allow_fit):
    path = folder/'projection.npz';meta = folder/'projection.json'
    scope = {**binding,'x':e.base.sha256_array(x),'y':e.base.sha256_array(y),'q':e.base.sha256_array(q)}
    if path.exists() and meta.exists():
        m=e.read_json(meta)
        if m['binding'] != scope or m['file'] != e.file_ref(path):raise ValueError('Projection cache changed')
        with np.load(path,allow_pickle=False) as a:
            if set(a.files) != {'train_x','train_y','query_x'}:raise ValueError('Payload schema changed')
            z,v=a['train_x'],a['query_x'];np.testing.assert_array_equal(a['train_y'],y)
        return eng.finite(z,(400,256)),eng.finite(v,(100,256)),m
    if not allow_fit:raise FileNotFoundError('Projection incomplete; no implicit fitting')
    z,v,info=eng.project(x,q)
    folder.mkdir(parents=True,exist_ok=True)
    tmp=folder/'projection.pending.npz'
    np.savez_compressed(tmp,train_x=z,train_y=y,query_x=v);tmp.replace(path)
    m={'binding':scope,'file':e.file_ref(path),'info':info}
    e.write_json(meta,m)
    return z,v,m


def run_worker(input_path,output_path,seed):
    log=output_path.with_suffix('.log')
    print(f'Mitra8 sequential children x50steps starting; progress log: {log}',flush=True)
    env=dict(os.environ)
    env['PYTHONIOENCODING']='utf-8'
    with log.open('w',encoding='utf-8') as f:
        r=subprocess.run([str(PYTHON),str(ROOT/'scripts/mitra_v2_worker.py'),'fit',
            '--input',str(input_path),'--output',str(output_path),'--seed',str(seed)],
            cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:
        tail='\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-30:])
        raise RuntimeError(f'Mitra worker failed; see {log}\n{tail}')
    return e.read_json(output_path)


def run_tabpfn(input_path,output_path,seed):
    # A separate main-env process exits before Mitra starts, releasing its
    # CUDA context as well as its cached tensors on this8GB GPU.
    log=output_path.with_suffix('.log')
    env=dict(os.environ);env['PYTHONIOENCODING']='utf-8'
    with log.open('w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,str(ROOT/'scripts/mitra_v2_tabpfn_worker.py'),
            '--input',str(input_path),'--output',str(output_path),'--seed',str(seed)],
            cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'TabPFN control failed; see {log}')
    result=e.read_json(output_path)
    if result['seed']!=seed or result['input_sha256']!=assets.sha(input_path):
        raise ValueError('TabPFN worker output scope changed')
    return result


def collect(folder,data,x,ref,binding,seed,allow_fit,env):
    p={n:np.full(500,np.nan) for n in ('tab256','mitra256')}; audits=[]
    for fold in range(5):
        ti=np.flatnonzero(ref['folds']!=fold);vi=np.flatnonzero(ref['folds']==fold)
        if (len(ti),len(vi)) != (400,100):raise ValueError('Unexpected fold size')
        sub=folder/f'fold{fold}';fit_seed=seed+1009*fold+9
        scope={**binding,'fold':fold,'train_indices':ti.tolist(),'query_indices':vi.tolist(),'fit_seed':fit_seed}
        z,q,pm=compress_cell(sub,scope,x[ti],data.y[ti],x[vi],allow_fit)
        bound={**scope,'projection':e.file_ref(sub/'projection.npz'),'metadata':e.file_ref(sub/'projection.json')}
        tr=e.cell(sub/'tabpfn256.json',bound,
            lambda:run_tabpfn(sub/'projection.npz',sub/'tabpfn_result.json',fit_seed),allow_fit)
        p['tab256'][vi]=eng.finite(tr['predictions'],(100,))
        mr=e.cell(sub/'mitra256.json',bound,
            lambda:run_worker(sub/'projection.npz',sub/'worker_result.json',fit_seed),allow_fit)
        worker.validate_audits(mr['audit'],50,mr['bag_members'])
        if (mr['synthetic'] or mr['steps_tested'] != 50 or mr['recipe'] != worker.RECIPE
            or mr['input_sha256'] != assets.sha(sub/'projection.npz') or mr['environment'] != env
            or mr['worker_sha256'] != assets.sha(worker.__file__) or mr['assets'] != assets.spec() or mr['seed'] != fit_seed):
            raise ValueError('Mitra worker output scope changed')
        p['mitra256'][vi]=eng.finite(mr['predictions'],(100,))
        audits.append({'fold':fold,'projection':pm['info'],'mitra_audit':mr['audit'],
            'mitra_seconds':mr['seconds'],'mitra_peak_allocated':mr['peak_allocated_bytes']})
        print(f'seed{seed} outer{fold}: controls/candidate completed,8bags x50steps; outer labels unused',flush=True)
    vectors=eng.compose(ref['base'],ref['large'],ref['radio'],ref['tab486'],p['tab256'],p['mitra256'])
    np.testing.assert_allclose(vectors['current'],ref['current'],rtol=0,atol=1e-10)
    return vectors,audits


def results(data,ref,p,audits,plan,seed,binding):
    mapped={diagnostics.PRIMARY if n==PRIMARY else n:v for n,v in p.items()}
    s,files=diagnostics.result_data(data,ref,mapped,audits,plan,seed)
    for k in ('metrics','diagnosis'):s[k][PRIMARY]=s[k].pop(diagnostics.PRIMARY)
    for rows in files.values():
        for row in rows:
            for k in list(row):
                new=k.replace(diagnostics.PRIMARY,PRIMARY)
                if k!=new:row[new]=row.pop(k)
            if row.get('model')==diagnostics.PRIMARY:row['model']=PRIMARY
    s.update(binding=binding,primary=PRIMARY,selection=audits,tabpfn486_anchor_matched=True)
    for n,m in s['metrics'].items():
        ctrl='tabpfn_svd256' if n in ('tabpfn486','tabpfn_svd256','mitra_svd256') else CONTROL
        m.update(compression_control=ctrl,gain_vs_control=s['metrics'][ctrl]['rmse']-m['rmse'])
    return s,files


def report(s):
    lines=[f'Mitra-v2 seed{s["seed"]}; replace ONLY TabPFN25%. Gain>0 is improvement.']
    for n,m in s['metrics'].items():
        lines.append(f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs SVD256 control={m["gain_vs_control"]:+.6f}')
    for g in ('common16','remaining484','noiseQ1','noiseQ2','noiseQ3','noiseQ4','low10','high10','middle80'):
        a=s['diagnosis']['current']['groups'][g];b=s['diagnosis'][PRIMARY]['groups'][g]
        c=s['diagnosis'][CONTROL]['groups'][g]
        lines.append(f'{g}: current={a["rmse"]:.6f},control={c["rmse"]:.6f},Mitra={b["rmse"]:.6f}; gain vs current={a["rmse"]-b["rmse"]:+.6f}, vs control={c["rmse"]-b["rmse"]:+.6f}')
    d=s['diagnosis'][PRIMARY]
    lines += [f'Common16 improved={d["common16_improved"]}/16; in worst20={d["common16_in_worst20"]}/16; same sign={d["common16_same_error_sign"]}/16.',
        f'New worst20={d["new_worst20_ids"]}',PROTOCOL['priority'],PROTOCOL['limits']]
    return '\n'.join(lines)+'\n'


def evaluate(folder=OUT,seed=42,allow_fit=True,state=None):
    data,x,refs,plan=state or validate(folder)
    sub=folder/f'seed{seed}';binding={'plan':e.digest(plan['frozen']),'seed':seed}
    complete=sub/'completed_evaluation.json'
    done=complete.exists()
    if done:
        d=e.read_json(complete)
        if d['binding'] != binding:raise ValueError('Completion scope changed')
        for n,h in d['files'].items():
            if e.base.sha256_file(sub/n)!=h:raise ValueError('Completed file changed: '+n)
    p,audits=collect(sub,data,x,refs[seed],binding,seed,allow_fit and not done,plan['frozen']['mitra_environment'])
    s,files=results(data,refs[seed],p,audits,plan,seed,binding)
    if done:
        if e.read_json(sub/'summary.json')!=s or (sub/'report.txt').read_text(encoding='utf-8')!=report(s):
            raise ValueError('Reconstructed results differ')
        print(f'Completed seed{seed} verified; no refitting');return s
    if not allow_fit:raise FileNotFoundError('Evaluation incomplete; no implicit training')
    for n,rows in files.items():e.write_csv(sub/n,rows)
    e.write_json(sub/'summary.json',s)
    (sub/'report.txt').write_text(report(s),encoding='utf-8')
    outputs=list(files)+['summary.json','report.txt']
    outputs += [f'fold{f}/{n}' for f in range(5) for n in ('projection.npz','projection.json','tabpfn256.json','tabpfn_result.json','mitra256.json','worker_result.json')]
    e.write_json(complete,{'binding':binding,'files':{n:e.base.sha256_file(sub/n) for n in outputs}})
    print(report(s));print('Report: '+str(sub/'report.txt'));return s


def compare(folder=OUT):
    state=validate(folder)
    summaries=[evaluate(folder,seed,False,state) for seed in SEEDS]
    rows=[]
    for s in summaries:
        m=s['metrics'][PRIMARY]
        rows.append({'seed':s['seed'],'current_rmse':s['metrics']['current']['rmse'],
            'control_rmse':s['metrics'][CONTROL]['rmse'],'candidate_rmse':m['rmse'],
            'gain_vs_current':m['gain_vs_current'],'gain_vs_control':m['gain_vs_control'],
            'diagnosis':s['diagnosis'][PRIMARY]})
    gain=float(np.mean([r['gain_vs_current'] for r in rows]))
    good=all(r['gain_vs_current']>0 and r['gain_vs_control']>0 for r in rows)
    r={'primary':PRIMARY,'per_seed':rows,'mean_gain_vs_current':gain,
        'mean_gain_vs_control':float(np.mean([r['gain_vs_control'] for r in rows])),
        'all_seeds_improve_both':good,'passes_expansion_priority':good and gain>=.3,
        'automatic_adoption':False,'independent_holdout':False,
        'additional_review_required':PROTOCOL['diagnostics'],
        'aggregation':'Mean per-seed OOF RMSE differences; same500, no prediction averaging/independent inference.',
        'protocol_sha256':e.digest(state[3]['frozen'])}
    e.write_json(folder/'comparison.json',r);print(e.base.canonical_json(r));return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','evaluate','compare'))
    p.add_argument('--seed',type=int,choices=SEEDS,default=42)
    p.add_argument('--output-dir',type=Path,default=OUT)
    a=p.parse_args();folder=a.output_dir if a.output_dir.is_absolute() else ROOT/a.output_dir
    if a.command=='prepare':prepare(folder)
    elif a.command=='evaluate':evaluate(folder,a.seed)
    else:compare(folder)


if __name__=='__main__':main()
