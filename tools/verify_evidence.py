"""Validate published aggregate evidence. No private data or ML packages needed.

This checks bookkeeping, not independent leaderboard or raw-OOF accuracy.
"""
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    def reject(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def metric_check(metric):
    for key in ('rmse', 'sse', 'mae'):
        if key in metric and (not math.isfinite(metric[key]) or metric[key] < 0):
            raise ValueError(f'Invalid {key}')
    if {'n', 'sse', 'rmse'} <= metric.keys():
        if metric['n'] <= 0 or not math.isclose(metric['rmse'], math.sqrt(metric['sse']/metric['n']), abs_tol=1e-9):
            raise ValueError('RMSE does not match aggregate SSE / n')


def verify(root=ROOT):
    root = Path(root)
    manifest = read_json(root/'evidence/source_manifest.json')
    paths = [row['path'] for row in manifest['files']]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate source paths')
    for row in manifest['files']:
        path = root/row['path']
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Source path escapes project')
        raw = path.read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError(f'Source differs: {row["path"]}')

    scores = read_json(root/'evidence/public_scores.json')
    entries = scores['entries']
    if len(entries) != 6 or len({r['id'] for r in entries}) != len(entries):
        raise ValueError('Expected six uniquely named submission milestones')
    if any(not math.isfinite(r['public_rmse']) or r['public_rmse'] <= 0 for r in entries):
        raise ValueError('Invalid reported score')
    gain = 1-entries[-1]['public_rmse']/entries[0]['public_rmse']
    if not math.isclose(gain, scores['relative_improvement'], abs_tol=1e-12):
        raise ValueError('Reported relative improvement differs')
    model = read_json(root/'evidence/final_model.json')
    for key in ('weights','effective_weights'):
        if any(w < 0 for w in model[key].values()) or not math.isclose(sum(model[key].values()),1):
            raise ValueError('Final weights must be nonnegative and sum to one')
    if model['effective_weights'] != {'hand185':.0625,'originalResNet':.0625,'DINO_small_D4':.125,'TabPFN486':.25,'Large518_D4_Ridge':.25,'CRADIO_D4_Ridge':.25}:
        raise ValueError('Published architecture changed')
    if model['submission_sha256'] != entries[-1]['submission_sha256'] or model['test_used_for_fit']:
        raise ValueError('Final submission provenance differs')

    evidence = read_json(root/'evidence/oof_metrics.json')
    count = 0
    if evidence['independent_holdout'] is not False:
        raise ValueError('Reused splits cannot be described as independent')
    for family, runs in evidence['experiments'].items():
        if [r['seed'] for r in runs] != [42,2026]:
            raise ValueError(f'Seed coverage differs: {family}')
        for run in runs:
            for m in run['metrics'].values():
                metric_check(m); count += 1
            for name, diagnostic in run['diagnosis'].items():
                groups = diagnostic['groups']
                for m in groups.values():
                    metric_check(m); count += 1
                if {'common16','remaining484'} <= groups.keys():
                    a,b = groups['common16'],groups['remaining484']
                    if (a['n'], b['n']) != (16,484) or not math.isclose(a['sse']+b['sse'],run['metrics'][name]['sse'],abs_tol=1e-7):
                        raise ValueError('Failure-group SSE does not sum to the total')

    # Internal documentation references must resolve. External URLs are not fetched.
    for p in [root/'README.md', *sorted((root/'docs').glob('*.md'))]:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            if not (p.parent/target.split('#')[0]).exists():
                raise ValueError(f'Broken local link in {p.name}: {target}')
    return {'status':'passed', 'source_files':len(paths), 'aggregate_metrics_checked':count,
            'public_relative_reduction_percent':100*gain,
            'independent_score_verification':False,
            'note':'Aggregate consistency and source snapshot only; no raw data, model inference or training.'}


if __name__ == '__main__':
    print(json.dumps(verify(),ensure_ascii=False,indent=2))
