"""Saved-prediction portfolio diagnosis; no fitting, tuning or image/model inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/post_micronet_strategy_20261013'
SOURCES = {}


def record(path):
    path = ROOT / path
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def table(path):
    df = pd.read_csv(record(path))
    if df.ID.duplicated().any():
        raise ValueError('Duplicate IDs')
    return df.set_index('ID')


def run():
    train = table('open/train.csv')
    ids, y = train.index.to_numpy(), train.hardness.to_numpy()
    old_path = record('reports/common_failure_diagnosis_20261011/common16_experts.csv')
    old = pd.read_csv(old_path)
    common_ids = sorted(old.ID.unique())
    mask = np.isin(ids, common_ids)
    assert len(ids) == 500 and mask.sum() == 16
    quality = table('reports/audit_train_features.csv').loc[ids]
    strata = np.empty(500, int)
    strata[np.argsort(y, kind='stable')] = np.repeat(np.arange(10), 50)
    noise_q = np.empty(500, int)
    noise_q[np.argsort(quality.haar_hh_noise_proxy, kind='stable')] = np.repeat(np.arange(4), 125)
    records, rows, late_scores = [], [], []
    for seed in (42, 2026):
        saved = old.loc[old.seed == seed].set_index('ID').loc[common_ids]
        np.testing.assert_allclose(saved.hardness, train.loc[common_ids].hardness, atol=1e-10, rtol=0)
        current = table(f'reports/scattering_tabpfn/seed{seed}/oof.csv').loc[ids]
        micronet = table(f'reports/efficientnet_micronet/seed{seed}/oof.csv').loc[ids]
        refined = table(f'reports/refined_structure_tabpfn/seed{seed}/oof.csv').loc[ids]
        for folder, manifest_name in (('scattering_tabpfn', 'completed_evaluation.json'),
                                       ('refined_structure_tabpfn', 'completed_evaluation.json'),
                                       ('efficientnet_micronet', 'completed.json')):
            manifest = json.loads(record(f'reports/{folder}/seed{seed}/{manifest_name}').read_text())
            path = ROOT / f'reports/{folder}/seed{seed}/oof.csv'
            assert SOURCES[str(path)] == manifest['files']['oof.csv']
        for frame in (current, micronet, refined):
            np.testing.assert_array_equal(frame.index, ids)
            np.testing.assert_allclose(frame.hardness, y, atol=1e-10, rtol=0)
            np.testing.assert_array_equal(frame.fold, current.fold)
        np.testing.assert_array_equal(current.scattering_replaced_blend, micronet.current)
        np.testing.assert_array_equal(current.scattering_replaced_blend, refined.current486_blend)
        np.testing.assert_allclose(micronet.current_micronet_equal,
                                   .5 * (micronet.current + micronet.micronet), rtol=0, atol=1e-10)
        new = pd.DataFrame({'tabpfn486': current.tabpfn486,
                            'tabpfn_legacy500': refined.tabpfn_legacy500,
                            'tabpfn_refined500': refined.tabpfn_refined500,
                            'imagenet': micronet.imagenet, 'micronet': micronet.micronet})
        assert np.isfinite(new.to_numpy()).all()
        extra = new.loc[common_ids]
        lo = np.minimum(saved.expert_min, extra.min(axis=1)).to_numpy()
        hi = np.maximum(saved.expert_max, extra.max(axis=1)).to_numpy()
        yy = train.loc[common_ids].hardness.to_numpy()
        outside = (yy < lo) | (yy > hi)
        for j, image_id in enumerate(common_ids):
            rows.append({'seed': seed, 'ID': image_id, 'hardness': float(yy[j]),
                         'expert_min_25': float(lo[j]), 'expert_max_25': float(hi[j]),
                         'target_outside_all25': bool(outside[j]),
                         'closest_interval_error_not_valid_prediction': float(np.clip(yy[j], lo[j], hi[j])-yy[j])})
        p = current.scattering_replaced_blend.to_numpy()
        err = p-y
        sse, common_sse = float(err @ err), float(err[mask] @ err[mask])
        all_rmse = float(np.sqrt(sse/500))
        for name in new:
            z = new[name].to_numpy()-y
            late_scores.append({'seed': seed, 'model': name, 'rmse': float(np.sqrt(np.mean(z*z))),
                                'common16_rmse': float(np.sqrt(np.mean(z[mask]**2))),
                                'same_side_as_current_common16': int((z[mask]*err[mask] > 0).sum())})
        records.append({'seed': seed, 'current_rmse': all_rmse,
                        'common16_rmse': float(np.sqrt(common_sse/16)),
                        'remaining484_rmse': float(np.sqrt((sse-common_sse)/484)),
                        'common16_sse_fraction': common_sse/sse,
                        'outside_range_of_25_standalone_models': int(outside.sum()),
                        'oracle_range_common16_rmse_not_validation': float(np.sqrt(np.mean((np.clip(yy,lo,hi)-yy)**2))),
                        'hypothetical_16_error_halved_all_rmse': float(np.sqrt((sse-.75*common_sse)/500)),
                        'hypothetical_16_error_zero_all_rmse': float(np.sqrt((sse-common_sse)/500)),
                        'common16_sse_gain_required_for_all_rmse_minus0p30_if_rest_fixed':
                            (sse-500*(all_rmse-.30)**2)/common_sse})
    source_reviews = {}
    for rel in ('reports/efficientnet_micronet/review/review_results.json',
                'reports/refined_structure_tabpfn/review/decision.json',
                'reports/post_scattering_strategy_20261012/diagnosis.json',
                'reports/structure_annotation_pilot/protocol.json',
                'reports/structure_auto_review/review_decision.json',
                'reports/structure_detector_refinement/review_decision.json'):
        source_reviews[rel] = str(record(rel))
    annotation_path = ROOT/'reports/structure_annotation_pilot/annotations/latest.json'
    result = {'scope': 'Saved train predictions only, no training/extraction/test access/weight optimization',
              'per_seed': records, 'common16': {'count': 16,
                   'target_decile_counts': [int((mask & (strata==q)).sum()) for q in range(10)],
                   'noise_quartile_counts': [int((mask & (noise_q==q)).sum()) for q in range(4)]},
              'annotation_file': str(annotation_path), 'annotation_file_exists': annotation_path.exists(),
              'review_sources': source_reviews,
              'limitations': ['25 saved model configurations are not independent experts',
                             'Range projection uses true labels: not a deployable prediction or validated score',
                             'Hypothetical changed errors are not Public estimates or an irreducible error floor',
                             'Common16 was selected retrospectively; neither fresh seeds nor subgroup labels create independent validation',
                             'Human annotation status refers only to the existing pilot save path'],
              'sources': SOURCES, 'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for path, sha in SOURCES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    pd.DataFrame(rows).to_csv(OUT/'common16_prediction_ranges.csv', index=False)
    pd.DataFrame(late_scores).to_csv(OUT/'recent_standalone_scores.csv', index=False)
    print(json.dumps({k:result[k] for k in ('per_seed','common16','annotation_file_exists')},indent=2))


if __name__ == '__main__':
    run()
