"""Saved-train diagnostics only: no fitting, image extraction, GPU or test access."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/post_scattering_strategy_20261012'
SOURCES = {}
SEEDS = (31415, 27182, 16180)


def record(path):
    path = ROOT / path
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def table(path):
    df = pd.read_csv(record(path)).set_index('ID')
    if not df.index.is_unique:
        raise ValueError('Nonunique IDs')
    return df


def features(path, ids, keys):
    with np.load(record(path), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids)
        x = np.concatenate([z[k].astype(np.float64) for k in keys], axis=1)
    assert np.isfinite(x).all()
    return x


def distances(a, b):
    return np.sqrt(np.maximum((a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2*a@b.T, 0) / a.shape[1])


def run():
    common = set(table('reports/scattering_tabpfn_confirmation/common16_summary.csv').index)
    train = table('open/train.csv')
    ids = train.index.to_numpy()
    y = train.hardness.to_numpy()
    assert len(common) == 16 and len(ids) == 500
    mask = np.isin(ids, list(common))
    tail = np.zeros(500, bool)
    order = np.argsort(y, kind='stable')
    tail[order[:50]] = True
    tail[order[-50:]] = True
    hand = features('reports/local_structure/features/features.npz', ids, ['hand_original', 'local_structure'])
    morphology = features('reports/morphology/features/features.npz', ids, ['morphology'])
    scattering = features('reports/scattering_tabpfn/features/features.npz', ids, ['scattering'])
    x285 = np.c_[hand, morphology]
    x486 = np.c_[x285, scattering]
    assert x285.shape == (500, 285) and x486.shape == (500, 486)
    per_seed, neighbors, residuals = [], [], []
    for seed in SEEDS:
        df = table(f'reports/scattering_tabpfn_confirmation/seed{seed}/oof.csv').loc[ids]
        np.testing.assert_allclose(df.hardness, y, atol=1e-10, rtol=0)
        err = df.scattering_replaced_blend.to_numpy() - y
        olderr = df.fixed285_control_blend.to_numpy() - y
        residuals.append(err)
        sse = float(err @ err)
        cse = float(err[mask] @ err[mask])
        per_seed.append({'seed': seed, 'all_rmse': float(np.sqrt(sse/500)),
            'common16_rmse': float(np.sqrt(cse/16)), 'remaining484_rmse': float(np.sqrt((sse-cse)/484)),
            'common16_sse_share': cse/sse, 'common16_all_same_error_side_as285': bool(np.all(err[mask]*olderr[mask]>0)),
            'common16_in_worst20': int(mask[np.argsort(np.abs(err))[-20:]].sum()),
            'hypothetical_halve_common16_residuals_all_rmse': float(np.sqrt((sse-.75*cse)/500)),
            'hypothetical_zero_common16_residuals_all_rmse': float(np.sqrt((sse-cse)/500)),
            'lower_two_noise_quartiles_common16': None})
        folds = df.fold.to_numpy()
        for name, x in [('existing285', x285), ('scattering201', scattering), ('combined486', x486)]:
            for fold in range(5):
                ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                assert len(ti) == 400 and len(vi) == 100
                mu, sd = x[ti].mean(0), x[ti].std(0)
                sd = np.where(sd < 1e-12, 1., sd)
                a, b = (x[ti]-mu)/sd, (x[vi]-mu)/sd
                dd = distances(a, a)
                np.fill_diagonal(dd, np.inf)
                support = np.partition(dd, 4, axis=1)[:, :5].mean(1)
                dq = distances(b, a)
                nearest = np.argsort(dq, axis=1, kind='stable')[:, :5]
                for j, i in enumerate(vi):
                    ni = ti[nearest[j]]
                    yy = y[ni]
                    assert i not in ni and np.all(folds[ni] != folds[i])
                    gap = float(y[i]-yy.mean())
                    neighbors.append({'seed': seed, 'ID': ids[i], 'representation': name,
                        'common16': bool(mask[i]), 'tail_noncommon': bool(tail[i] and not mask[i]),
                        'distance_percentile': float(np.mean(support <= dq[j, nearest[j]].mean())*100),
                        'outside_neighbor_range': bool(y[i] < yy.min() or y[i] > yy.max()),
                        'target_minus_neighbor_mean': gap, 'absolute_target_gap': abs(gap),
                        **{f'neighbor{k+1}': ids[n] for k, n in enumerate(ni)}})
    nn = pd.DataFrame(neighbors)
    neighbor_summary = []
    for (seed, name), df in nn.groupby(['seed', 'representation'], sort=False):
        for group, m in [('common16', df.common16), ('remaining484', ~df.common16), ('tail_noncommon', df.tail_noncommon)]:
            z = df.loc[m]
            neighbor_summary.append({'seed': seed, 'representation': name, 'group': group, 'n': len(z),
                'median_distance_percentile': float(z.distance_percentile.median()),
                'above_95_distance_count': int((z.distance_percentile > 95).sum()),
                'outside_neighbor_range_count': int(z.outside_neighbor_range.sum()),
                'median_absolute_target_gap': float(z.absolute_target_gap.median())})
    err = np.stack(residuals)
    split = {}
    for group, m in [('all500', np.ones(500, bool)), ('common16', mask), ('remaining484', ~mask)]:
        z = err[:, m]
        persistent = float(np.mean(z.mean(0)**2))
        variance = float(np.mean(z.var(0)))
        mse = float(np.mean(z*z))
        np.testing.assert_allclose(persistent+variance, mse, rtol=1e-12)
        split[group] = {'mean_mse': mse, 'squared_mean_residual': persistent,
            'between_split_variance': variance, 'between_split_variance_fraction': variance/mse,
            'mean_prediction_sd': float(np.mean(z.std(0))), 'min_residual_correlation': float(np.min(np.corrcoef(z)[np.triu_indices(3, 1)]))}
    old_experts = pd.read_csv(record('reports/common_failure_diagnosis_20261011/common16_experts.csv'))
    ranges = []
    for seed in (42, 2026):
        df = table(f'reports/scattering_tabpfn/seed{seed}/oof.csv')
        z = old_experts.loc[old_experts.seed == seed].set_index('ID').loc[sorted(common)]
        np.testing.assert_allclose(z.hardness, df.loc[z.index].hardness, atol=1e-10, rtol=0)
        lo = np.minimum(z.expert_min.to_numpy(), df.loc[z.index].tabpfn486.to_numpy())
        hi = np.maximum(z.expert_max.to_numpy(), df.loc[z.index].tabpfn486.to_numpy())
        yy = z.hardness.to_numpy()
        ranges.append({'seed': seed, 'experts': 21, 'same_side_common16': int(((yy < lo) | (yy > hi)).sum()),
            'oracle_range_rmse_not_validation': float(np.sqrt(np.mean((np.clip(yy, lo, hi)-yy)**2)))})
    audit = table('reports/audit_train_features.csv').loc[ids]
    noise = audit.haar_hh_noise_proxy.to_numpy()
    nq = np.empty(500, int); nq[np.argsort(noise, kind='stable')] = np.repeat(np.arange(4), 125)
    for item in per_seed:
        item['lower_two_noise_quartiles_common16'] = int((mask & (nq < 2)).sum())
    result = {'scope': 'Retrospective saved train arrays only; no fitting, new extraction, GPU, test access or submission changes.',
        'per_seed': per_seed, 'split_variance': split, 'expert_ranges': ranges,
        'neighbor_summary': neighbor_summary, 'common16_noise_quartiles': [int((mask & (nq == q)).sum()) for q in range(4)],
        'hypothetical_warning': 'Residual changes assume knowledge of true labels and leave all other errors unchanged. Not achievable scores, Public estimates, or an irreducible error floor.',
        'neighbor_warning': 'Euclidean standardized feature neighbors, not the learned TabPFN metric. Train400 references only; labels used after neighbor search for diagnosis. No proof of label errors or absence of information in pixels.',
        'split_warning': 'Same500 reused, not independent samples or epistemic/aleatoric decomposition. Common16 selected retrospectively.',
        'sources': SOURCES, 'source': {'path': str(Path(__file__).resolve()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    for p, expected in SOURCES.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected
    OUT.mkdir(parents=True, exist_ok=True)
    nn.to_csv(OUT/'neighbors_all500.csv', index=False)
    nn.loc[nn.common16].to_csv(OUT/'neighbors_common16.csv', index=False)
    pd.DataFrame(neighbor_summary).to_csv(OUT/'neighbor_summary.csv', index=False)
    pd.DataFrame(per_seed).to_csv(OUT/'error_budget.csv', index=False)
    (OUT/'diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('per_seed', 'split_variance', 'expert_ranges')}, indent=2))
    print(pd.DataFrame(neighbor_summary).query("group == 'common16'").to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
