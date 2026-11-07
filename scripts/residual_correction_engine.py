"""Small, fixed residual models. All fit APIs accept context labels only."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__:
    from . import evaluate_frozen_resnet as core
else:
    import evaluate_frozen_resnet as core

COMPONENTS = ('hand185', 'resnet1792', 'dino1152', 'tabpfn486')
GRIDS = {'hand185': (10., 100., 1000.), 'resnet1792': (10., 100., 1000.),
         'dino1152': (10., 100., 1000., 3000., 10000., 30000.)}
WEIGHTS = np.array([.125, .125, .25, .5])
SHRINKAGE = .5
PCA_DIM = 16
TREE = dict(loss='squared_error', learning_rate=.03, max_iter=100,
            max_leaf_nodes=4, max_depth=2, min_samples_leaf=40,
            l2_regularization=10., max_bins=32, early_stopping=False,
            categorical_features=None)


def finite(a, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError('Invalid array shape/finite values')
    return a


def splits(y, seed):
    y = finite(y)
    if y.ndim != 1 or len(y) < 60:
        raise ValueError('Expected at least60 fitting labels')
    return list(StratifiedKFold(3, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), core.rank_groups(y, 10)))


def ensemble(components):
    p = finite(components)
    if p.ndim != 2 or p.shape[1] != 4:
        raise ValueError('Four ordered component predictions required')
    # Preserve the original arithmetic as closely as possible.
    return .5 * (.25*p[:, 0] + .25*p[:, 1] + .5*p[:, 2] + p[:, 3])


@threadpool_limits.wrap(limits=2)
def fit_ridges(train_x, y, query_x, seed):
    """CPU limits end before TabPFN, preserving its archived torch4 runtime."""
    n_query = len(query_x['tabpfn486'])
    partitions = splits(y, seed)
    predictions, selections = [], {}
    for name, alphas in GRIDS.items():
        x, q = finite(train_x[name]), finite(query_x[name])
        if len(x) != len(y) or len(q) != n_query or x.shape[1] != q.shape[1]:
            raise ValueError('Base context/query dimensions differ')
        candidates = []
        for alpha in alphas:
            oof = np.full(len(y), np.nan)
            for ti, vi in partitions:
                model = core.make_estimator(alpha)
                model.fit(x[ti], y[ti])
                oof[vi] = model.predict(x[vi])
            finite(oof)
            candidates.append({'alpha': alpha, 'mse': float(np.mean((oof-y)**2))})
        best = min(candidates, key=lambda c: (c['mse'], -c['alpha']))
        model = core.make_estimator(best['alpha'])
        model.fit(x, y)
        predictions.append(finite(model.predict(q), (n_query,)))
        selections[name] = {'selected': best, 'candidates': candidates}
    return predictions, selections


def fit_base(train_x, train_y, query_x, seed, predictor):
    """Fresh base predictions; alpha choice never sees query labels.

    Each Ridge alpha is selected by a further3-fold CV INSIDE this context.
    It is invalid to reuse alpha selected on the enclosing outer400 labels.
    """
    if set(train_x) != set(COMPONENTS) or set(query_x) != set(COMPONENTS):
        raise ValueError('Missing/extra base representation')
    y = finite(train_y)
    n_query = len(query_x['tabpfn486'])
    predictions, selections = fit_ridges(train_x, y, query_x, seed)
    p, runtime = predictor(finite(train_x['tabpfn486']), y.copy(),
                           finite(query_x['tabpfn486']), seed)
    predictions.append(finite(p, (n_query,)))
    return np.column_stack(predictions), {'ridge_selections': selections,
        'alpha_selection_seed': seed, 'tabpfn': runtime,
        'context_rows': len(y), 'query_rows': n_query}


def fit_projection(train_x):
    x = finite(train_x)
    scale = StandardScaler().fit(x)
    pca = PCA(n_components=PCA_DIM, svd_solver='full', whiten=False).fit(scale.transform(x))
    return {'mean': scale.mean_, 'scale': scale.scale_, 'pca_mean': pca.mean_,
            'components': pca.components_, 'variance_ratio': pca.explained_variance_ratio_}


def project(x, params):
    return ((finite(x)-params['mean'])/params['scale']-params['pca_mean']) @ params['components'].T


def meta_features(components, noise, projection):
    p = finite(components)
    current = ensemble(p)
    # No targets, historical error ranks, common16 membership or image IDs.
    x = np.column_stack((p, current, p.std(axis=1),
                         p[:, 0]-p[:, 3], p[:, 1]-p[:, 3], p[:, 2]-p[:, 3],
                         finite(noise, (len(p),)), finite(projection, (len(p), PCA_DIM))))
    return finite(x, (len(p), 26))


@threadpool_limits.wrap(limits=2)
def fit_corrections(train_x, train_noise, train_components, train_y,
                    query_x, query_noise, query_components, seed):
    """Train residual targets from cross-fitted components, never in-sample fits."""
    p, q = finite(train_components), finite(query_components)
    residual = finite(train_y, (len(p),)) - ensemble(p)
    params = fit_projection(train_x)
    tx = meta_features(p, train_noise, project(train_x, params))
    qx = meta_features(q, query_noise, project(query_x, params))
    model = HistGradientBoostingRegressor(**TREE, random_state=seed)
    model.fit(tx, residual)
    correction = finite(model.predict(qx), (len(q),))
    # Calibration controls global under/overprediction without texture inputs.
    linear = core.make_estimator(100.)
    linear.fit(ensemble(p)[:, None], residual)
    calibration = finite(linear.predict(ensemble(q)[:, None]), (len(q),))
    return {'tree_delta': correction, 'calibration_delta': calibration}, {
        'context_rows': len(p), 'query_rows': len(q), 'meta_dimensions': 26,
        'pca': {n: a.tolist() for n, a in params.items()},
        'residual_mean': float(residual.mean()), 'residual_std': float(residual.std()),
        'tree_iterations': int(model.n_iter_), 'shrinkage': SHRINKAGE,
        'query_labels_accepted': False, 'early_stopping': False,
        'training_fit_is_not_validation': True}


def compose(current, deltas):
    current = finite(current)
    return {'current': current.copy(),
            'calibration_control': current+SHRINKAGE*finite(deltas['calibration_delta'], current.shape),
            'residual_corrected': current+SHRINKAGE*finite(deltas['tree_delta'], current.shape)}
