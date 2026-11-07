"""Context-only local-descriptor Fisher encoding and matched Ridge selection."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import warnings
import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

if __package__:
    from . import large_radio_d4_engine as d4
else:
    import large_radio_d4_engine as d4

ridge = d4.ridge
PRIMARY = 'fisher_replaced_blend'
CONTROL = 'global_retuned_blend'
REPS = ('global_retuned', 'global_plus_fisher')


@dataclass(frozen=True)
class FisherSpec:
    components: int = 32
    clusters: int = 8
    views: int = 8
    samples_per_view: int = 16
    max_iter: int = 200
    tolerance: float = 1e-3
    reg_covar: float = 1e-4

    @property
    def width(self): return 2*self.components*self.clusters+self.clusters


SPEC = FisherSpec()


def sample_positions(image_id, tokens=1369, spec=SPEC):
    if tokens < spec.samples_per_view: raise ValueError('Insufficient spatial tokens')
    # Identical samples whenever an interrupted run is resumed; no label access.
    seed = int.from_bytes(hashlib.sha256(('radio-fv-v1:'+str(image_id)).encode()).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    return np.stack([np.sort(rng.choice(tokens, spec.samples_per_view, replace=False)) for _ in range(spec.views)])


def sample_views(tokens, image_id, spec=SPEC):
    x = np.asarray(tokens)
    if x.ndim != 3 or len(x) != spec.views: raise ValueError('Expected view/token/channel array')
    pos = sample_positions(image_id, x.shape[1], spec)
    return np.concatenate([x[v, pos[v]] for v in range(spec.views)]).astype(np.float32)


def fit_dictionary(samples, seed, spec=SPEC):
    """samples contains ONLY context images, with the same count per image/view."""
    a = np.asarray(samples, dtype=np.float32)
    if a.ndim != 3 or a.shape[1] != spec.views*spec.samples_per_view or not np.isfinite(a).all():
        raise ValueError('Expected balanced context image samples')
    x = a.reshape(-1, a.shape[-1])
    if min(x.shape) < spec.components or len(x) < spec.clusters: raise ValueError('Dictionary dimensions not supported')
    pca = PCA(spec.components, whiten=True, svd_solver='randomized', iterated_power=4, random_state=seed)
    pca.fit(x)
    # Use the same projection as inference (randomized fit_transform can differ).
    z = pca.transform(x)
    if np.any(pca.explained_variance_ <= 1e-12): raise ValueError('Degenerate PCA; no implicit dimension change')
    model = GaussianMixture(spec.clusters, covariance_type='diag', tol=spec.tolerance,
        reg_covar=spec.reg_covar, max_iter=spec.max_iter, n_init=1, init_params='kmeans', random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        model.fit(z)
    if not model.converged_: raise ValueError('GMM did not converge within fixed200 iterations; inspect before changing protocol')
    params = {'pca_mean':pca.mean_.astype(np.float64), 'pca_components':pca.components_.astype(np.float64),
        'pca_scale':np.sqrt(pca.explained_variance_).astype(np.float64), 'weights':model.weights_.astype(np.float64),
        'means':model.means_.astype(np.float64), 'covariances':model.covariances_.astype(np.float64)}
    validate_dictionary(params, x.shape[1], spec)
    audit = {'context_images':len(a), 'sampled_tokens_per_image':a.shape[1], 'sampled_tokens':len(x),
        'seed':int(seed), 'spec':asdict(spec), 'pca_explained_variance_ratio':float(pca.explained_variance_ratio_.sum()),
        'gmm_converged':True, 'gmm_iterations':int(model.n_iter_), 'gmm_weights':model.weights_.tolist(),
        'gmm_lower_bound':float(model.lower_bound_), 'labels_used':False, 'query_samples_used':False}
    return params, audit


def validate_dictionary(p, channels, spec=SPEC):
    shapes = {'pca_mean':(channels,), 'pca_components':(spec.components,channels), 'pca_scale':(spec.components,),
        'weights':(spec.clusters,), 'means':(spec.clusters,spec.components), 'covariances':(spec.clusters,spec.components)}
    if set(p) != set(shapes): raise ValueError('Unexpected dictionary schema')
    for k, shape in shapes.items(): ridge.finite(p[k], shape)
    if (np.any(p['pca_scale'] <= 0) or np.any(p['weights'] <= 0) or np.any(p['covariances'] <= 0)
            or abs(np.sum(p['weights'])-1) > 1e-6): raise ValueError('Invalid dictionary scales/mixture')


def project(tokens, params):
    x = ridge.finite(tokens)
    return ((x-params['pca_mean']) @ params['pca_components'].T)/params['pca_scale']


def fisher_projected(z, p):
    """Diagonal-GMM mean/variance score, power/L2 normalized, plus occupancy.

    Sufficient statistics avoid materializing tokens x clusters x dimensions.
    Occupancy is stored separately and sums to one for every view.
    """
    z = ridge.finite(z)
    weights, means, cov = p['weights'], p['means'], p['covariances']
    if z.ndim != 2 or z.shape[1] != means.shape[1] or len(z) == 0: raise ValueError('Bad projected tokens')
    log_prob = -.5*(z*z @ (1/cov).T - 2*z @ (means/cov).T
        + np.sum(means*means/cov+np.log(cov),axis=1)[None,:] + z.shape[1]*np.log(2*np.pi))
    log_prob += np.log(weights)[None,:]
    prob = np.exp(log_prob-logsumexp(log_prob,axis=1,keepdims=True))
    mass = prob.mean(0)
    first = prob.T @ z / len(z)
    second = prob.T @ (z*z) / len(z)
    u = (first-mass[:,None]*means)/np.sqrt(weights[:,None]*cov)
    v = (second-2*means*first+mass[:,None]*(means*means-cov))/(np.sqrt(2*weights[:,None])*cov)
    fv = np.concatenate((u.ravel(),v.ravel()))
    fv = np.sign(fv)*np.sqrt(np.abs(fv))
    fv /= max(float(np.linalg.norm(fv)),1e-12)
    result = np.concatenate((fv,mass))
    return ridge.finite(result)


def encode_image(tokens, params, spec=SPEC):
    x = np.asarray(tokens)
    if x.ndim != 3 or len(x) != spec.views: raise ValueError('Expected all eight views of one image')
    validate_dictionary(params, x.shape[-1], spec)
    total = np.zeros(spec.width, np.float64)
    for view in x:
        total += fisher_projected(project(view,params),params)
    return (total/spec.views).astype(np.float32)


def choose(inner, y, other):
    """other=.5S_inner+.25LargeD4_inner, independent of the same query targets."""
    y = ridge.finite(y); other = ridge.finite(other, y.shape)
    if set(inner) != {str(i) for i in d4.INDICES}: raise ValueError('Unexpected Ridge candidates')
    scores = []
    for i in d4.INDICES:
        pred = ridge.finite(inner[str(i)], y.shape)
        scores.append({'index':i, 'alpha':ridge.CANDIDATES[i]['alpha'],
            'standalone_mse':float(np.mean((pred-y)**2)), 'blend_mse':float(np.mean((other+.25*pred-y)**2))})
    return {'selected':min(scores,key=lambda r:(r['blend_mse'],r['index'])), 'scores':scores,
        'objective':'.5S_inner+.25LargeD4_inner+.25C_inner'}


def inner_large(x, y, q, seed, runner=None):
    """Select Large alpha within this inner context; never accept query labels."""
    run = runner or ridge.fit_candidates
    inner = {str(i):np.full(len(y),np.nan) for i in d4.INDICES}
    for ti, vi in ridge.inner_splits(y,seed):
        r = run(x[ti], y[ti], x[vi], d4.INDICES)
        for i,p in r['predictions'].items(): inner[i][vi] = p
    scores = {i:float(np.mean((ridge.finite(p)-y)**2)) for i,p in inner.items()}
    best = min(scores,key=lambda i:(scores[i],int(i)))
    r = run(x,y,q,[int(best)])
    return {'predictions':r['predictions'][best], 'alpha':ridge.CANDIDATES[int(best)]['alpha'],
        'context_rows':len(y), 'query_rows':len(q), 'inner_mse':scores, 'query_labels_accepted':False}


def compose(s, l, old_c, control_c, fisher_c):
    s = ridge.finite(s)
    l,a,b,c = [ridge.finite(p,s.shape) for p in (l,old_c,control_c,fisher_c)]
    other = .5*s+.25*l
    return {'current':other+.25*a, 'radio_d4_ridge':a, 'global_retuned_ridge':b,
        'global_plus_fisher_ridge':c, CONTROL:other+.25*b, PRIMARY:other+.25*c}
