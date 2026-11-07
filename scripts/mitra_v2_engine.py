"""Shared context-only compression and fixed quarter-branch replacement."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits


def finite(a, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError('Invalid finite array/shape')
    return a


def project(train_x, query_x):
    """Official SVD recipe = mean-imputation, scaling, centered PCA; no y accepted."""
    x = finite(train_x, (400,486))
    q = finite(query_x, (100,486))
    pipe = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(),
                         PCA(n_components=256, random_state=0, svd_solver='full'))
    with threadpool_limits(limits=2):
        z = pipe.fit_transform(x)
        v = pipe.transform(q)
    pca = pipe[-1]
    return finite(z, (400,256)), finite(v, (100,256)), {
        'fit_rows':400,'query_rows':100,'input_width':486,'output_width':256,
        'method':'mean-imputation/StandardScaler/PCA256,fullSVD,whitenFalse',
        'explained_variance_fraction':float(pca.explained_variance_ratio_.sum()),
        'scaler_mean':pipe[1].mean_.tolist(), 'pca_mean':pca.mean_.tolist()}


def compose(base, large, radio, tab486, tab256, mitra256):
    b,l,c,t,u,m = [finite(a, (500,)) for a in (base,large,radio,tab486,tab256,mitra256)]
    fixed = .25*b + .25*l + .25*c
    return {'current':fixed+.25*t, 'tabpfn486':t, 'tabpfn_svd256':u, 'mitra_svd256':m,
            'tabpfn_svd256_control_blend':fixed+.25*u, 'mitra_replaced_blend':fixed+.25*m}
