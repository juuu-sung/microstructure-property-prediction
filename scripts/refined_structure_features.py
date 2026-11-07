"""Fourteen fixed automatic-proposal summaries, never semantic training masks.

No labels, IDs, cohort statistics, learned transforms, or feature selection.
The legacy and refined branches share a schema and exactly six perturbations.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
if __package__:
    from . import refine_structure_detector as detector
else:
    import refine_structure_detector as detector

NAMES = ('boundary_fraction', 'h_spacing', 'h_spacing_missing', 'v_spacing',
         'v_spacing_missing', 'transect_coverage', 'dark_fraction',
         'selected_object_fraction', 'ambiguous_object_fraction',
         'log_selected_count', 'log_ambiguous_count', 'boundary_repeatability_min',
         'object_iou_min', 'object_iou_missing')
NULLABLE = (1, 3, 12)
FLAGS = (2, 4, 13)
REPS = ('legacy14', 'refined14')
MAP_KEYS = ('before', 'structural_proxy', 'dark', 'old_compact', 'round_deep', 'ambiguous_compact')
SPEC = {
    'names': list(NAMES), 'dimensions': 14, 'probes': list(detector.quality.INTERNAL),
    'legacy': 'v2 mixed boundary + old compact objects; ambiguous mask empty.',
    'refined': 'Frozen refined structural_proxy (mixed union minus round-deep contour), '
               'round_deep selected objects and ambiguous_compact. Dark mask unchanged.',
    'summary': 'Boundary area; H/V observed intercept spacing and individual missing flags; '
               'usable transects/512; dark, selected, ambiguous area fractions; log1p 8-connected '
               'selected/ambiguous counts; worst six aligned 1px-tolerant boundary F1; '
               'worst six exact selected-object IoU and missing flag.',
    'absence': 'No interval or all-six-both-empty object comparisons => NaN with flag1. '
               'If all-six-both-empty boundaries, F1 set0 accompanied by zero boundary area/coverage. '
               'A missing object in only one view has IoU0, not missing.',
    'interpretation': 'Observed candidate summaries, not grain size, pore fraction, labels or confidence probabilities.',
}


def validate_matrix(x):
    if x.ndim != 2 or x.shape[1] != 14 or x.dtype != np.float64:
        raise ValueError('Expected float64 matrix with14 features')
    finite = [j for j in range(14) if j not in NULLABLE]
    if np.isinf(x).any() or not np.isfinite(x[:, finite]).all():
        raise ValueError('Invalid structure summary')
    if not np.array_equal(x[:, FLAGS], np.isnan(x[:, NULLABLE]).astype(float)):
        raise ValueError('Missing indicators disagree')


def summarize(maps, probes, refined):
    if len(probes) != 6:
        raise ValueError('All six aligned probes required')
    boundary = 'structural_proxy' if refined else 'before'
    selected = 'round_deep' if refined else 'old_compact'
    b, obj = maps[boundary], maps[selected]
    ambiguous = maps['ambiguous_compact'] if refined else np.zeros_like(obj)
    spacing = detector.old.all_intercepts(b)
    bf = [detector.previous.overlap(b, p[boundary], 'f1') for p in probes]
    oi = [detector.previous.overlap(obj, p[selected], 'iou') for p in probes]
    min_obj = min((v for v in oi if v is not None), default=None)
    h, v = spacing['h_spacing'], spacing['v_spacing']
    values = [b.mean(), h, h is None, v, v is None, spacing['usable_transect_fraction'],
              maps['dark'].mean(), obj.mean(), ambiguous.mean(),
              np.log1p(ndi.label(obj, np.ones((3, 3), bool))[1]),
              np.log1p(ndi.label(ambiguous, np.ones((3, 3), bool))[1]),
              min((v for v in bf if v is not None), default=0.), min_obj, min_obj is None]
    row = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    validate_matrix(row[None, :])
    return row


def extract(gray):
    _, maps = detector.detect(gray)
    probes = []
    for name in detector.quality.INTERNAL:
        _, other = detector.detect(detector.quality.perturb(gray, name))
        probes.append({k: detector.old.v1.align(other[k], name) for k in MAP_KEYS})
    rows = {rep: summarize(maps, probes, rep == 'refined14') for rep in REPS}
    return rows, {k: maps[k] for k in MAP_KEYS}


def impute_context(train_x, query_x):
    a, b = train_x.copy(), query_x.copy()
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 500 or b.shape[1] != 500:
        raise ValueError('Expected486 fixed +14 structure columns')
    for x in (a, b):
        if not np.isfinite(x[:, :486]).all():
            raise ValueError('Original486 must be finite')
        validate_matrix(x[:, 486:])
    medians = []
    for j in NULLABLE:
        col = 486+j
        observed = a[np.isfinite(a[:, col]), col]
        median = float(np.median(observed)) if observed.size else 0.
        a[np.isnan(a[:, col]), col] = median
        b[np.isnan(b[:, col]), col] = median
        medians.append({'column': col, 'median': median, 'all_missing_context': not bool(observed.size)})
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Imputation did not produce finite inputs')
    return a, b, {'context_medians': medians, 'query_used_for_fit': False,
                  'original486_unchanged': True, 'missing_flags_retained': True}
