"""Portable final S50/L25/C25 composition, with strict ID/order checks.

Inputs must already be predictions from the matching fitted components.
This utility does not train models, select weights, calibrate or upload.
"""
from pathlib import Path
import argparse
import csv
import math
import os
import tempfile


def read_predictions(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['ID','hardness']:
            raise ValueError('Expected exactly ID,hardness columns')
        rows = list(reader)
    ids = [r['ID'] for r in rows]
    if not ids or any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('IDs must be nonempty and unique')
    values = [float(r['hardness']) for r in rows]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Predictions must be finite')
    return ids, values


def blend(scattering, large, radio):
    ids, s = scattering
    li,l = large
    ci,c = radio
    if ids != li or ids != ci or not ids:
        raise ValueError('All components must have identical ordered IDs')
    if any(len(x) != len(ids) for x in (s,l,c)):
        raise ValueError('Prediction length mismatch')
    if not all(math.isfinite(v) for x in (s,l,c) for v in x):
        raise ValueError('Non-finite component prediction')
    pred = [.5*a+.25*b+.25*d for a,b,d in zip(s,l,c,strict=True)]
    if not all(math.isfinite(v) for v in pred):
        raise ValueError('Non-finite ensemble prediction')
    return list(ids), pred


def save_new(path, result):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve prior submissions; choose a new output path')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='',dir=path.parent,delete=False) as f:
            temporary=Path(f.name)
            w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(*result,strict=True))
        if read_predictions(temporary) != result:
            raise ValueError('CSV round-trip differs')
        # Exclusive creation avoids overwriting another process's output.
        with path.open('xb') as out:
            out.write(temporary.read_bytes())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scattering','large','radio','output'):
        p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args()
    result=blend(*(read_predictions(getattr(args,n)) for n in ('scattering','large','radio')))
    save_new(args.output,result)
    print(f'Saved {len(result[0])} aligned predictions to {args.output}; weights=.50/.25/.25')


if __name__=='__main__':
    main()
