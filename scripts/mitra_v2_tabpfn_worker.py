"""Original TabPFN-v2 environment; one context-only prediction and process exit."""
import argparse
from pathlib import Path
import numpy as np
import evaluate_tabpfn_v2 as t


def run(source,output,seed):
    with np.load(source,allow_pickle=False) as a:
        if set(a.files) != {'train_x','train_y','query_x'}:
            raise ValueError('Only context targets may be sent to a predictor')
        x,y,q=a['train_x'],a['train_y'],a['query_x']
    if (x.shape,y.shape,q.shape) != ((400,256),(400,),(100,256)) or any(not np.isfinite(v).all() for v in (x,y,q)):
        raise ValueError('Invalid shared projection payload')
    predictor=t.Predictor(t.verify_weights(t.OUT),'cuda')
    pred,info=predictor(x,y,q,seed)
    t.write_json(output,{'predictions':pred.tolist(),'runtime':info,
        'seed':seed,'input_sha256':t.common.sha256_file(source)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();run(a.input,a.output,a.seed)
