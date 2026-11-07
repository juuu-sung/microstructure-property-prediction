"""Run the unchanged pinned worker with an explicitly recorded RAM-wait addendum."""
import argparse
import json
from pathlib import Path
import mitra_v2_worker as worker
import mitra_v2_memory_guard as guard


def run(input_path,output_path,seed,addendum_path):
    addendum=json.loads(addendum_path.read_text(encoding='utf-8'))
    if addendum['policy']!=guard.POLICY:raise ValueError('Recovery policy changed')
    for ref in addendum['sources'].values():
        if worker.assets.sha(ref['path'])!=ref['sha256']:raise ValueError('Recovery source changed')
    protocol=Path(addendum['original_protocol']['path'])
    if worker.assets.sha(protocol)!=addendum['original_protocol']['sha256']:
        raise ValueError('Original experiment protocol changed')
    events=[]
    startup=guard.install(events)
    events.append({'stage':'worker_startup',**startup()})
    original_write=worker.write
    def write_with_receipt(path,result):
        result['recovery_runtime']={'addendum_sha256':worker.assets.sha(addendum_path),
            'policy':guard.POLICY,'events':events,'mathematical_recipe_changed':False,
            'original_worker_sha256':worker.assets.sha(worker.__file__)}
        original_write(path,result)
    worker.write=write_with_receipt
    worker.run(input_path,output_path,seed,synthetic=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--addendum',type=Path,required=True)
    a=p.parse_args();run(a.input,a.output,a.seed,a.addendum)
