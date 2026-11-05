"""Offline PyTorch-only conversion; the Jittor data path never imports torch."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='Original dataset root containing train/test')
    p.add_argument('--output',required=True);p.add_argument('--stage',choices=['train','test'],required=True)
    p.add_argument('--evaluation-index',help='Only export scenes in this fixed index')
    p.add_argument('--limit',type=int);p.add_argument('--trusted-input',action='store_true',help='Required: .torch files are pickle and may execute code')
    args=p.parse_args()
    if not args.trusted_input:p.error('Only convert trusted official/local dataset files; acknowledge with --trusted-input')
    import torch
    source=Path(args.source)/args.stage;out=Path(args.output)/args.stage;out.mkdir(parents=True,exist_ok=True)
    if source.resolve()==out.resolve():raise ValueError('Never overwrite the original dataset')
    if (out/'index.json').exists():raise FileExistsError('Use a fresh conversion directory; existing dataset is preserved')
    index=json.loads((source/'index.json').read_text(encoding='utf-8'))
    fixed=json.loads(Path(args.evaluation_index).read_text(encoding='utf-8')) if args.evaluation_index else None
    keys=sorted(k for k in index if fixed is None or fixed.get(k) is not None)
    if fixed:
        missing=set(k for k,v in fixed.items() if v is not None)-set(index)
        if missing and args.limit is None:raise ValueError(f'{len(missing)} requested scenes are unavailable locally')
    if args.limit:keys=keys[:args.limit]
    by_chunk={}
    for key in keys:by_chunk.setdefault(index[key],set()).add(key)
    exported={};hashes={}
    for chunk_name,wanted in by_chunk.items():
        path=source/chunk_name
        chunk=torch.load(path,map_location='cpu',weights_only=False)
        for scene in chunk:
            key=scene['key']
            if key not in wanted:continue
            # A digest prevents arbitrary scene keys from becoming filesystem paths.
            name=hashlib.sha256(key.encode()).hexdigest()+'.npz'
            buffers=[np.asarray(x.cpu(),dtype=np.uint8).reshape(-1) for x in scene['images']]
            offsets=np.concatenate(([0],np.cumsum([len(x) for x in buffers],dtype=np.int64)))
            np.savez(out/name,key=np.asarray(key),cameras=scene['cameras'].cpu().numpy().astype(np.float32),image_bytes=np.concatenate(buffers),offsets=offsets)
            exported[key]=name
        hashes[chunk_name]=hashlib.sha256(path.read_bytes()).hexdigest()
        print(f'{chunk_name}: {len(exported)}/{len(keys)} scenes',flush=True)
        del chunk
    if set(exported)!=set(keys):raise RuntimeError('Chunk index and scene payload do not agree')
    (out/'index.json').write_text(json.dumps(exported,indent=2),encoding='utf-8')
    (out/'conversion.json').write_text(json.dumps({'source':str(source.resolve()),'count':len(exported),'source_sha256':hashes},indent=2),encoding='utf-8')
    if fixed:
        (out/'evaluation_subset.json').write_text(json.dumps({k:fixed[k] for k in keys},indent=2),encoding='utf-8')


if __name__=='__main__':main()

