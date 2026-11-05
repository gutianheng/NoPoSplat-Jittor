"""Download an author checkpoint with size/SHA256 verification and resumable .part."""
from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import hashlib
import json
import os
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset',choices=['re10k','acid'],required=True)
p.add_argument('--max-seconds',type=int,default=900)
args=p.parse_args()
name=args.dataset+'.ckpt'
api='https://huggingface.co/api/models/botaoye/NoPoSplat/tree/main'
with urlopen(api,timeout=30) as response: entries=json.load(response)
entry=next(x for x in entries if x.get('path')==name)
expected_size=int(entry['size'])
expected_sha=entry['lfs']['oid'].removeprefix('sha256:')
if len(expected_sha)!=64:raise ValueError('Missing author repository SHA256')
path=Path('weights')/name
partial=path.with_suffix('.ckpt.part')
path.parent.mkdir(exist_ok=True)
report_path=Path('reports')/(args.dataset+'_checkpoint_download.json')
report={'repository':'botaoye/NoPoSplat','file':name,'expected_bytes':expected_size,
        'expected_sha256':expected_sha,'status':'started'}
report_path.write_text(json.dumps(report,indent=2))
start=time.monotonic();last=start
if not path.exists():
    offset=partial.stat().st_size if partial.exists() else 0
    headers={'Range':f'bytes={offset}-'} if offset else {}
    url='https://huggingface.co/botaoye/NoPoSplat/resolve/main/'+name
    with urlopen(Request(url,headers=headers),timeout=60) as response:
        append=bool(offset and response.status==206)
        if not append:offset=0
        with partial.open('ab' if append else 'wb') as out:
            while True:
                block=response.read(4*1024*1024)
                if not block:break
                out.write(block);offset+=len(block)
                now=time.monotonic()
                if now-last>=10:
                    print(f'{name}: {offset}/{expected_size} bytes; elapsed {now-start:.1f}s',flush=True)
                    last=now
                if now-start>args.max_seconds:
                    report.update(status='partial_time_limit',bytes=offset)
                    report_path.write_text(json.dumps(report,indent=2))
                    raise SystemExit('Download time guard reached; partial retained')
    if partial.stat().st_size!=expected_size:raise ValueError('Checkpoint download size mismatch')
    candidate=partial
else:candidate=path
if candidate.stat().st_size!=expected_size:raise ValueError('Existing checkpoint size mismatch')
sha=hashlib.sha256()
with candidate.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):sha.update(block)
actual_sha=sha.hexdigest()
if actual_sha!=expected_sha:raise ValueError('Author checkpoint SHA256 mismatch')
if candidate==partial:os.replace(partial,path)
report.update(status='verified',actual_sha256=actual_sha,elapsed_seconds=time.monotonic()-start)
report_path.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
