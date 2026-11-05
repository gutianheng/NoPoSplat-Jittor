"""Build a pinned scene/chunk plan from publisher indices; no image downloads."""
from pathlib import Path,PurePosixPath
from urllib.request import Request,urlopen
from urllib.parse import quote
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from noposplat_jittor.dataset_conversion import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',required=True);p.add_argument('--dataset',choices=('re10k','acid'),required=True)
    p.add_argument('--stage',choices=('train','test'),required=True);p.add_argument('--evaluation-index');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    info=json.loads(Path(a.inventory).read_text(encoding='utf-8'))['datasets'][a.dataset];repo=info['repository'];commit=info['commit']
    filename=a.stage+'/index.json';files={f['path']:f for f in info['files_metadata']};meta=files[filename]
    url=f'https://huggingface.co/datasets/{repo}/resolve/{commit}/{filename}'
    with urlopen(Request(url,headers={'User-Agent':'NoPoSplat-data-planner/1.0'}),timeout=30) as response:data=response.read(16*1024*1024+1)
    if len(data)>16*1024*1024 or len(data)!=meta['bytes']:raise ValueError('Unexpected source index size')
    if meta.get('lfs_sha256') and hashlib.sha256(data).hexdigest()!=meta['lfs_sha256']:raise ValueError('Source index digest mismatch')
    index=json.loads(data);fixed=json.loads(Path(a.evaluation_index).read_text(encoding='utf-8')) if a.evaluation_index else None
    keys=sorted(k for k,v in fixed.items() if v is not None) if fixed is not None else sorted(index)
    missing=set(keys)-set(index)
    if missing:raise ValueError(f'{len(missing)} selected scenes unavailable')
    groups={}
    for key in keys:
        name=index[key];relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe chunk path')
        path=name if name.startswith(a.stage+'/') else a.stage+'/'+name
        if path not in files:raise ValueError('Missing chunk metadata: '+path)
        groups.setdefault(path,[]).append(key)
    chunks=[]
    for path,scenes in sorted(groups.items()):
        item=files[path];sha=item['lfs_sha256']
        if not sha or len(sha)!=64:raise ValueError('Publisher LFS SHA256 required for '+path)
        chunks.append({'path':path,'bytes':item['bytes'],'sha256':sha,'scenes':scenes,
            'url':f'https://huggingface.co/datasets/{repo}/resolve/{commit}/{quote(path,safe="/")}'})
    plan={'version':1,'dataset':a.dataset,'stage':a.stage,'repository':repo,'commit':commit,'source_index_sha256':hashlib.sha256(data).hexdigest(),
        'selection':'official_nonempty_evaluation_index' if fixed is not None else 'all_source_index_scenes',
        'evaluation_index_sha256':hashlib.sha256(Path(a.evaluation_index).read_bytes()).hexdigest() if fixed is not None else None,
        'source_scene_count':len(index),'selected_scene_count':len(keys),'source_chunk_bytes':sum(x['bytes'] for x in chunks),'chunks':chunks}
    out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,plan)
    print(json.dumps({k:v for k,v in plan.items() if k!='chunks'}),flush=True)


if __name__=='__main__':main()
