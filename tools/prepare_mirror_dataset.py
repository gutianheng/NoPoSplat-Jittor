"""Verified, resumable chunk download/conversion; single writer per output.

Default execution handles ONE new chunk. --all-chunks explicitly selects the
whole plan, with a disk-space estimate before starting. Only managed downloads
can be removed, after durable conversion; local source files are never deleted.
"""
from pathlib import Path,PurePosixPath
from urllib.request import Request,urlopen
from urllib.parse import urlsplit
import argparse,json,os,shutil,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from noposplat_jittor.dataset_conversion import ChunkConversion,sha256_file


def checked_relative(path):
    p=PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or p.suffix!='.torch':raise ValueError('Unsafe chunk path')
    return Path(str(p))


def torch_scenes(path):
    import torch  # Offline conversion only; never imported by the Jittor reader.
    chunk=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(chunk,list):raise ValueError('Expected a list of scenes')
    for scene in chunk:
        yield scene['key'],scene['cameras'].cpu().numpy(),[x.cpu().numpy() for x in scene['images']]


def download(row,path,endpoint=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size!=row['bytes'] or sha256_file(path)!=row['sha256']:raise ValueError('Existing raw cache conflicts with pinned source')
        return
    partial=path.with_name(path.name+'.part');offset=partial.stat().st_size if partial.exists() else 0
    if offset>row['bytes']:raise ValueError('Partial download exceeds expected size')
    if offset!=row['bytes']:
        url=row['url']
        if endpoint is not None:
            if endpoint!='https://hf-mirror.com' or urlsplit(url).netloc!='huggingface.co' or urlsplit(url).scheme!='https':raise ValueError('Only the documented HTTPS mirror of a pinned HuggingFace URL is supported')
            url=url.replace('https://huggingface.co/',endpoint+'/',1)
        request=Request(url,headers={'Range':f'bytes={offset}-','User-Agent':'NoPoSplat-pinned-data/1.0'})
        with urlopen(request,timeout=30) as response:
            if response.status==206:
                if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Bad resumed byte range')
                mode='ab'
            elif response.status==200:mode='wb';offset=0
            else:raise ValueError('Unexpected HTTP status')
            last=time.monotonic()
            with partial.open(mode) as f:
                while True:
                    block=response.read(1024*1024)
                    if not block:break
                    f.write(block);offset+=len(block)
                    if offset>row['bytes']:raise ValueError('Response exceeds declared chunk size')
                    if time.monotonic()-last>30:print(json.dumps({'chunk':row['path'],'downloaded':offset,'total':row['bytes']}),flush=True);last=time.monotonic()
                f.flush();os.fsync(f.fileno())
    if partial.stat().st_size!=row['bytes'] or sha256_file(partial)!=row['sha256']:raise ValueError('Incomplete or corrupted download; source not deserialized')
    os.replace(partial,path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--cache')
    p.add_argument('--local-source',help='Existing original dataset root; never modified');p.add_argument('--execute',action='store_true')
    p.add_argument('--source-store-allocation',help='Pinned permanent source placement; source files are never removed')
    group=p.add_mutually_exclusive_group();group.add_argument('--max-chunks',type=int,default=1);group.add_argument('--all-chunks',action='store_true')
    p.add_argument('--remove-downloaded-chunks',action='store_true');p.add_argument('--reserve-gib',type=float,default=10)
    p.add_argument('--mirror-endpoint',choices=['https://hf-mirror.com'])
    p.add_argument('--download-workers',type=int,default=1);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text(encoding='utf-8'));out=Path(a.output).resolve()/plan['stage']
    if a.max_chunks<1 or a.reserve_gib<0 or not 1<=a.download_workers<=16:p.error('Invalid chunk limit, reserve or concurrency')
    for row in plan['chunks']:checked_relative(row['path'])
    print(json.dumps({'dataset':plan['dataset'],'stage':plan['stage'],'selection':plan['selection'],'required_scenes':plan['selected_scene_count'],
        'required_chunks':len(plan['chunks']),'source_gib':plan['source_chunk_bytes']/2**30,'execute':a.execute,'new_chunk_limit':'all' if a.all_chunks else a.max_chunks}),flush=True)
    if not a.execute:return
    if a.local_source and a.source_store_allocation:p.error('Choose one original-source mode')
    if not a.local_source and not a.source_store_allocation and not a.cache:p.error('Download mode needs --cache')
    source_store=None
    if a.source_store_allocation:
        from noposplat_jittor.source_store import SourceStore
        source_store=SourceStore(a.source_store_allocation)
    cache=Path(a.cache).resolve()/plan['dataset']/plan['commit'] if a.cache else None
    source=Path(a.local_source).resolve() if a.local_source else None
    if source and (source==out or source.is_relative_to(out) or out.is_relative_to(source)):raise ValueError('Separate original source and converted output')
    if cache and (cache==out or cache.is_relative_to(out) or out.is_relative_to(cache)):raise ValueError('Separate cache and converted output')
    conversion=ChunkConversion(out,plan);pending=[r for r in plan['chunks'] if not conversion.verify_committed(r['path'])]
    if a.all_chunks and pending:
        needed=sum(r['bytes'] for r in pending)*1.05+max(r['bytes'] for r in pending)+a.reserve_gib*2**30
        if shutil.disk_usage(out).free<needed:raise OSError(f'Insufficient disk for the complete remaining stage: estimated {needed/2**30:.1f} GiB; no download started')
    count=0
    for row in pending:
        if not a.all_chunks and count>=a.max_chunks:break
        relative=checked_relative(row['path'])
        if source_store:path=source_store.ensure(plan,row,endpoint=a.mirror_endpoint,workers=a.download_workers)
        else:
            path=(source/relative) if source else cache/relative
            if not path.resolve().is_relative_to(source or cache):raise ValueError('Source escapes selected root')
        if not source and not source_store:
            if shutil.disk_usage(out).free<row['bytes']*2+256*1024**2:raise OSError('Not enough scratch space for one source+converted chunk')
            if a.download_workers==1:download(row,path,a.mirror_endpoint)
            else:
                from noposplat_jittor.verified_download import parallel_download
                parallel_download(row,path,a.mirror_endpoint,a.download_workers)
        result=conversion.convert(row,path,torch_scenes);count+=1
        if not source and not source_store and a.remove_downloaded_chunks:
            if not path.resolve().is_relative_to(cache) or not conversion.verify_committed(row['path']):raise ValueError('Unsafe cache cleanup')
            path.unlink()
        print(json.dumps({'chunk':row['path'],'result':result,'newly_processed':count}),flush=True)
    print((out/'conversion.json').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':main()
