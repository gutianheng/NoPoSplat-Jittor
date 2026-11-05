"""Freeze an explicit reduced protocol, then prepare SHA-verified scene archives.

Training uses seeded chunk-cluster sampling to bound transport volume. Test
selection is seeded and stratified by official overlap group. No test images or
metric values are used to choose scenes. This is not the paper full-data setup.
"""
from pathlib import Path
import argparse,concurrent.futures,copy,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from noposplat_jittor.dataset_conversion import ChunkConversion,atomic_json,plan_digest
from noposplat_jittor.evaluation_summary import overlap_metadata


def release_verified_source(source,row,conversion,cache):
    """Delete only this exact managed source after its payload commit is verified."""
    from noposplat_jittor.dataset_conversion import sha256_file
    source=Path(source);cache=Path(cache).resolve()
    if source.is_symlink() or not source.resolve().is_relative_to(cache) or not source.is_file():
        raise ValueError('Source cleanup path escapes managed cache')
    if not conversion.verify_committed(row['path']):raise ValueError('Source has no verified committed conversion')
    if source.stat().st_size!=row['bytes'] or sha256_file(source)!=row['sha256']:
        raise ValueError('Source changed after conversion; preserve it')
    source.unlink()
    report=conversion.root/'source_cache_cleanup.json'
    state=json.loads(report.read_text()) if report.exists() else {'scope':'Exact raw source files only; retained scene payloads and source SHA ledger are not deleted.','released':[]}
    state['released'].append({'source_chunk':row['path'],'sha256':row['sha256'],'bytes':row['bytes'],'unix_time':time.time()})
    atomic_json(report,state)


def reduced_plan(original,selected,selection):
    plan=copy.deepcopy(original);wanted=set(selected)
    plan['chunks']=[dict(row,scenes=[k for k in row['scenes'] if k in wanted]) for row in original['chunks'] if wanted.intersection(row['scenes'])]
    plan.update(selection=selection,selected_scene_count=len(wanted),source_chunk_bytes=sum(r['bytes'] for r in plan['chunks']))
    if {k for r in plan['chunks'] for k in r['scenes']}!=wanted:raise ValueError('Selected scenes missing from source plan')
    return plan


def create(a):
    out=Path(a.protocol)
    if out.exists():raise FileExistsError(out)
    plans=Path(a.plans);rng=np.random.default_rng(a.seed)
    train=json.loads((plans/'re10k_train.json').read_text());pool=[];cluster_paths=[]
    for i in rng.permutation(len(train['chunks'])):
        row=train['chunks'][int(i)];pool.extend(row['scenes']);cluster_paths.append(row['path'])
        if len(pool)>=a.train_scenes:break
    selected=rng.choice(sorted(pool),a.train_scenes,replace=False).tolist()
    train=reduced_plan(train,selected,'seeded_chunk_cluster_subset')
    train.update(selection_seed=a.seed,selection_cluster_paths=cluster_paths)
    out.mkdir(parents=True);atomic_json(out/'re10k_train_plan.json',train)
    reports={'scope':__doc__,'seed':a.seed,'training':{'scenes':len(selected),'chunks':len(train['chunks']),'bytes':train['source_chunk_bytes'],'plan_sha256':plan_digest(train)},'testing':{}}
    for dataset in ('re10k','acid'):
        plan=json.loads((plans/f'{dataset}_test.json').read_text())
        index=json.loads((Path(a.indices)/f'evaluation_index_{dataset}.json').read_text())
        available={k for row in plan['chunks'] for k in row['scenes']};groups={name:[] for name in ('small','medium','large')}
        for key,entry in sorted(index.items()):
            if entry is None or key not in available:continue
            tag=overlap_metadata(entry)['overlap_group']
            if tag in groups:groups[tag].append(key)
        keys=[];counts={}
        for i,(tag,members) in enumerate(groups.items()):
            count=a.test_scenes//3+(i<a.test_scenes%3)
            if len(members)<count:raise ValueError('Insufficient group '+tag)
            keys.extend(rng.choice(members,count,replace=False).tolist());counts[tag]=count
        subset={k:index[k] for k in sorted(keys)}
        if dataset=='re10k' and set(selected).intersection(keys):raise ValueError('Train/test scene leakage')
        plan=reduced_plan(plan,keys,'seeded_overlap_stratified_subset');plan['selection_seed']=a.seed
        atomic_json(out/f'{dataset}_test_plan.json',plan);atomic_json(out/f'evaluation_index_{dataset}.json',subset)
        reports['testing'][dataset]={'scenes':len(keys),'chunks':len(plan['chunks']),'bytes':plan['source_chunk_bytes'],'overlap_groups':counts,'plan_sha256':plan_digest(plan)}
    atomic_json(out/'protocol.json',reports);print(json.dumps(reports,indent=2),flush=True)


def prepare(a):
    from noposplat_jittor.verified_download import parallel_download
    from tools.prepare_mirror_dataset import torch_scenes
    protocol=Path(a.protocol);output=Path(a.output);cache=Path(a.cache)
    for name in a.selection:
        plan=json.loads((protocol/f'{name}_plan.json').read_text());dataset=plan['dataset'];conversion=ChunkConversion(output/dataset/plan['stage'],plan)
        pending=[r for r in plan['chunks'] if not conversion.verify_committed(r['path'])]
        def fetch(row):
            path=cache/dataset/plan['commit']/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
            size=a.range_bytes;marker=path.with_name(path.name+'.ranges')/'source.json'
            if marker.exists():
                size=json.loads(marker.read_text())['range_bytes']
                if size not in (4*1024**2,16*1024**2,32*1024**2):raise ValueError('Unexpected existing range size')
            split=a.download_route=='official_split'
            parallel_download(row,path,endpoint=None if split else 'https://hf-mirror.com',workers=a.connections,range_bytes=size,split_route=split)
            return path
        # Bounded prefetch: at most `parallel_chunks` sources are in flight.
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel_chunks) as pool:
            futures={};iterator=iter(pending)
            for _ in range(a.parallel_chunks):
                row=next(iterator,None)
                if row is not None:futures[pool.submit(fetch,row)]=row
            completed=0
            while futures:
                done,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    tick=time.monotonic();row=futures.pop(future);source=future.result();result=conversion.convert(row,source,torch_scenes);completed+=1
                    if a.delete_committed_sources:release_verified_source(source,row,conversion,cache)
                    print(json.dumps({'selection':name,'chunk':row['path'],'result':result,'completed':completed,'required':len(pending),'seconds':time.monotonic()-tick}),flush=True)
                    nxt=next(iterator,None)
                    if nxt is not None:futures[pool.submit(fetch,nxt)]=nxt
        print((output/dataset/plan['stage']/'conversion.json').read_text(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['plan','prepare']);p.add_argument('--protocol',required=True)
    p.add_argument('--plans',default='configs/data_plans');p.add_argument('--indices',default='configs');p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--train-scenes',type=int,default=1000);p.add_argument('--test-scenes',type=int,default=100)
    p.add_argument('--selection',nargs='+',choices=['re10k_train','re10k_test','acid_test'],default=['re10k_train','re10k_test','acid_test'])
    p.add_argument('--output',default='data/reduced_20261103');p.add_argument('--cache',default='data/reduced_sources_20261103')
    p.add_argument('--parallel-chunks',type=int,default=4);p.add_argument('--connections',type=int,default=6)
    p.add_argument('--range-bytes',type=int,default=4*1024**2)
    p.add_argument('--download-route',choices=['mirror','official_split'],default='mirror')
    p.add_argument('--delete-committed-sources',action='store_true',help='Release exact raw cache file only after full SHA and committed converted payload verification')
    a=p.parse_args()
    if not 1<=a.parallel_chunks<=8 or not 1<=a.connections<=16:p.error('Invalid concurrency')
    try:(create if a.mode=='plan' else prepare)(a)
    except Exception as error:
        if a.mode=='prepare':
            from noposplat_jittor.verified_download import is_transient_download_error
            atomic_json(Path(a.output)/'preparation_failure.json',{'error_type':type(error).__name__,
                'http_status':getattr(error,'code',None),'retryable_transport_failure':is_transient_download_error(error),
                'unix_time':time.time(),'note':'No source hash, image, mathematical or filesystem integrity error is classified as a retryable HTTP failure.'})
        raise


if __name__=='__main__':main()
