"""Bounded verified chunk cache and resumable author-style chunk traversal.

The neural-network process never imports PyTorch. A separate CPU process reads
SHA-verified original .torch files and publishes framework-neutral NPZ scenes.
Eviction only touches entries created under this cache's ownership marker.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import copy,json,math,os,shutil,subprocess,time
import numpy as np
from .dataset_conversion import atomic_json,plan_digest,sha256_file


@contextmanager
def file_lock(path,*,shared=False,blocking=True):
    """OS locks release on process death; stale marker files are harmless."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if not path.stat().st_size:f.write(b'0');f.flush()
        acquired=False
        try:
            if os.name=='nt':
                import msvcrt
                mode=(msvcrt.LK_RLCK if shared else msvcrt.LK_LOCK) if blocking else (msvcrt.LK_NBRLCK if shared else msvcrt.LK_NBLCK)
                f.seek(0);msvcrt.locking(f.fileno(),mode,1)
            else:
                import fcntl
                mode=fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                fcntl.flock(f.fileno(),mode|(0 if blocking else fcntl.LOCK_NB))
            acquired=True;yield
        finally:
            if acquired:
                if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
                else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def single_chunk_plan(plan,row):
    subset=copy.deepcopy(plan)
    subset.update(chunks=[row],selection='bounded_cache_entry',selected_scene_count=len(row['scenes']),source_chunk_bytes=row['bytes'])
    return subset


class BoundedChunkCache:
    def __init__(self,root,max_bytes,reserve_bytes=10*2**30,*,converter_python=None,network='direct',preparer=None,local_source=None,download_workers=12,source_store_allocation=None):
        self.root=Path(root).resolve();self.max_bytes=int(max_bytes);self.reserve_bytes=int(reserve_bytes)
        if self.max_bytes<=0 or self.reserve_bytes<0:raise ValueError('Invalid cache/disk reserve')
        self.python=converter_python;self.network=network;self.preparer=preparer;self.local_source=local_source;self.download_workers=int(download_workers)
        self.source_store_allocation=source_store_allocation
        if local_source and source_store_allocation:raise ValueError('Choose local original source or allocated persistent source store')
        if not 1<=self.download_workers<=16:raise ValueError('Download workers must be 1..16')
        if network not in ('direct','autodl_turbo','hf_mirror'):raise ValueError('Unknown download transport')
        marker=self.root/'.nps_chunk_cache.json'
        if not marker.exists() and self.root.exists() and any(self.root.iterdir()):raise FileExistsError('Refusing to manage a nonempty, unowned cache directory')
        self.root.mkdir(parents=True,exist_ok=True)
        with file_lock(self.root/'cache.lock'):
            if marker.exists():
                binding=json.loads(marker.read_text())
                if binding!={'format':'noposplat-bounded-cache-v1','max_bytes':self.max_bytes}:raise ValueError('Cache ownership/size differs; do not reuse blindly')
            else:atomic_json(marker,{'format':'noposplat-bounded-cache-v1','max_bytes':self.max_bytes})
            self.entries=self.root/'entries';self.entries.mkdir(exist_ok=True)
            if not (self.root/'state.json').exists():self._save({'version':1,'entries':{}})

    def _state(self):return json.loads((self.root/'state.json').read_text())
    def _save(self,state):atomic_json(self.root/'state.json',state)
    def _id(self,plan,row):return plan_digest({'plan':plan_digest(plan),'chunk':row['path']})
    def _entry(self,ident):
        if len(ident)!=64 or any(x not in '0123456789abcdef' for x in ident):raise ValueError('Unsafe cache entry id')
        path=self.entries/ident
        if path.is_symlink() or path.resolve().parent!=self.entries:raise ValueError('Cache entry escapes owned entries')
        return path
    def _lock(self,ident,**kwargs):return file_lock(self.root/'locks'/f'{ident}.lock',**kwargs)
    def _audit(self,event):
        with (self.root/'audit.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'time_ns':time.time_ns(),**event},ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    def _bytes(self,path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())

    def _room(self,state,required,protected):
        def enough():
            return sum(x['accounted_bytes'] for x in state['entries'].values())+required<=self.max_bytes and shutil.disk_usage(self.root).free>=required+self.reserve_bytes
        for ident,row in sorted(list(state['entries'].items()),key=lambda pair:pair[1]['used_ns']):
            if enough():break
            if ident==protected:continue
            lock=self._lock(ident,blocking=False)
            try:lock.__enter__()
            except OSError:continue  # A trainer reader or a live converter owns it.
            try:
                path=self._entry(ident)
                if path.exists():shutil.rmtree(path)
                del state['entries'][ident];self._save(state);self._audit({'event':'evicted','entry':ident,'chunk':row['chunk'],'plan_sha256':row['plan_sha256']})
            finally:lock.__exit__(None,None,None)
        if not enough():raise OSError('Owned cache cannot free sufficient space without evicting active leases; increase cache/disk or lower prefetch')

    def _verify(self,folder,plan,row):
        subset=single_chunk_plan(plan,row);stage=folder/'converted'/plan['stage']
        state=json.loads((stage/'conversion_ledger.json').read_text())
        if state['plan_sha256']!=plan_digest(subset) or set(state['chunks'])!={row['path']}:raise ValueError('Cache ledger belongs to another pinned chunk')
        record=state['chunks'][row['path']]
        if record['source_sha256']!=row['sha256'] or set(record['scenes'])!=set(row['scenes']):raise ValueError('Cache scene coverage/SHA binding differs')
        index=json.loads((stage/'index.json').read_text())
        if set(index)!=set(row['scenes']):raise ValueError('Cache index omits or adds scenes')
        for key,item in record['scenes'].items():
            path=(stage/item['file']).resolve()
            if path.parent!=stage.resolve() or index[key]!=item['file'] or path.stat().st_size!=item['bytes'] or sha256_file(path)!=item['sha256']:raise ValueError('Cached scene changed: '+key)
        return stage

    def _prepare_process(self,folder,plan,row):
        if not self.python:raise ValueError('An explicit CPU converter Python is required')
        subset=single_chunk_plan(plan,row);planfile=folder/'plan.json';atomic_json(planfile,subset)
        project=Path(__file__).resolve().parents[1]
        command=[str(self.python),str(project/'tools/prepare_mirror_dataset.py'),'--plan',str(planfile),'--output',str(folder/'converted'),
            '--cache',str(folder/'raw'),'--execute','--max-chunks','1','--remove-downloaded-chunks','--download-workers',str(self.download_workers)]
        if self.local_source:command+=['--local-source',str(self.local_source)]
        if self.source_store_allocation:command+=['--source-store-allocation',str(self.source_store_allocation)]
        if self.network=='hf_mirror':command+=['--mirror-endpoint','https://hf-mirror.com']
        if self.network=='autodl_turbo':command=['bash','-c','source /etc/network_turbo >/dev/null 2>&1; exec "$@"','nps-converter',*command]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        if self.network=='hf_mirror':
            for k in ('http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy'):env.pop(k,None)
        with (folder/'prepare.log').open('ab') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=project,timeout=900)
        if completed.returncode:raise RuntimeError(f'Offline conversion exit {completed.returncode}; see {folder}/prepare.log')

    def prepare(self,plan,row):
        ident=self._id(plan,row);folder=self._entry(ident)
        with self._lock(ident):
            with file_lock(self.root/'cache.lock'):
                state=self._state();existing=state['entries'].get(ident)
                if existing and existing['status']=='ready':return folder/'converted'/plan['stage']
                # Reserve raw + converted bytes, metadata and temporary publication.
                required=math.ceil(row['bytes']*2.2)+65536
                if required>self.max_bytes:raise OSError('A single chunk exceeds the cache capacity')
                if existing:del state['entries'][ident]
                self._room(state,required,ident);folder.mkdir(exist_ok=True)
                state['entries'][ident]={'status':'building','accounted_bytes':required,'used_ns':time.time_ns(),'plan_sha256':plan_digest(plan),'chunk':row['path']}
                self._save(state)
            try:
                (self.preparer or self._prepare_process)(folder,plan,row)
                stage=self._verify(folder,plan,row);actual=self._bytes(folder)
                if actual>required:raise OSError('Converted output exceeded reserved size; cache publication refused')
                with file_lock(self.root/'cache.lock'):
                    state=self._state();state['entries'][ident].update(status='ready',accounted_bytes=actual,used_ns=time.time_ns());self._save(state)
                    self._audit({'event':'prepared','entry':ident,'chunk':row['path'],'plan_sha256':plan_digest(plan),'source_sha256':row['sha256'],'scenes':len(row['scenes']),'bytes':actual})
                return stage
            except Exception as e:
                with file_lock(self.root/'cache.lock'):
                    state=self._state();state['entries'][ident].update(status='failed',accounted_bytes=self._bytes(folder),used_ns=time.time_ns());self._save(state)
                    self._audit({'event':'failed','entry':ident,'error':type(e).__name__+': '+str(e)})
                raise

    @contextmanager
    def lease(self,plan,row):
        ident=self._id(plan,row)
        for _ in range(10):
            self.prepare(plan,row)
            with file_lock(self.root/'cache.lock'):
                state=self._state()
                if ident not in state['entries'] or state['entries'][ident]['status']!='ready':continue
                lock=self._lock(ident,shared=True,blocking=False)
                try:lock.__enter__()
                except OSError:continue  # Never wait on an entry while holding metadata lock.
                state['entries'][ident]['used_ns']=time.time_ns();self._save(state)
            try:yield self._verify(self._entry(ident),plan,row)
            finally:lock.__exit__(None,None,None)
            return
        raise RuntimeError('Cache contention prevented acquiring a prepared chunk')


class ChunkSceneStream:
    """Shuffle all chunks, shuffle scenes within each chunk, never subsample it.

Cursor + RNG state is serialized so a resumed run continues at the same next
scene/frame sample. Network prefetch never draws from the sampling RNG.
"""
    def __init__(self,cfg,stage,seed=0,index=None,limit=None,*,cache=None):
        self.cfg=cfg;self.stage=stage;self.rng=np.random.default_rng(seed)
        self.plan=json.loads(Path(cfg['source_plans'][stage]).read_text(encoding='utf-8'));self.binding=plan_digest(self.plan)
        self.sampler_binding=plan_digest({k:v for k,v in cfg.items() if k not in ('source_plans','chunk_stream','root','eval_index')})
        if self.plan['stage']!=stage or not self.plan['chunks']:raise ValueError('Wrong/empty source stage')
        self.scene_chunk={}
        for i,row in enumerate(self.plan['chunks']):
            for key in row['scenes']:
                if key in self.scene_chunk:raise ValueError('Repeated scene in pinned source plan')
                self.scene_chunk[key]=i
        if self.plan.get('selected_scene_count',len(self.scene_chunk))!=len(self.scene_chunk):raise ValueError('Source scene count differs from pinned chunk coverage')
        options=cfg['chunk_stream'];self.prefetch=int(options.get('prefetch',2))
        if not 0<=self.prefetch<=8:raise ValueError('Prefetch must be 0..8 chunks')
        self.cache=cache or BoundedChunkCache(options['cache_root'],int(options.get('cache_gib',40)*2**30),int(options.get('reserve_gib',10)*2**30),
            converter_python=options.get('converter_python'),network=options.get('network','direct'),local_source=options.get('local_source'),download_workers=options.get('download_workers',12),source_store_allocation=options.get('source_store_allocation'))
        self.pool=ThreadPoolExecutor(max_workers=max(1,self.prefetch));self.futures={};self.lease=None;self.directory=None
        self.epoch=0;self.chunk_cursor=0;self.scene_cursor=0;self.scene_order=None;self.emitted=0;self.filtered={}
        self.chunk_order=self.rng.permutation(len(self.plan['chunks'])).tolist() if stage=='train' else list(range(len(self.plan['chunks'])))
        self.fixed=json.loads(Path(index).read_text(encoding='utf-8')) if index else None
        self.keys=sorted(self.scene_chunk if self.fixed is None else [k for k,v in self.fixed.items() if v is not None])
        if self.fixed is not None:
            missing=set(self.keys)-set(self.scene_chunk)
            if missing:raise ValueError(f'{len(missing)} evaluation scenes absent from full source plan; refusing partial evaluation')
        if limit is not None:self.keys=self.keys[:limit]
        if not self.keys:raise ValueError('Empty data selection')

    def _schedule(self,indices):
        for i in indices:
            if i not in self.futures:self.futures[i]=self.pool.submit(self.cache.prepare,self.plan,self.plan['chunks'][i])
    def _release(self):
        if self.lease is not None:self.lease.__exit__(None,None,None);self.lease=None;self.directory=None
    def _open(self,i,upcoming):
        if self.directory is not None:return
        future=self.futures.pop(i,None)
        if future is not None:future.result()
        self.lease=self.cache.lease(self.plan,self.plan['chunks'][i]);self.directory=self.lease.__enter__()
        self.files=json.loads((self.directory/'index.json').read_text());self._schedule(upcoming[:self.prefetch])

    def train_example(self,step):
        from .data import load_scene,sample_indices
        if self.stage!='train':raise ValueError('Not a training stream')
        reasons={}
        for _ in range(max(100,len(self.scene_chunk)*2)):
            if self.scene_order is not None and self.scene_cursor>=len(self.scene_order):
                self._release();self.chunk_cursor+=1;self.scene_order=None;self.scene_cursor=0
            if self.chunk_cursor==len(self.chunk_order):
                self.epoch+=1;self.chunk_order=self.rng.permutation(len(self.chunk_order)).tolist();self.chunk_cursor=0
            i=self.chunk_order[self.chunk_cursor];self._open(i,self.chunk_order[self.chunk_cursor+1:])
            if self.scene_order is None:self.scene_order=self.rng.permutation(self.plan['chunks'][i]['scenes']).tolist()
            key=self.scene_order[self.scene_cursor];self.scene_cursor+=1;path=self.directory/self.files[key]
            try:
                with np.load(path,allow_pickle=False) as a:n=len(a['cameras'])
                ci,ti=sample_indices(n,step,self.cfg,self.rng)
                result=load_scene(path,{'context':ci,'target':ti},self.cfg,True,self.rng)
            except ValueError as e:
                message=str(e)
                # These are the author's declared data filters, not archive errors.
                allowed=('Not enough frames for bounded sampler','Not enough distinct interior context frames','Field of view exceeds official filter','Baseline outside allowed range:','Bad original image shape:')
                if not message.startswith(allowed):raise
                reasons[message]=reasons.get(message,0)+1;self.filtered[message]=self.filtered.get(message,0)+1;continue
            self.emitted+=1;return result
        raise RuntimeError('No valid sample after traversing source scenes: '+str(reasons))

    def test_examples(self):
        from .data import load_scene
        if self.fixed is None:raise ValueError('Fixed evaluation index required')
        groups={}
        for key in self.keys:groups.setdefault(self.scene_chunk[key],[]).append(key)
        indices=sorted(groups)
        try:
            for pos,i in enumerate(indices):
                self._open(i,indices[pos+1:])
                for key in groups[i]:yield load_scene(self.directory/self.files[key],self.fixed[key],self.cfg)
                self._release()
        finally:self._release()

    def state_dict(self):
        return copy.deepcopy({'version':1,'plan_sha256':self.binding,'sampler_sha256':self.sampler_binding,'stage':self.stage,'rng':self.rng.bit_generator.state,'epoch':self.epoch,
            'chunk_order':self.chunk_order,'chunk_cursor':self.chunk_cursor,'scene_order':self.scene_order,'scene_cursor':self.scene_cursor,'emitted':self.emitted,'filtered':self.filtered})
    def load_state_dict(self,state):
        if state['version']!=1 or state['plan_sha256']!=self.binding or state['sampler_sha256']!=self.sampler_binding or state['stage']!=self.stage:raise ValueError('Resume source plan/stage/sampling configuration differs')
        if sorted(state['chunk_order'])!=list(range(len(self.chunk_order))):raise ValueError('Invalid resumed chunk order')
        cursor=state['chunk_cursor'];scenes=state['scene_order']
        if not 0<=cursor<len(self.chunk_order):raise ValueError('Invalid resumed chunk cursor')
        if scenes is not None and sorted(scenes)!=sorted(self.plan['chunks'][state['chunk_order'][cursor]]['scenes']):raise ValueError('Invalid resumed scene order')
        if state['scene_cursor']<0 or (scenes is not None and state['scene_cursor']>len(scenes)):raise ValueError('Invalid resumed scene cursor')
        self._release()
        for key in ('epoch','chunk_order','chunk_cursor','scene_order','scene_cursor','emitted','filtered'):setattr(self,key,copy.deepcopy(state[key]))
        self.rng.bit_generator.state=copy.deepcopy(state['rng'])
    def close(self):
        self._release();self.pool.shutdown(wait=True,cancel_futures=True)
