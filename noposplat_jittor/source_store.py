"""Retain pinned original training chunks across two explicitly owned stores.

This store never evicts sources or imports Torch. Offline conversion checks the
same SHA again before deserialization. A plan fixes placement before download.
"""
from pathlib import Path,PurePosixPath
import json,os,shutil,time
from .chunk_stream import file_lock
from .dataset_conversion import atomic_json,plan_digest,sha256_file


def source_identity(plan,row):
    return plan_digest({'dataset':plan['dataset'],'repository':plan['repository'],'commit':plan['commit'],'stage':plan['stage'],
        'path':row['path'],'bytes':row['bytes'],'sha256':row['sha256']})


def allocate_sources(plans,stores):
    """Largest-first fixed placement; never shorten the requested source plans."""
    if not plans or not stores or any(not x['chunks'] for x in plans):raise ValueError('Empty source allocation')
    used={k:0 for k in stores};chunks={};descriptions=[];pending=[]
    for plan in plans:
        if plan['stage']!='train':raise ValueError('Permanent source allocation currently covers training only')
        descriptions.append({'dataset':plan['dataset'],'stage':plan['stage'],'plan_sha256':plan_digest(plan),'commit':plan['commit'],
            'scene_count':sum(len(r['scenes']) for r in plan['chunks']),'source_bytes':sum(r['bytes'] for r in plan['chunks'])})
        pending.extend((plan,row) for row in plan['chunks'])
    for plan,row in sorted(pending,key=lambda pair:(-pair[1]['bytes'],pair[0]['dataset'],pair[1]['path'])):
        ident=source_identity(plan,row)
        if ident in chunks:raise ValueError('Repeated pinned source')
        available=[name for name in stores if stores[name]['max_bytes']-used[name]>=row['bytes']]
        if not available:raise OSError('Requested plans do not fit the supplied source-store budgets; no partial allocation published')
        name=max(available,key=lambda k:stores[k]['max_bytes']-used[k]);used[name]+=row['bytes']
        chunks[ident]={'store':name,'dataset':plan['dataset'],'repository':plan['repository'],'commit':plan['commit'],'stage':plan['stage'],
            'path':row['path'],'bytes':row['bytes'],'sha256':row['sha256'],'url':row['url']}
    return {'version':1,'scope':'All pinned original TRAIN chunks; no deletion/eviction; not test data or training completion',
        'stores':stores,'plans':descriptions,'allocated_bytes':used,'chunks':chunks}


class SourceStore:
    def __init__(self,allocation):
        self.plan=json.loads(Path(allocation).read_text(encoding='utf-8')) if isinstance(allocation,(str,Path)) else allocation
        if self.plan['version']!=1:raise ValueError('Unknown source-store format')
        self.binding=plan_digest(self.plan);self.roots={}
        for name,spec in self.plan['stores'].items():
            mount=spec.get('required_mount')
            if mount and not os.path.ismount(Path(mount).resolve()):raise OSError('Required filesystem is not mounted: '+mount)
            root=Path(spec['root']).resolve();root.parent.mkdir(parents=True,exist_ok=True)
            with file_lock(root.parent/('.'+root.name+'.initialization.lock')):
                marker=root/'source_store.json';expected={'version':1,'allocation_sha256':self.binding,'store':name,'max_bytes':spec['max_bytes']}
                if marker.exists():
                    if json.loads(marker.read_text())!=expected:raise ValueError('Source store belongs to a different allocation')
                else:
                    if root.exists() and any(root.iterdir()):raise FileExistsError('Refusing unowned source-store directory')
                    root.mkdir(exist_ok=True);atomic_json(marker,expected)
                if not (root/'ledger.json').exists():atomic_json(root/'ledger.json',{'version':1,'completed':{},'inflight':{}})
            self.roots[name]=root
        for name,spec in self.plan['stores'].items():
            expected=sum(x['bytes'] for x in self.plan['chunks'].values() if x['store']==name)
            if expected!=self.plan['allocated_bytes'][name] or expected>spec['max_bytes']:raise ValueError('Allocation byte budget inconsistent')

    def _path(self,root,record):
        path=PurePosixPath(record['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in record['path'] or path.suffix!='.torch':raise ValueError('Unsafe source chunk path')
        for key in ('dataset','commit'):
            value=record[key]
            if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):raise ValueError('Unsafe source namespace')
        result=root/'chunks'/record['dataset']/record['commit']/str(path)
        if not result.resolve().is_relative_to(root):raise ValueError('Source path escaped owned root')
        return result

    def ensure(self,plan,row,*,endpoint=None,workers=12,downloader=None):
        from .verified_download import parallel_download
        ident=source_identity(plan,row)
        if ident not in self.plan['chunks']:raise ValueError('Chunk absent from complete pinned allocation')
        record=self.plan['chunks'][ident];name=record['store'];root=self.roots[name];spec=self.plan['stores'][name];path=self._path(root,record)
        if row['url']!=record.get('url'):raise ValueError('Transport URL differs from the pinned source allocation')
        with file_lock(root/'locks'/f'{ident}.lock'):
            # A complete file written before a crash is accepted only after SHA.
            if path.exists():
                if path.stat().st_size!=row['bytes'] or sha256_file(path)!=row['sha256']:raise ValueError('Persistent source changed; never deserialize it')
            else:
                with file_lock(root/'store.lock'):
                    state=json.loads((root/'ledger.json').read_text())
                    for other in list(state['inflight']):
                        if other==ident:del state['inflight'][other];continue
                        lock=file_lock(root/'locks'/f'{other}.lock',blocking=False)
                        try:lock.__enter__()
                        except OSError:continue
                        else:
                            del state['inflight'][other];lock.__exit__(None,None,None)
                    # Conservative reservations intentionally include ongoing I/O.
                    reserved=sum(x['reserved_bytes'] for x in state['inflight'].values());required=int(row['bytes']*2.2)+65536
                    if shutil.disk_usage(root).free<spec.get('reserve_bytes',0)+reserved+required:raise OSError('Source-store scratch/disk reserve unavailable')
                    state['inflight'][ident]={'reserved_bytes':required,'source_bytes':row['bytes']};atomic_json(root/'ledger.json',state)
                try:
                    (downloader or parallel_download)(row,path,endpoint=endpoint,workers=workers)
                    if path.stat().st_size!=row['bytes'] or sha256_file(path)!=row['sha256']:raise ValueError('Downloaded source failed full SHA256')
                except Exception:
                    with file_lock(root/'store.lock'):
                        state=json.loads((root/'ledger.json').read_text());state['inflight'].pop(ident,None);atomic_json(root/'ledger.json',state)
                    raise
            with file_lock(root/'store.lock'):
                state=json.loads((root/'ledger.json').read_text());state['inflight'].pop(ident,None)
                state['completed'][ident]={'bytes':row['bytes'],'sha256':row['sha256'],'path':str(path.relative_to(root)),'verified_ns':time.time_ns()}
                atomic_json(root/'ledger.json',state)
            return path
