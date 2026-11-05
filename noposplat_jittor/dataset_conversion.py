"""Resumable, framework-neutral per-scene archives and chunk commit ledger.

JPEG bytes are copied, never decoded/re-encoded. A single writer owns each
output stage. Source chunks are not deleted by this module.
"""
from pathlib import Path
import hashlib,json,os,uuid
import numpy as np


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('w',encoding='utf-8',newline='\n') as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def scene_arrays(key,cameras,images):
    cameras=np.asarray(cameras,dtype=np.float32)
    if cameras.ndim!=2 or cameras.shape[1]!=18 or len(cameras)!=len(images) or not len(images):raise ValueError('Bad camera/frame contract')
    if not np.isfinite(cameras).all():raise ValueError('Nonfinite cameras')
    buffers=[]
    for image in images:
        x=np.asarray(image)
        if x.dtype!=np.uint8 or x.ndim!=1 or not x.size:raise ValueError('Expected nonempty encoded uint8 frame bytes')
        buffers.append(x)
    offsets=np.concatenate(([0],np.cumsum([len(x) for x in buffers],dtype=np.int64)))
    return {'key':np.asarray(key),'cameras':cameras,'image_bytes':np.concatenate(buffers),'offsets':offsets}


def write_scene(path,arrays):
    path=Path(path)
    if path.exists():
        with np.load(path,allow_pickle=False) as existing:
            if set(existing.files)!=set(arrays) or any(not np.array_equal(existing[k],v) for k,v in arrays.items()):raise ValueError('Existing scene conflicts with verified source: '+str(path))
    else:
        temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with temp.open('wb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
            os.replace(temp,path)
        finally:
            if temp.exists():temp.unlink()
    return {'file':path.name,'bytes':path.stat().st_size,'sha256':sha256_file(path),'frames':len(arrays['offsets'])-1,
        'encoded_images_sha256':hashlib.sha256(arrays['image_bytes'].tobytes()).hexdigest(),
        'cameras_sha256':hashlib.sha256(arrays['cameras'].tobytes()).hexdigest()}


class ChunkConversion:
    """Publish index/status from a durable ledger; replay interrupted chunks."""
    def __init__(self,output,plan):
        self.root=Path(output).resolve();self.root.mkdir(parents=True,exist_ok=True);self.plan=plan
        self.ledger_path=self.root/'conversion_ledger.json';binding=plan_digest(plan)
        keys=[key for row in plan['chunks'] for key in row['scenes']]
        if len(keys)!=len(set(keys)):raise ValueError('Scene assigned to multiple chunks')
        self.required=set(keys)
        if self.ledger_path.exists():
            self.state=json.loads(self.ledger_path.read_text(encoding='utf-8'))
            if self.state['plan_sha256']!=binding:raise ValueError('Output belongs to a different source commit/selection')
        else:
            if (self.root/'index.json').exists():raise FileExistsError('Existing dataset has no matching ledger; use a new output')
            self.state={'version':1,'plan_sha256':binding,'source_repository':plan.get('repository'),'source_commit':plan.get('commit'),'stage':plan['stage'],'chunks':{}}
            atomic_json(self.ledger_path,self.state)
        self.publish()

    def publish(self):
        index={key:row['file'] for chunk in self.state['chunks'].values() for key,row in chunk['scenes'].items()}
        if not set(index)<=self.required:raise ValueError('Ledger contains unexpected scenes')
        complete=set(index)==self.required and len(self.state['chunks'])==len(self.plan['chunks'])
        atomic_json(self.root/'index.json',index)
        atomic_json(self.root/'conversion.json',{'status':'complete' if complete else 'partial','count':len(index),'required_scene_count':len(self.required),
            'selection':self.plan.get('selection','unspecified'),'source_scene_count':self.plan.get('source_scene_count'),
            'completed_chunks':len(self.state['chunks']),'required_chunks':len(self.plan['chunks']),'plan_sha256':self.state['plan_sha256'],
            'source_repository':self.plan.get('repository'),'source_commit':self.plan.get('commit'),
            'source_sha256':{k:v['source_sha256'] for k,v in self.state['chunks'].items()},
            'note':'Completion refers to this pinned selection; dataset/protocol equivalence is a separate check.'})

    def verify_committed(self,name):
        row=self.state['chunks'].get(name)
        if row is None:return False
        for key,record in row['scenes'].items():
            path=(self.root/record['file']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file() or path.stat().st_size!=record['bytes'] or sha256_file(path)!=record['sha256']:raise ValueError('Committed payload missing/changed: '+key)
        return True

    def convert(self,row,source,loader):
        if self.verify_committed(row['path']):return 'already_committed'
        source=Path(source)
        if source.stat().st_size!=row['bytes'] or sha256_file(source)!=row['sha256']:raise ValueError('Source size/SHA256 mismatch; refusing deserialization')
        wanted=set(row['scenes']);converted={}
        for key,cameras,images in loader(source):
            if key not in wanted:continue
            if key in converted:raise ValueError('Duplicate scene payload: '+key)
            name=hashlib.sha256(key.encode()).hexdigest()+'.npz'
            converted[key]=write_scene(self.root/name,scene_arrays(key,cameras,images))
        if set(converted)!=wanted:raise ValueError('Chunk payload does not match source scene index')
        self.state['chunks'][row['path']]={'source_sha256':row['sha256'],'scenes':converted}
        atomic_json(self.ledger_path,self.state)  # Ground truth first; index can be reconstructed.
        self.publish();return 'converted'
