"""Restricted read-only tensor extraction from an already SHA-verified source.

Only the storage and tensor metadata emitted by the pinned data archive are
accepted. Every selected storage is read through ZipFile CRC/header validation.
No arbitrary pickle globals, Torch execution, JPEG re-encoding or data repair.
"""
import io,pickle,collections,zipfile
from dataclasses import dataclass
import numpy as np

@dataclass
class Storage:
    key:str
    dtype:str
    count:int

@dataclass
class Tensor:
    storage:Storage
    offset:int
    shape:tuple
    stride:tuple

def rebuild(storage,offset,shape,stride,requires_grad=False,hooks=None,*unused):
    if not isinstance(storage,Storage) or not isinstance(offset,int) or offset<0:raise ValueError('Unsupported tensor storage')
    if requires_grad or hooks:raise ValueError('Dataset tensor has unexpected gradient metadata')
    shape=tuple(shape);stride=tuple(stride)
    if len(shape)!=len(stride) or any(not isinstance(x,int) or x<0 for x in shape+stride):raise ValueError('Invalid tensor layout')
    last=offset+sum((s-1)*t for s,t in zip(shape,stride) if s)
    if last>=storage.count and np.prod(shape)>0:raise ValueError('Tensor view exceeds its declared storage')
    return Tensor(storage,offset,shape,stride)

class MetadataOnly(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):return rebuild
        dtypes={'ByteStorage':'u1','FloatStorage':'<f4','LongStorage':'<i8'}
        if module=='torch' and name in dtypes:return ('dtype',dtypes[name])
        raise pickle.UnpicklingError('Refused dataset pickle global: '+module+'.'+name)
    def persistent_load(self,pid):
        if not isinstance(pid,tuple) or len(pid)!=5 or pid[0]!='storage':raise pickle.UnpicklingError('Unsupported persistent id')
        _,dtype,key,device,count=pid
        if not isinstance(dtype,tuple) or len(dtype)!=2 or dtype[0]!='dtype':raise pickle.UnpicklingError('Unsupported storage type')
        if not isinstance(key,str) or not key.isdigit() or device!='cpu' or not isinstance(count,int) or not 0<=count<=2**31:raise pickle.UnpicklingError('Invalid storage metadata')
        return Storage(key,dtype[1],count)

def selected_scenes(path,wanted):
    wanted=set(wanted);seen=set()
    with zipfile.ZipFile(path) as archive:
        candidates=[n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(candidates)!=1:raise ValueError('Expected one source metadata pickle')
        prefix=candidates[0][:-len('data.pkl')]
        byteorder=prefix+'byteorder'
        if byteorder in archive.namelist() and archive.read(byteorder).strip()!=b'little':raise ValueError('Unsupported source byte order')
        data=MetadataOnly(io.BytesIO(archive.read(candidates[0]))).load()
        if not isinstance(data,list):raise ValueError('Expected scene list')
        def array(tensor):
            if not isinstance(tensor,Tensor):raise ValueError('Expected restricted tensor metadata')
            storage=tensor.storage;raw=archive.read(prefix+'data/'+storage.key)
            dtype=np.dtype(storage.dtype)
            if len(raw)!=storage.count*dtype.itemsize:raise ValueError('Selected storage size differs')
            flat=np.frombuffer(raw,dtype=dtype)
            return np.lib.stride_tricks.as_strided(flat[tensor.offset:],shape=tensor.shape,
                strides=tuple(s*dtype.itemsize for s in tensor.stride),writeable=False).copy()
        for scene in data:
            if not isinstance(scene,dict) or not isinstance(scene.get('key'),str):raise ValueError('Invalid scene metadata')
            key=scene['key']
            if key not in wanted:continue
            if key in seen:raise ValueError('Duplicate selected scene')
            cameras=array(scene['cameras']);images=[array(x) for x in scene['images']]
            if cameras.dtype!=np.float32 or any(x.dtype!=np.uint8 or x.ndim!=1 for x in images):raise ValueError('Unexpected camera/image dtype')
            seen.add(key);yield key,cameras,images
        if seen!=wanted:raise ValueError('Selected scene absent from source')
