"""Framework-neutral scene archives -> identical geometry convention -> Jittor.

Original .torch chunks are converted ONCE by tools/convert_dataset.py (PyTorch).
No torch import in this module. JPEG bytes stay compressed until selected.
"""
from pathlib import Path
from io import BytesIO
import json
import numpy as np
from PIL import Image
from .geometry_numpy import cameras_from_vectors,get_fov,normalize_cameras
from .evaluation_summary import overlap_metadata


def crop_resize(images,k,shape):
    """Faithfully preserve official center_crop behavior: fx/fy ONLY are updated."""
    h,w=images.shape[-2:];oh,ow=map(int,shape)
    if oh>h or ow>w:raise ValueError('Official crop shim does not upscale inputs')
    ratio=max(oh/h,ow/w);rh,rw=round(h*ratio),round(w*ratio)
    output=[]
    for x in images:
        uint8=(x*255).clip(0,255).astype(np.uint8).transpose(1,2,0)
        resized=Image.fromarray(uint8).resize((rw,rh),Image.Resampling.LANCZOS)
        output.append((np.asarray(resized)/255).astype(np.float32).transpose(2,0,1))
    images=np.stack(output)
    top,left=(rh-oh)//2,(rw-ow)//2
    k=k.copy();k[:,0,0]*=rw/ow;k[:,1,1]*=rh/oh
    return images[:,:,top:top+oh,left:left+ow],k


def sample_indices(num_views,step,cfg,rng):
    t=min(step/max(cfg['gap_warmup'],1),1) if cfg['gap_warmup'] else 1
    low=int(cfg['initial_gap_min']+(cfg['gap_min']-cfg['initial_gap_min'])*t)
    high=int(cfg['initial_gap_max']+(cfg['gap_max']-cfg['initial_gap_max'])*t)
    high=min(num_views-1,high)
    if high<low:raise ValueError('Not enough frames for bounded sampler')
    gap=int(rng.integers(low,high+1));left=int(rng.integers(0,num_views-gap))
    target=rng.integers(left,left+gap+1,size=cfg['target_views'])
    count=cfg.get('context_views',2)
    if count<2:raise ValueError('Need at least two context views')
    if gap-1<count-2:raise ValueError('Not enough distinct interior context frames')
    extra=rng.choice(np.arange(left+1,left+gap),size=count-2,replace=False).tolist() if count>2 else []
    return np.array([left,*extra,left+gap],dtype=np.int64),target.astype(np.int64)


def evaluation_context_indices(indices,count):
    """Author additional-view rule: float32 linspace, truncate to integer."""
    indices=np.asarray(indices,dtype=np.int64)
    if count==len(indices):return indices
    return np.linspace(indices[0],indices[-1],count,dtype=np.float32).astype(np.int64)


def load_scene(path,entry,cfg,training=False,rng=None):
    with np.load(path,allow_pickle=False) as archive:
        c2w,k=cameras_from_vectors(archive['cameras'])
        if np.degrees(get_fov(k)).max()>cfg['max_fov']:raise ValueError('Field of view exceeds official filter')
        ci=np.asarray(entry['context'],np.int64);ti=np.asarray(entry['target'],np.int64)
        expected=cfg.get('context_views',2)
        if not training:ci=evaluation_context_indices(ci,expected)
        if len(ci)!=expected or expected<2:raise ValueError('Context count does not match configured model')
        if len(ti)==0:raise ValueError('No target frame')
        c2w,near,far=normalize_cameras(c2w,ci,cfg['near'],cfg['far'],cfg['baseline_min'],cfg['baseline_max'])
        image_bytes,offsets=archive['image_bytes'],archive['offsets']
        views={}
        for kind,indices in (('context',ci),('target',ti)):
            imgs=[]
            for i in indices:
                if not 0<=i<len(offsets)-1:raise ValueError(f'Frame index {i} outside archive')
                data=image_bytes[int(offsets[i]):int(offsets[i+1])].tobytes()
                image=np.asarray(Image.open(BytesIO(data)))
                if image.shape!=(*cfg['original_shape'],3):raise ValueError(f'Bad original image shape: {image.shape}')
                imgs.append((image.astype(np.float32)/255).transpose(2,0,1))
            views[kind]={'image':np.stack(imgs),'intrinsics':k[indices].copy(),'extrinsics':c2w[indices].copy(),
                         'near':np.full(len(indices),near,np.float32),'far':np.full(len(indices),far,np.float32),'index':indices}
        if training and cfg['augment'] and rng.random()>=.5:
            reflect=np.diag([-1,1,1,1]).astype(np.float32)
            for view in views.values():
                view['image']=view['image'][... ,::-1].copy()
                view['extrinsics']=reflect@view['extrinsics']@reflect
        for view in views.values():view['image'],view['intrinsics']=crop_resize(view['image'],view['intrinsics'],cfg['image_shape'])
        views['context']['image']=views['context']['image']*2-1
        views['scene']=str(archive['key'].item())
        views.update(overlap_metadata(entry))
        return views


class SceneStream:
    def __init__(self,root,stage,cfg,seed=0,index=None,limit=None):
        self.root=Path(root)/stage;self.cfg=cfg;self.rng=np.random.default_rng(seed)
        self.files=json.loads((self.root/'index.json').read_text(encoding='utf-8'))
        self.fixed=json.loads(Path(index).read_text(encoding='utf-8')) if index else None
        self.keys=sorted(k for k in self.files if self.fixed is None or self.fixed.get(k) is not None)
        if self.fixed:
            missing=set(k for k,v in self.fixed.items() if v is not None)-set(self.files)
            if missing:raise ValueError(f'{len(missing)} evaluation scenes are missing. Use an explicit subset index; never silently skip missing scenes.')
        if limit is not None:self.keys=self.keys[:limit]
        if not self.keys:raise ValueError('No available scene archives')
    def test_examples(self):
        if self.fixed is None:raise ValueError('Fixed official/subset evaluation index is required')
        for key in self.keys:
            yield load_scene(self.root/self.files[key],self.fixed[key],self.cfg)
    def train_example(self,step):
        reasons={}
        for _ in range(max(100,len(self.keys)*2)):
            key=self.keys[int(self.rng.integers(len(self.keys)))];path=self.root/self.files[key]
            try:
                with np.load(path,allow_pickle=False) as a:n=len(a['cameras'])
                ci,ti=sample_indices(n,step,self.cfg,self.rng)
                return load_scene(path,{'context':ci,'target':ti},self.cfg,True,self.rng)
            except ValueError as e:reasons[str(e)]=reasons.get(str(e),0)+1
        raise RuntimeError(f'No valid training sample found: {reasons}')
    def state_dict(self):return {'kind':'local_scene_stream','rng':self.rng.bit_generator.state}
    def load_state_dict(self,state):
        if state.get('kind')!='local_scene_stream':raise ValueError('Resume data-stream type differs')
        self.rng.bit_generator.state=state['rng']
    def close(self):pass


def make_scene_stream(cfg,stage,seed=0,index=None,limit=None):
    if cfg.get('source_plans'):
        from .chunk_stream import ChunkSceneStream
        return ChunkSceneStream(cfg,stage,seed,index,limit)
    return SceneStream(cfg['root'],stage,cfg,seed,index,limit)


def collate_numpy(examples):
    return {kind:{key:np.stack([x[kind][key] for x in examples]) for key in examples[0][kind]} for kind in ('context','target')}


def context_jittor(batch):
    import jittor as jt
    return {key:jt.array(value).stop_grad() for key,value in batch['context'].items()}

