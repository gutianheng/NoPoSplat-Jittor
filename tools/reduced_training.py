"""Real full-model short training: original Torch modules vs Jittor port.

Both read the same framework-neutral JPEG/camera archives and deterministic
NumPy sampling. Torch neural model, renderer, autograd and AdamW are the author's
implementation/PyTorch; no Jittor import is allowed in the reference process.
This is an explicitly reduced training protocol, not paper convergence.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,csv,hashlib,json,os,random,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from noposplat_jittor.dataset_conversion import atomic_json,sha256_file


def make_torch_model(config,official,weights):
    import torch
    from hydra import initialize_config_dir,compose
    from noposplat_jittor.weights import canonical_key
    sys.path.insert(0,str(Path(official).resolve()))
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder
    torch.manual_seed(config['seed'])
    overrides=['+experiment=re10k']
    mode=config['model'].get('intrinsics_mode','token')
    if mode=='none':overrides.append('model.encoder.backbone.intrinsics_embed_loc=none')
    if config['model'].get('gs_head')=='dpt':overrides.append('model.encoder.gs_params_head_type=dpt')
    with initialize_config_dir(version_base=None,config_dir=str(Path(official).resolve()/'config')):
        raw=compose(config_name='main',overrides=overrides)
    set_cfg(raw);typed=load_typed_root_config(raw);model,_=get_encoder(typed.model.encoder)
    with np.load(weights,allow_pickle=False) as a:
        state={};used=set()
        for name,value in model.state_dict().items():
            key=canonical_key(name)
            if key not in a:raise ValueError('Missing initial parameter '+key)
            if tuple(value.shape)!=a[key].shape:raise ValueError('Bad initial shape '+key)
            state[name]=torch.from_numpy(a[key].copy());used.add(key)
        extras=set(a.files)-used
        if extras:raise ValueError('Unexpected initialization '+str(sorted(extras)[:4]))
    model.load_state_dict(state,strict=True)
    return model


def torch_train(a,cfg):
    import torch
    from noposplat_jittor.data import make_scene_stream,collate_numpy
    from noposplat_jittor.optim import learning_rates
    from noposplat_jittor.weights import canonical_key
    from tools.export_pose_reference import load_lpips
    model=make_torch_model(cfg,a.official,a.weights).cuda().train()
    from src.model.decoder.cuda_splatting import render_cuda
    mode=cfg.get('runtime',{}).get('math_mode','fp32')
    torch.backends.cuda.matmul.allow_tf32=mode=='tf32';torch.backends.cudnn.allow_tf32=mode=='tf32'
    lpips=load_lpips(a.lpips_weights);c=cfg['train'];groups=[[],[]]
    for name,p in model.named_parameters():groups[0 if 'gaussian_param_head' in name or 'intrinsic_encoder' in name else 1].append(p)
    optimizer=torch.optim.AdamW([{'params':v} for v in groups],lr=c['lr'],betas=tuple(c['betas']),weight_decay=c['weight_decay'],eps=1e-8,foreach=False)
    stream=make_scene_stream(cfg['data'],'train',cfg['seed']);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    start=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        if state['config']!=cfg:raise ValueError('Resume configuration differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);stream.load_state_dict(state['data_stream']);start=state['step']
    end=min(c['max_steps'],a.stop_step or c['max_steps']);log=out/'train.csv'
    if log.exists() and not a.resume:raise FileExistsError(log)
    atomic_json(out/'config.json',cfg)
    fields=['step','loss','mse','lpips','lr_new','lr_pretrained','seconds','effective_batch','batch_sha256','gradient_norm']
    audit=(out/'batch_hashes.jsonl').open('a')
    try:
        with log.open('a',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=fields)
            if file.tell()==0:writer.writeheader()
            for step in range(start,end):
                tick=time.monotonic();optimizer.zero_grad(set_to_none=True);values=[];digest=hashlib.sha256()
                for _ in range(c['accumulate']):
                    examples=[stream.train_example(step) for _ in range(c['batch_size'])];batch=collate_numpy(examples)
                    micro=hashlib.sha256()
                    for kind in sorted(batch):
                        for key,value in sorted(batch[kind].items()):
                            block=kind.encode()+key.encode()+np.ascontiguousarray(value).tobytes();digest.update(block);micro.update(block)
                    audit.write(json.dumps({'microbatch_sha256':micro.hexdigest(),'scenes':[e['scene'] for e in examples]})+'\n');audit.flush()
                    context={k:torch.tensor(v,device='cuda') for k,v in batch['context'].items()}
                    gaussians=model(context,global_step=step);target={k:torch.tensor(v,device='cuda') for k,v in batch['target'].items()}
                    colors=[];truth=[]
                    for v in range(target['image'].shape[1]):
                        color,_=render_cuda(target['extrinsics'][:,v],target['intrinsics'][:,v],target['near'][:,v],target['far'][:,v],tuple(cfg['data']['image_shape']),
                            torch.zeros((c['batch_size'],3),device='cuda'),gaussians.means,gaussians.covariances,gaussians.harmonics,gaussians.opacities,scale_invariant=cfg['renderer']['scale_invariant'])
                        colors.append(color);truth.append(target['image'][:,v])
                    prediction=torch.stack(colors,1).flatten(0,1);gt=torch.stack(truth,1).flatten(0,1)
                    mse=((prediction-gt)**2).mean();perceptual=lpips(prediction,gt,normalize=True).mean();loss=mse+c['lpips_weight']*perceptual
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    (loss/c['accumulate']).backward();values.append([loss.item(),mse.item(),perceptual.item()])
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                rates=learning_rates(step,c)
                for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate
                optimizer.step();torch.cuda.synchronize();loss,mse,perceptual=np.mean(values,axis=0)
                row=dict(step=step+1,loss=float(loss),mse=float(mse),lpips=float(perceptual),lr_new=rates[0],lr_pretrained=rates[1],seconds=time.monotonic()-tick,
                    effective_batch=c['batch_size']*c['accumulate'],batch_sha256=digest.hexdigest(),gradient_norm=float(norm))
                writer.writerow(row);file.flush()
                if step==start or (step+1)%c['log_every']==0:print(json.dumps(row),flush=True)
                if (step+1)%c['checkpoint_every']==0 or step+1==end:
                    destination=out/f'step_{step+1:06d}.pt'
                    if destination.exists():raise FileExistsError(destination)
                    temporary=destination.with_suffix('.pt.tmp')
                    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'data_stream':stream.state_dict(),'step':step+1,'config':cfg},temporary)
                    os.replace(temporary,destination)
                    if c.get('keep_last_checkpoints') is not None:
                        from noposplat_jittor.checkpointing import retain_owned_checkpoints
                        retain_owned_checkpoints(destination,c['keep_last_checkpoints'])
            if not getattr(a,'skip_weight_export',False):
                np.savez(out/f'encoder_{end:06d}.npz',**{canonical_key(k):v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            atomic_json(out/'run_status.json',{'status':'training_completed' if end==c['max_steps'] else 'paused_at_requested_step','steps':end,'config':cfg,
                'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'jittor_imported':'jittor' in sys.modules,'note':__doc__})
    finally:stream.close();audit.close()


def jittor_train(a,cfg):
    import jittor as jt
    from noposplat_jittor.precision import configure_math
    from noposplat_jittor.model import NoPoSplat
    from noposplat_jittor.weights import load_weights,save_arrays
    from noposplat_jittor.lpips import LPIPSVGG
    from noposplat_jittor.cli import train
    import noposplat_jittor.data as data
    jt.flags.use_cuda=1;configure_math(jt,cfg.get('runtime',{}).get('math_mode','fp32'));jt.set_global_seed(cfg['seed'])
    model=NoPoSplat(cfg['model']);load_weights(model,a.weights,strict=True)
    # Record batch hashes without changing data or the actual CLI transaction.
    original=data.collate_numpy;rows=[];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    audit=(out/'batch_hashes.jsonl').open('a')
    def collate(examples):
        result=original(examples);digest=hashlib.sha256()
        for kind in sorted(result):
            for key,value in sorted(result[kind].items()):digest.update(kind.encode()+key.encode()+np.ascontiguousarray(value).tobytes())
        rows.append(digest.hexdigest());audit.write(json.dumps({'microbatch_sha256':digest.hexdigest(),'scenes':[e['scene'] for e in examples]})+'\n');audit.flush()
        return result
    data.collate_numpy=collate
    try:
        train(SimpleNamespace(output=a.output,resume=a.resume,stop_step=a.stop_step),cfg,model,LPIPSVGG(a.lpips_weights),jt)
        end=min(cfg['train']['max_steps'],a.stop_step or cfg['train']['max_steps'])
        if not getattr(a,'skip_weight_export',False):save_arrays(model,out/f'encoder_{end:06d}.npz')
        atomic_json(out/'runtime.json',{'jittor':jt.__version__,'amp_level':jt.flags.amp_level,'use_tensorcore':jt.flags.use_tensorcore,'torch_imported':'torch' in sys.modules})
        if 'torch' in sys.modules:raise RuntimeError('Torch imported into Jittor training')
    finally:data.collate_numpy=original;audit.close()


def main():
    from noposplat_jittor.config import load_config
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('framework',choices=['torch','jittor']);p.add_argument('--config',required=True)
    p.add_argument('--official',default='../NoPoSplat-original');p.add_argument('--weights',required=True);p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--stop-step',type=int)
    p.add_argument('--lpips-weights',default='weights/lpips_vgg.npz')
    p.add_argument('--skip-weight-export',action='store_true',help='Intermediate segment still saves a complete resumable checkpoint; defer redundant NPZ export to declared milestones')
    a=p.parse_args();cfg=load_config(a.config)
    np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    if cfg.get('runtime',{}).get('math_mode','fp32')=='fp32':os.environ['NVIDIA_TF32_OVERRIDE']='0'
    (torch_train if a.framework=='torch' else jittor_train)(a,cfg)


if __name__=='__main__':main()
