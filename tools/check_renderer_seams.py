"""Isolate camera preparation, native rasterization and photometric backward.

Reference-camera and frozen pixel-gradient cases are diagnostics ONLY; they do
not replace the production camera or the actual training loss. Strict failures
stay failures. Run export in .refvenv and each check in .venv, on CUDA.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,traceback
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
KEYS=('means','covariances','harmonics','opacities')
CASES=('legacy_equivalence','numpy_photometric','reference_camera_photometric','reference_camera_linear')


def reserve(path):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


def export(args):
    out=reserve(args.output);reserve(out.with_suffix('.json'))
    sys.path.insert(0,str(Path(args.official).resolve()))
    import torch
    import src.model.decoder.cuda_splatting as original
    from tools.export_pose_reference import load_lpips
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=Path(args.reference)
    with np.load(source/'forward.npz',allow_pickle=False) as a:values={key:a[key].copy() for key in KEYS}
    with np.load(source/'batch.npz',allow_pickle=False) as a:target={key.split('.',1)[1]:a[key][:,0].copy() for key in a.files if key.startswith('target.')}
    params=[torch.tensor(values[key],device='cuda',requires_grad=True) for key in KEYS]
    t={key:torch.tensor(value,device='cuda') for key,value in target.items()}
    camera={};rasterizer_class=original.GaussianRasterizer
    def capture(settings):
        for key,attribute in (('view','viewmatrix'),('projection','projmatrix'),('projection_raw','projmatrix_raw'),('position','campos'),('background','bg')):
            camera[key]=getattr(settings,attribute).detach().cpu().numpy().copy()
        camera.update(tanx=np.float32(settings.tanfovx),tany=np.float32(settings.tanfovy))
        return rasterizer_class(settings)
    original.GaussianRasterizer=capture
    try:
        color,depth=original.render_cuda(t['extrinsics'],t['intrinsics'],t['near'],t['far'],tuple(target['image'].shape[-2:]),torch.zeros((1,3),device='cuda'),*params,scale_invariant=True)
    finally:original.GaussianRasterizer=rasterizer_class
    color.retain_grad();perceptual=load_lpips(args.lpips_weights)
    mse=((color-t['image'])**2).mean();lpips=perceptual(color,t['image'],normalize=True).mean();loss=mse+.05*lpips
    loss.backward();torch.cuda.synchronize()
    scale=1/t['near'];camera['scale']=scale[0].item();camera['scale_squared']=(scale**2)[0].item()
    data={**values,**{'target.'+key:value for key,value in target.items()},**{'camera.'+key:np.asarray(value,np.float32) for key,value in camera.items()}}
    data.update(__format__=np.asarray('dense_renderer_seams_v1'),color=color.detach().cpu().numpy(),depth=depth.detach().cpu().numpy(),pixel_gradient=color.grad.detach().cpu().numpy(),
        mse=np.asarray(mse.detach().cpu()),lpips=np.asarray(lpips.detach().cpu()),loss=np.asarray(loss.detach().cpu()))
    data.update({'gradient.'+key:param.grad.detach().cpu().numpy() for key,param in zip(KEYS,params)})
    for key,value in data.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise FloatingPointError('Nonfinite reference: '+key)
    np.savez(out,**data)
    metadata={'status':'reference_export_completed','scope':__doc__,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'gaussians':int(params[0].shape[1]),'image_shape':list(target['image'].shape),'loss':float(loss.item()),
        'original_renderer_sha256':hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32}
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)


def check(args):
    import jittor as jt
    from noposplat_jittor.model import Gaussians
    from noposplat_jittor.renderer import render_one,render_with_camera
    from noposplat_jittor.rasterizer import release_states
    from noposplat_jittor.lpips import LPIPSVGG
    from tools.check_alignment import differences
    jt.flags.use_cuda=1;jt.flags.amp_level=0;jt.flags.use_tensorcore=0
    if args.workspace_ratio is not None:
        from jittor import compile_extern
        compile_extern.cudnn.set_max_workspace_ratio(args.workspace_ratio)
    with np.load(args.reference,allow_pickle=False) as a:data={key:a[key].copy() for key in a.files}
    if data['__format__'].item()!='dense_renderer_seams_v1':raise ValueError('Wrong fixture format')
    camera={key.split('.',1)[1]:value for key,value in data.items() if key.startswith('camera.')}
    target={key.split('.',1)[1]:value[0] for key,value in data.items() if key.startswith('target.')}
    shape=tuple(target['image'].shape[-2:]);cfg={'scale_invariant':True,'background':camera['background'].tolist()}
    g=Gaussians(*[jt.array(data[key]).start_grad() for key in KEYS]);params=[getattr(g,key) for key in KEYS]
    normal=lambda:render_one(g,target['extrinsics'],target['intrinsics'],target['near'],target['far'],shape,cfg)
    checks={};cleanup_error=None
    try:
        if args.case=='legacy_equivalence':
            path=Path(args.legacy)
            if not path.is_file():raise FileNotFoundError(path)
            spec=importlib.util.spec_from_file_location('noposplat_jittor._renderer_validation_snapshot',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            old,old_depth=module.render_one(g,target['extrinsics'],target['intrinsics'],target['near'],target['far'],shape,cfg)
            new,new_depth=normal()
            checks={'color':differences(old.numpy(),new.numpy(),0,0),'depth':differences(old_depth.numpy(),new_depth.numpy(),0,0)}
        else:
            image,_=normal() if args.case=='numpy_photometric' else render_with_camera(g,camera,shape)
            color=image.unsqueeze(0);checks['color']=differences(data['color'],color.numpy(),3e-4,3e-4)
            if args.case=='reference_camera_linear':
                loss=(color*jt.array(data['pixel_gradient']).stop_grad()).sum()
            else:
                lpips=LPIPSVGG(args.lpips_weights);truth=jt.array(target['image'][None]).stop_grad()
                if args.stable_mse:
                    from noposplat_jittor.losses import mse_loss
                    mse=mse_loss(color,truth)
                else:mse=((color-truth)**2).mean()
                perceptual=lpips(color,truth).mean();loss=mse+.05*perceptual
                for key,value in zip(('mse','lpips','loss'),(mse,perceptual,loss)):checks[key]=differences(data[key],value.numpy(),3e-4,3e-4)
            gradients=jt.grad(loss,params)
            for key,gradient in zip(KEYS,gradients):
                item=differences(data['gradient.'+key],gradient.numpy(),1e-6,.01)
                item['pass']=bool(item['pass'] and item['relative_l2']<=.02);checks['gradient.'+key]=item
    except Exception:
        try:release_states()
        except Exception as caught:print('CLEANUP_ERROR: '+repr(caught),file=sys.stderr)
        raise
    release_states()
    return {'status':'diagnostic_completed','scope':__doc__,'case':args.case,'checks':checks,'pass':all(x['pass'] for x in checks.values()),
        'runtime':{'jittor':jt.__version__,'use_tensorcore':jt.flags.use_tensorcore,'NVIDIA_TF32_OVERRIDE':os.environ.get('NVIDIA_TF32_OVERRIDE')},
        'mse_reduction':'FP64 accumulation of FP32 squares' if args.stable_mse else 'native FP32',
        'workspace_ratio_override':args.workspace_ratio,
        'thresholds':{'color_atol':3e-4,'color_rtol':3e-4,'gradient_atol':1e-6,'gradient_rtol':.01,'gradient_relative_l2':.02,'legacy_color_depth':'bitwise equality'}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('export','check'));p.add_argument('--official');p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--case',choices=CASES);p.add_argument('--lpips-weights',default='weights/lpips_vgg.npz');p.add_argument('--legacy',default='validation/frozen_sources/renderer_before_2026-11-02.py')
    p.add_argument('--workspace-ratio',type=float);p.add_argument('--stable-mse',action='store_true');args=p.parse_args()
    if args.workspace_ratio is not None and not 0<=args.workspace_ratio<=1:p.error('workspace ratio must be in [0,1]')
    if args.mode=='export':
        if not args.official:p.error('export needs --official')
        export(args);return
    if not args.case:p.error('check needs --case')
    out=reserve(args.output)
    try:report=check(args)
    except Exception as caught:
        out.write_text(json.dumps({'status':'execution_error','case':args.case,'error':repr(caught),'traceback':traceback.format_exc()},indent=2));raise
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps({'case':args.case,'pass':report['pass'],'checks':report['checks'],'report':str(out)}),flush=True)
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
