"""Jittor vs exported PyTorch arrays; fail loudly on numerical mismatch."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def differences(a,b,atol,rtol):
    if a.size==1 and b.size==1:a=a.reshape(());b=b.reshape(())
    if a.shape!=b.shape:return {'pass':False,'shape_reference':list(a.shape),'shape_jittor':list(b.shape)}
    diff=np.abs(a-b)
    return {'pass':bool(np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=rtol)),
            'shape':list(a.shape),'max_abs':float(diff.max(initial=0)),'mean_abs':float(diff.mean()),'relative_l2':float(np.linalg.norm(diff.reshape(-1))/(np.linalg.norm(a.reshape(-1))+1e-12))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True);p.add_argument('--config',default='configs/re10k.yaml')
    p.add_argument('--atol',type=float,default=3e-4);p.add_argument('--rtol',type=float,default=3e-4);p.add_argument('--grad',action='store_true');p.add_argument('--cpu',action='store_true')
    p.add_argument('--grad-relative-l2',type=float,default=.02)
    p.add_argument('--photometric',action='store_true')
    p.add_argument('--lpips-weights',default='weights/lpips_vgg.npz')
    p.add_argument('--dump-failed-gradients',action='store_true',help='Keep original and actual failing gradient arrays for diagnosis; does not change thresholds')
    p.add_argument('--math-mode',choices=['fp32','tf32'],default='fp32')
    args=p.parse_args();root=Path(args.reference)
    import jittor as jt
    from noposplat_jittor.config import load_config
    from noposplat_jittor.model import NoPoSplat
    from noposplat_jittor.weights import load_weights
    from noposplat_jittor.optim import active_parameters
    jt.flags.use_cuda=0 if args.cpu else 1
    from noposplat_jittor.precision import configure_math
    configure_math(jt,args.math_mode)
    import os
    (root/'alignment_runtime.json').write_text(json.dumps({
        'jittor':jt.__version__,'use_cuda':jt.flags.use_cuda,
        'amp_level':jt.flags.amp_level,'use_tensorcore':jt.flags.use_tensorcore,'math_mode':args.math_mode,
        'NVIDIA_TF32_OVERRIDE':os.environ.get('NVIDIA_TF32_OVERRIDE'),
        'atol':args.atol,'rtol':args.rtol,'gradient_check':args.grad,
        'scope':'Encoder numerical alignment; not dataset metrics or training reproduction.'
    },indent=2),encoding='utf-8')
    cfg=load_config(args.config);model=NoPoSplat(cfg['model'])
    load_weights(model,root/'encoder.npz',strict=True,report_path=root/'jittor_weight_report.json')
    model.eval()
    if args.grad:
        for param in model.parameters():param.start_grad()
    with np.load(root/'batch.npz',allow_pickle=False) as a:context={k.split('.',1)[1]:jt.array(a[k]).stop_grad() for k in a.files if k.startswith('context.')}
    trace={};g=model(context,trace=trace)
    if args.photometric:
        from noposplat_jittor.renderer import render_one
        from noposplat_jittor.lpips import LPIPSVGG
        with np.load(root/'batch.npz',allow_pickle=False) as a:
            target={k.split('.',1)[1]:a[k][:,0].copy() for k in a.files if k.startswith('target.')}
        lpips=LPIPSVGG(args.lpips_weights)
        color,_=render_one(g,target['extrinsics'][0],target['intrinsics'][0],target['near'][0],target['far'][0],cfg['data']['image_shape'],cfg['renderer'])
        color=color.unsqueeze(0);truth=jt.array(target['image']).stop_grad()
        from noposplat_jittor.losses import mse_loss
        mse=mse_loss(color,truth);lpips_loss=lpips(color,truth).mean();loss=mse+cfg['train']['lpips_weight']*lpips_loss
        trace.update(training_color=color,mse_loss=mse,lpips_loss=lpips_loss)
    else:loss=sum((getattr(g,k)**2).mean() for k in ('means','covariances','harmonics','opacities'))
    trace['probe_loss']=loss
    # Materialize the whole live forward once, as the production transaction
    # does; one-by-one fetches otherwise repeatedly specialize partial graphs.
    jt.sync_all()
    report={}
    with np.load(root/'forward.npz',allow_pickle=False) as a:
        for key in a.files:
            if key not in trace:raise ValueError(f'Missing Jittor trace: {key}')
            report[key]=differences(a[key],trace[key].numpy(),args.atol,args.rtol)
    if args.grad:
        params=active_parameters(model);grads=jt.grad(loss,[p for _,p in params]);jt.sync_all()
        with np.load(root/'gradients.npz',allow_pickle=False) as a:
            actual={k:g.numpy() for (k,_),g in zip(params,grads)}
            for position,key in enumerate(a.files,1):
                if key not in actual:raise ValueError(f'Missing Jittor gradient: {key}')
                item=differences(a[key],actual[key],1e-6,.01)
                if 'relative_l2' in item and np.linalg.norm(a[key].reshape(-1))>1e-12:
                    item['pass']=bool(item['pass'] and item['relative_l2']<=args.grad_relative_l2)
                report['grad:'+key]=item
                if position%200==0:print(json.dumps({'gradient_comparison':position,'total':len(a.files)}),flush=True)
    path=root/'jittor_alignment.json';path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    failures=[k for k,v in report.items() if not v['pass']]
    if args.dump_failed_gradients and args.grad:
        failed_gradient_keys=[k.removeprefix('grad:') for k in failures if k.startswith('grad:')]
        if failed_gradient_keys:
            selected=sorted(failed_gradient_keys,key=lambda k:report['grad:'+k].get('relative_l2',float('inf')),reverse=True)[:8]
            with np.load(root/'gradients.npz',allow_pickle=False) as expected:
                np.savez_compressed(root/'failed_gradient_arrays.npz',**{
                    prefix+key:array for key in selected
                    for prefix,array in (('reference:',expected[key]),('jittor:',actual[key]))})
            (root/'failed_gradient_arrays_scope.json').write_text(json.dumps({'selected':selected,'total_failed_tensors':len(failed_gradient_keys),
                'selection':'Up to 8 largest relative-L2 failures; complete strict comparisons retained in jittor_alignment.json'},indent=2))
    if args.photometric:
        from noposplat_jittor.rasterizer import release_states
        release_states()
    print(json.dumps({'checks':len(report),'failed':failures,'report':str(path)}))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
