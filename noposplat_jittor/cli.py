"""Train/test entry point, intentionally explicit instead of a Lightning shim."""
from pathlib import Path
import argparse
import csv
import json
import time
import random
import numpy as np
from .config import load_config


def dump_json(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_image(path,image):
    from PIL import Image
    Image.fromarray((np.clip(image,0,1).transpose(1,2,0)*255).astype(np.uint8)).save(path)


def train(args,cfg,model,lpips,jt):
    from .data import make_scene_stream
    stream=make_scene_stream(cfg['data'],'train',cfg['seed'])
    try:return _train_stream(args,cfg,model,lpips,jt,stream)
    finally:stream.close()


def _train_stream(args,cfg,model,lpips,jt,stream):
    from .checkpointing import save_checkpoint
    from .losses import mse_loss
    from .data import collate_numpy,context_jittor
    from .renderer import render_one
    from .rasterizer import release_states
    from .optim import AdamW,active_parameters,learning_rates
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    c=cfg['train']
    end_step=min(c['max_steps'],getattr(args,'stop_step',None) or c['max_steps'])
    optimizer=AdamW(active_parameters(model),c['lr'],c['betas'],weight_decay=c['weight_decay'])
    start_step=0
    if args.resume:
        state=jt.load(args.resume)
        for key in ('model','renderer','train','data','runtime'):
            if key not in state['config'] and key not in cfg:continue
            if state['config'].get(key)!=cfg.get(key):raise ValueError(f'Resume changes {key} configuration')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start_step=state['step']
        if 'data_stream' in state:stream.load_state_dict(state['data_stream'])
        elif cfg['data'].get('source_plans'):raise ValueError('Legacy checkpoint has no chunk cursor; refusing an inexact streaming resume')
        else:stream.rng.bit_generator.state=state['numpy_rng']
        # Dropout RNG restoration is not guaranteed across Jittor versions.
    logfile=out/'train.csv'
    if logfile.exists() and not args.resume:raise FileExistsError(f'Refusing to overwrite existing run: {logfile}')
    model.train();dump_json(out/'config.json',cfg)
    with logfile.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['step','loss','mse','lpips','lr_new','lr_pretrained','seconds','effective_batch'])
        if f.tell()==0:writer.writeheader()
        for step in range(start_step,end_step):
            tick=time.perf_counter();acc=None;values=[]
            try:
                for _ in range(c['accumulate']):
                    examples=[stream.train_example(step) for _ in range(c['batch_size'])]
                    batch=collate_numpy(examples);gaussians=model(context_jittor(batch),step)
                    colors=[];truth=[]
                    for b in range(c['batch_size']):
                        target=batch['target']
                        for v in range(target['image'].shape[1]):
                            image,_=render_one(gaussians,target['extrinsics'][b,v],target['intrinsics'][b,v],target['near'][b,v],target['far'][b,v],cfg['data']['image_shape'],cfg['renderer'],batch=b)
                            colors.append(image);truth.append(target['image'][b,v])
                    prediction=jt.stack(colors);gt=jt.array(np.stack(truth)).stop_grad()
                    mse=mse_loss(prediction,gt);perceptual=lpips(prediction,gt).mean()
                    loss=mse+c['lpips_weight']*perceptual
                    grads=jt.grad(loss,optimizer.parameters)
                    grads=[g.detach()/c['accumulate'] for g in grads]
                    acc=grads if acc is None else [a+g for a,g in zip(acc,grads)]
                    jt.sync_all()
                    values.append([float(x.item()) for x in (loss,mse,perceptual)])
                    release_states()
                averages=np.mean(values,axis=0)
                if not np.isfinite(averages).all():raise FloatingPointError('Nonfinite loss: optimizer update refused')
                rates=learning_rates(step,c);optimizer.apply(acc,rates,c['gradient_clip']);jt.sync_all()
            finally:release_states()
            averages=np.mean(values,axis=0)
            row=dict(step=step+1,loss=float(averages[0]),mse=float(averages[1]),lpips=float(averages[2]),lr_new=rates[0],lr_pretrained=rates[1],seconds=time.perf_counter()-tick,effective_batch=c['batch_size']*c['accumulate'])
            if not np.isfinite(averages).all():raise FloatingPointError(f'Nonfinite training loss: {row}')
            writer.writerow(row);f.flush()
            if (step+1)%c['log_every']==0 or step==start_step:print(json.dumps(row),flush=True)
            if (step+1)%c['checkpoint_every']==0 or step+1==end_step:
                state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'config':cfg,'numpy_rng':stream.rng.bit_generator.state,'data_stream':stream.state_dict()}
                save_checkpoint(state,out/f'step_{step+1:06d}.pkl',writer=jt.save)
                if c.get('keep_last_checkpoints') is not None:
                    from .checkpointing import retain_owned_checkpoints
                    retain_owned_checkpoints(out/f'step_{step+1:06d}.pkl',c['keep_last_checkpoints'])
    dump_json(out/'run_status.json',{'status':'training_completed' if end_step==c['max_steps'] else 'paused_at_requested_step','steps':end_step,'config':cfg,'note':'This records execution, not a claim of paper-equivalent accuracy.'})


def evaluate(args,cfg,model,lpips,jt):
    from .data import make_scene_stream
    stream=make_scene_stream(cfg['data'],'test',cfg['seed'],args.eval_index or cfg['data']['eval_index'],args.limit)
    try:return _evaluate_stream(args,cfg,model,lpips,jt,stream)
    finally:stream.close()


def _evaluate_stream(args,cfg,model,lpips,jt,stream):
    from .evaluation_summary import summarize_nvs
    from .data import collate_numpy,context_jittor
    from .renderer import render_one
    from .rasterizer import release_states
    from .model import Gaussians
    from .pose_alignment import align_target_poses
    from .metrics import psnr,ssim
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists() or (out/'per_target.csv').exists():raise FileExistsError('Use a new output directory; existing evaluation is preserved')
    model.eval();rows=[];scenes=[]
    align=cfg['test']['align_pose'] and not args.no_align
    with (out/'per_target.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=['scene','target','target_slot','overlap','overlap_group','psnr','ssim','lpips','encode_seconds','render_align_metric_seconds','align_pose'])
        writer.writeheader()
        for example in stream.test_examples():
            batch=collate_numpy([example]);tick=time.perf_counter()
            with jt.no_grad():g=model(context_jittor(batch))
            gaussians=Gaussians(*(x.detach() for x in (g.means,g.covariances,g.harmonics,g.opacities)))
            jt.sync_all();encode_time=time.perf_counter()-tick
            scene_rows=[]
            target=example['target'];alignment_start=time.perf_counter()
            poses=align_target_poses(gaussians,target,cfg,lpips) if align else target['extrinsics']
            alignment_per_target=(time.perf_counter()-alignment_start)/len(target['image'])
            for v in range(len(example['target']['image'])):
                pose=poses[v];gt_np=target['image'][v:v+1];gt=jt.array(gt_np).stop_grad()
                tick=time.perf_counter()
                try:
                    with jt.no_grad():
                        image,_=render_one(gaussians,pose,target['intrinsics'][v],target['near'][v],target['far'][v],cfg['data']['image_shape'],cfg['renderer'])
                        pred_np=image.numpy()[None]
                        perceptual=float(lpips(jt.array(pred_np),gt).mean().item())
                    row={'scene':example['scene'],'target':int(target['index'][v]),'overlap':example['overlap'],'overlap_group':example['overlap_group'],'psnr':float(psnr(gt_np,pred_np)[0]),'ssim':float(ssim(gt_np,pred_np)[0]),'lpips':perceptual,'encode_seconds':encode_time,'render_align_metric_seconds':time.perf_counter()-tick+alignment_per_target,'align_pose':align}
                    row['target_slot']=v
                    scene_rows.append(row);rows.append(row);writer.writerow(row);file.flush()
                    if cfg['test']['save_images']:
                        directory=out/'images'/example['scene'];directory.mkdir(parents=True,exist_ok=True)
                        write_image(directory/f'{row["target"]:06d}_render.png',pred_np[0]);write_image(directory/f'{row["target"]:06d}_gt.png',gt_np[0])
                finally:release_states()
            summary={'scene':example['scene'],**{k:float(np.mean([r[k] for r in scene_rows])) for k in ('psnr','ssim','lpips')},'targets':len(scene_rows)}
            scenes.append(summary);print(json.dumps(summary),flush=True)
    keys=('psnr','ssim','lpips')
    summary={'status':'evaluation_completed','scene_count':len(scenes),'target_count':len(rows),'align_pose':align,'pose_steps':cfg['test']['pose_steps'] if align else 0,
             **summarize_nvs(rows),'config':cfg,
             'timing_note':'Pose optimization is joint across target views; its wall time is divided equally among per-target timing rows.',
             'scope':'subset' if args.limit else 'specified_index','note':'Subset statistics are not full-test-set paper reproduction.'}
    dump_json(out/'results.json',summary);print(json.dumps(summary['scene_mean']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','test','smoke'])
    parser.add_argument('--config',default='configs/re10k.yaml')
    parser.add_argument('--weights',help='Converted encoder .npz or native checkpoint .pkl')
    parser.add_argument('--lpips-weights',default='weights/lpips_vgg.npz')
    parser.add_argument('--output',default='runs/experiment')
    parser.add_argument('--resume');parser.add_argument('--eval-index');parser.add_argument('--limit',type=int)
    parser.add_argument('--stop-step',type=int,help='Pause at this update without changing the learning-rate schedule')
    parser.add_argument('--no-align',action='store_true');parser.add_argument('--random-init',action='store_true')
    parser.add_argument('--cpu',action='store_true',help='Model-only smoke supports CPU; training/test require CUDA')
    args=parser.parse_args();cfg=load_config(args.config)
    import jittor as jt
    jt.flags.use_cuda=0 if args.cpu else 1
    from .precision import configure_math
    configure_math(jt,cfg.get('runtime',{}).get('math_mode','fp32'))
    jt.set_global_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    from .model import NoPoSplat
    from .weights import load_weights
    if args.mode!='smoke' and not (args.weights or args.resume or args.random_init):parser.error('Provide weights/resume, or explicitly select --random-init for a non-paper training experiment')
    if args.mode=='test' and not args.weights:parser.error('Testing requires a complete model checkpoint')
    model=NoPoSplat(cfg['model'])
    if args.weights:
        if args.weights.endswith('.npz'):load_weights(model,args.weights,strict=True)
        else:
            state=jt.load(args.weights)
            if state['config']['model']!=cfg['model']:raise ValueError('Model configuration differs from saved checkpoint')
            model.load_state_dict(state['model'])
    if args.mode=='smoke':
        from .optim import active_parameters
        h,w=cfg['data']['image_shape'];context={'image':jt.randn(1,2,3,h,w).stop_grad(),'intrinsics':jt.array(np.array([[[[.9,0,.5],[0,.9,.5],[0,0,1]]]*2],np.float32)).stop_grad()}
        model.eval()
        # Unlike PyTorch, Jittor eval() also stops parameter gradients. Keep
        # dropout disabled but explicitly restore gradients for this check.
        for parameter in model.parameters():parameter.start_grad()
        trace={};g=model(context,trace=trace)
        loss=sum((x*x).mean() for x in (g.means,g.covariances,g.harmonics,g.opacities))
        grads=jt.grad(loss,[p for _,p in active_parameters(model)])
        arrays=[x.numpy() for x in (g.means,g.covariances,g.harmonics,g.opacities)]+[x.numpy() for x in grads]
        if not all(np.isfinite(x).all() for x in arrays):raise FloatingPointError('Nonfinite model forward/gradient')
        if tuple(g.means.shape)!=(1,2*h*w,3):raise AssertionError('Gaussian count mismatch')
        directory=Path(args.output);directory.mkdir(parents=True,exist_ok=True)
        dump_json(directory/'smoke.json',{'status':'passed','loss':float(loss.item()),'gaussian_shape':list(g.means.shape),'gradient_parameter_count':len(grads),'config':cfg,'note':'Model-only shape/finite-gradient smoke; not renderer or paper reproduction.'})
        print('Model-only smoke passed');return
    if args.cpu:parser.error('Train/test require the native CUDA rasterizer')
    from .lpips import LPIPSVGG
    lpips=LPIPSVGG(args.lpips_weights)
    (train if args.mode=='train' else evaluate)(args,cfg,model,lpips,jt)


if __name__=='__main__':main()

