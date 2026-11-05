"""Author NoPoSplat encoder/renderer/pose optimizer on the identical fixed subset.

Shared framework-neutral JPEG/camera preparation and result journaling; neural
and renderer computation is original PyTorch code. This independently checks
the Jittor model/rendering/target-pose-optimization implementation.
"""
from pathlib import Path
import argparse,csv,json,os,sys,time,uuid
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    p.add_argument('--official',default='../NoPoSplat-original');p.add_argument('--lpips-weights',default='weights/lpips_vgg.npz')
    p.add_argument('--resume-evaluation',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--save-scene-limit',type=int,default=12)
    p.add_argument('--save-extra-scenes',help='JSON list of predeclared scene IDs, independent of result quality')
    p.add_argument('--weights-role',choices=['specified','official_pretrained','own_reduced_training'],default='specified');a=p.parse_args()
    from noposplat_jittor.config import load_config
    from noposplat_jittor.data import make_scene_stream,collate_numpy
    from noposplat_jittor.dataset_conversion import atomic_json,plan_digest,sha256_file
    from noposplat_jittor.evaluation_journal import EvaluationJournal
    from noposplat_jittor.evaluation_summary import summarize_nvs
    from noposplat_jittor.cli import write_image
    from tools.reduced_training import make_torch_model
    from tools.export_pose_reference import load_lpips
    cfg=load_config(a.config)
    if cfg.get('runtime',{}).get('math_mode','fp32')=='fp32':os.environ['NVIDIA_TF32_OVERRIDE']='0'
    import torch
    model=make_torch_model(cfg,a.official,a.weights).cuda().eval().requires_grad_(False)
    from src.model.decoder.cuda_splatting import render_cuda
    from src.misc.cam_utils import update_pose
    from src.evaluation.metrics import compute_psnr,compute_ssim
    torch.backends.cuda.matmul.allow_tf32=cfg.get('runtime',{}).get('math_mode')=='tf32';torch.backends.cudnn.allow_tf32=torch.backends.cuda.matmul.allow_tf32
    lpips=load_lpips(a.lpips_weights);stream=make_scene_stream(cfg['data'],'test',cfg['seed'],cfg['data']['eval_index'],a.limit)
    out=Path(a.output);binding={'framework':'author_torch','weights_role':a.weights_role,'weights_sha256':sha256_file(a.weights),'lpips_sha256':sha256_file(a.lpips_weights),
        'config':cfg,'index_sha256':plan_digest(stream.fixed),'limit':a.limit}
    journal=EvaluationJournal(out,binding,stream.keys,resume=a.resume_evaluation);all_keys=list(stream.keys);stream.keys=[k for k in all_keys if k not in journal.completed]
    from noposplat_jittor.evaluation_summary import overlap_metadata
    save_keys=set(all_keys[:a.save_scene_limit])
    if a.save_scene_limit:
        for group in ('small','medium','large'):
            candidates=[k for k in all_keys if overlap_metadata(stream.fixed[k])['overlap_group']==group]
            if candidates:save_keys.add(min(candidates))
    if a.save_extra_scenes:
        extra=json.loads(Path(a.save_extra_scenes).read_text())
        if not isinstance(extra,list) or any(not isinstance(k,str) for k in extra) or not set(extra)<=set(all_keys):raise ValueError('Extra qualitative scene IDs are not in the fixed evaluation index')
        save_keys.update(extra)
    atomic_json(out/'runtime.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'allow_tf32_cudnn':torch.backends.cudnn.allow_tf32,'jittor_imported':'jittor' in sys.modules,'saved_scene_keys':sorted(save_keys),'image_selection':'first fixed-index scene IDs plus smallest ID per overlap group; never selected by metrics'})
    if 'jittor' in sys.modules:raise RuntimeError('Jittor imported into Torch reference')
    started=time.monotonic()
    try:
        for example in stream.test_examples():
            if (out/'PAUSE_REQUESTED').exists():break
            batch=collate_numpy([example]);context={k:torch.tensor(v,device='cuda') for k,v in batch['context'].items()};torch.cuda.synchronize();tick=time.monotonic()
            with torch.no_grad():g=model(context)
            torch.cuda.synchronize();encode_seconds=time.monotonic()-tick
            target={k:torch.tensor(v,device='cuda') for k,v in example['target'].items()};views=len(target['image']);pose=target['extrinsics'].clone()
            parameters=[x.expand(views,*x.shape[1:]) for x in (g.means,g.covariances,g.harmonics,g.opacities)]
            theta=torch.nn.Parameter(torch.zeros((views,3),device='cuda'));rho=torch.nn.Parameter(torch.zeros((views,3),device='cuda'))
            opt=torch.optim.Adam([theta,rho],lr=cfg['test']['pose_lr'],foreach=False);tick=time.monotonic()
            def render(grad):
                return render_cuda(pose,target['intrinsics'],target['near'],target['far'],tuple(cfg['data']['image_shape']),torch.zeros((views,3),device='cuda'),
                    *parameters,scale_invariant=cfg['renderer']['scale_invariant'],cam_rot_delta=theta if grad else None,cam_trans_delta=rho if grad else None)[0]
            loss=None
            if cfg['test']['align_pose']:
                for _ in range(cfg['test']['pose_steps']):
                    opt.zero_grad(set_to_none=True);color=render(True);loss=((color-target['image'])**2).mean()+cfg['train']['lpips_weight']*lpips(color,target['image'],normalize=True).mean()
                    loss.backward();opt.step()
                    with torch.no_grad():pose=update_pose(rho,theta,pose);theta.zero_();rho.zero_()
            with torch.no_grad():
                color=render(False);metrics={'psnr':compute_psnr(target['image'],color).cpu().numpy(),'ssim':compute_ssim(target['image'],color).cpu().numpy(),
                    'lpips':lpips(target['image'],color,normalize=True).reshape(-1).cpu().numpy()};images=color.cpu().numpy()
            seconds=(time.monotonic()-tick)/views;rows=[];attempt=out/'attempts'/(plan_digest({'scene':example['scene']})[:16]+'-'+uuid.uuid4().hex);attempt.mkdir(parents=True)
            if example['scene'] in save_keys:
                for i,image in enumerate(example['context']['image']):write_image(attempt/f'context_{i}.png',image*.5+.5)
                atomic_json(attempt/'context_frames.json',{'context_indices':example['context']['index'].tolist(),'scene':example['scene']})
            for i,index in enumerate(example['target']['index']):
                row={'scene':example['scene'],'target':int(index),'target_slot':i,'overlap':example['overlap'],'overlap_group':example['overlap_group'],
                    **{k:float(v[i]) for k,v in metrics.items()},'encode_seconds':encode_seconds,'render_align_metric_seconds':seconds,'align_pose':cfg['test']['align_pose']};rows.append(row)
                if example['scene'] in save_keys:
                    write_image(attempt/f'{index:06d}_render.png',images[i]);write_image(attempt/f'{index:06d}_gt.png',example['target']['image'][i])
            journal.commit(example['scene'],rows,example['target']['index'],attempt.relative_to(out));print(json.dumps(journal.progress('running')),flush=True)
            del g,parameters,theta,rho,opt,color,loss
        progress=journal.progress('completed' if len(journal.completed)==len(all_keys) else 'paused')
        if progress['complete']:
            rows=journal.rows();atomic_json(out/'results.json',{'status':'evaluation_completed','scene_count':len(all_keys),'target_count':len(rows),'binding':binding,
                **summarize_nvs(rows),'elapsed_this_invocation_seconds':time.monotonic()-started,'scope':'Identical explicit fixed subset; not paper population mean'})
            with (out/'per_target.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(json.dumps(progress),flush=True)
    except Exception as e:
        atomic_json(out/'failure.json',{'status':'failed','error':type(e).__name__+': '+str(e)});raise
    finally:stream.close()


if __name__=='__main__':main()
