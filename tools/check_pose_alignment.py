"""Jittor: compare the complete 100-step pose-alignment pipeline to PyTorch."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True);p.add_argument('--pose-reference',required=True)
    p.add_argument('--weights',default='weights/lpips_vgg.npz');p.add_argument('--config',default='configs/re10k.yaml')
    p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    import jittor as jt
    from noposplat_jittor.config import load_config
    from noposplat_jittor.model import Gaussians
    from noposplat_jittor.lpips import LPIPSVGG
    from noposplat_jittor.renderer import render_one
    from noposplat_jittor.rasterizer import release_states
    from noposplat_jittor.pose_alignment import align_target_poses
    from noposplat_jittor.metrics import psnr,ssim
    jt.flags.use_cuda=1;jt.flags.amp_level=0;jt.flags.use_tensorcore=0
    cfg=load_config(args.config);root=Path(args.reference);lpips=LPIPSVGG(args.weights)
    with np.load(root/'batch.npz',allow_pickle=False) as a:
        target={k.split('.',1)[1]:a[k][0].copy() for k in a.files if k.startswith('target.')}
    with np.load(root/'forward.npz',allow_pickle=False) as a:
        gs=Gaussians(*[jt.array(a[k]).stop_grad() for k in ('means','covariances','harmonics','opacities')])
    trace={};poses=align_target_poses(gs,target,cfg,lpips,trace)
    try:
        with jt.no_grad():
            images=jt.stack([render_one(gs,poses[v],target['intrinsics'][v],target['near'][v],target['far'][v],
                                      cfg['data']['image_shape'],cfg['renderer'])[0] for v in range(len(poses))])
            actual=images.numpy();perceptual=lpips(jt.array(target['image']),images).numpy().reshape(-1)
    finally:release_states()
    metrics={'psnr':psnr(target['image'],actual),'ssim':ssim(target['image'],actual),'lpips':perceptual}
    # Fixed before collecting the first reference: allow small accumulated FP32
    # and raster-branch differences over 100 SE3 steps, never paper-score gaps.
    tolerances={'psnr':.02,'ssim':.001,'lpips':.001}
    with np.load(args.pose_reference,allow_pickle=False) as ref:
        report={'pose_max_abs':float(np.max(np.abs(poses-ref['poses_final']))),
                'image_mean_abs':float(np.mean(np.abs(actual-ref['color_final']))),'metrics':{},'tolerances':tolerances}
        for key,value in metrics.items():
            error=float(np.max(np.abs(value-ref[key])))
            report['metrics'][key]={'max_abs':error,'actual':value.tolist(),'reference':ref[key].tolist(),'pass':error<=tolerances[key]}
        if 'pose_history' in ref:
            report['trajectory_diagnostics']=[{'step':i,
                'pose_max_abs':float(np.max(np.abs(trace['poses'][i]-ref['pose_history'][i]))),
                'gradient_max_abs':float(np.max(np.abs(trace['gradients'][i]-ref['gradient_history'][i]))) if i<len(trace['gradients']) else None,
                'loss_abs':abs(trace['losses'][i]-float(ref['loss_history'][i])) if i<len(trace['losses']) else None}
                for i in (0,1,2,5,10,20,50,99,100)]
    report['pass']=bool(report['pose_max_abs']<=.001 and report['image_mean_abs']<=.001 and all(x['pass'] for x in report['metrics'].values()))
    report['scope']='Fixed Gaussian inputs; original 100-step joint target-pose optimization and final image metrics'
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
