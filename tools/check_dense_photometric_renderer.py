"""Compare native renderer + perceptual backward at identical dense Gaussians.

No encoder executes: this isolates downstream numerical behavior, not the full
end-to-end network. Uses the original photometric gradient from the same export.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.reference);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    import jittor as jt
    from noposplat_jittor.model import Gaussians
    from noposplat_jittor.config import load_config
    from noposplat_jittor.renderer import render_one
    from noposplat_jittor.lpips import LPIPSVGG
    from noposplat_jittor.rasterizer import release_states
    from tools.check_alignment import differences
    jt.flags.use_cuda=1;jt.flags.use_tensorcore=0;jt.flags.amp_level=0;cfg=load_config('configs/re10k.yaml')
    keys=('means','covariances','harmonics','opacities')
    with np.load(root/'forward.npz',allow_pickle=False) as a:
        g=Gaussians(*[jt.array(a[key]).start_grad() for key in keys])
        expected={key:a[key].copy() for key in ('training_color','mse_loss','lpips_loss','probe_loss')}
    with np.load(root/'batch.npz',allow_pickle=False) as a:
        target={k.split('.',1)[1]:a[k][0,0].copy() for k in a.files if k.startswith('target.')}
    lpips=LPIPSVGG('weights/lpips_vgg.npz');truth=jt.array(target['image'][None]).stop_grad()
    color,_=render_one(g,target['extrinsics'],target['intrinsics'],target['near'],target['far'],cfg['data']['image_shape'],cfg['renderer']);color=color.unsqueeze(0)
    from noposplat_jittor.losses import mse_loss
    mse=mse_loss(color,truth);perceptual=lpips(color,truth).mean();loss=mse+.05*perceptual
    jt.sync_all()
    checks={key:differences(expected[key],actual.numpy(),3e-4,3e-4) for key,actual in zip(expected,(color,mse,perceptual,loss))}
    gradients=jt.grad(loss,[getattr(g,key) for key in keys]);jt.sync_all()
    with np.load(root/'gaussian_gradients.npz',allow_pickle=False) as a:
        for key,gradient in zip(keys,gradients):
            item=differences(a[key],gradient.numpy(),1e-6,.01);item['pass']=bool(item['pass'] and item['relative_l2']<=.02);checks['grad:'+key]=item
    release_states();report={'scope':__doc__,'checks':checks,'pass':all(x['pass'] for x in checks.values())}
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
