"""Compare LPIPS feature maps and input gradients on actual 256px renders."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('mode',choices=['torch','jittor'])
p.add_argument('--archive',default='reference/lpips_real_images_001.npz');args=p.parse_args()
archive=Path(args.archive)
if args.mode=='torch':
    if archive.exists():raise FileExistsError(archive)
    import torch
    from tools.export_pose_reference import load_lpips
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_lpips('weights/lpips_vgg.npz')
    with np.load('reference/re10k_pose_002.npz',allow_pickle=False) as a:x_np=a['color_initial'].copy()
    with np.load('reference/re10k_forward_001/batch.npz',allow_pickle=False) as a:y_np=a['target.image'][0].copy()
    x=torch.tensor(x_np,device='cuda',requires_grad=True);y=torch.tensor(y_np,device='cuda')
    loss=model(x,y,normalize=True).mean();loss.backward()
    # LPIPS calls net.forward() directly, bypassing PyTorch Module hooks.
    with torch.no_grad():features=model.net.forward(model.scaling_layer(2*x-1))
    np.savez(archive,x=x_np,y=y_np,loss=loss.item(),gradient=x.grad.cpu().numpy(),
             **{f'feature_{i}':v.detach().cpu().numpy() for i,v in enumerate(features)})
    print('Actual 256px PyTorch LPIPS features and gradients exported')
else:
    import jittor as jt
    from noposplat_jittor.lpips import LPIPSVGG
    from tools.check_alignment import differences
    jt.flags.use_cuda=1;jt.flags.use_tensorcore=0;jt.flags.amp_level=0
    model=LPIPSVGG('weights/lpips_vgg.npz')
    with np.load(archive,allow_pickle=False) as a:
        x=jt.array(a['x']).start_grad();y=jt.array(a['y']).stop_grad()
        loss=model(x,y).mean();gradient=jt.grad(loss,x)
        report={'loss':differences(a['loss'],loss.numpy(),1e-5,1e-4),
                'gradient':differences(a['gradient'],gradient.numpy(),1e-5,1e-4)}
        for i,f in enumerate(model._features(x)):
            report[f'feature_{i}']=differences(a[f'feature_{i}'],f.numpy(),1e-5,1e-4)
    Path('runs/lpips_real_images_alignment_001.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
