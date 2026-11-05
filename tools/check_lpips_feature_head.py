"""Freeze identical VGG features to isolate LPIPS normalization/calibration math.

This diagnostic bypasses feature extraction only during the test. It never
changes the production LPIPS implementation or its pretrained weights.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('export','check'));p.add_argument('--images',required=True);p.add_argument('--reference',required=True);p.add_argument('--output');args=p.parse_args();path=Path(args.reference)
    if args.mode=='export':
        if path.exists():raise FileExistsError(path)
        import torch
        from tools.export_pose_reference import load_lpips
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model=load_lpips('weights/lpips_vgg.npz')
        with np.load(args.images,allow_pickle=False) as a:x=torch.tensor(a['color'],device='cuda');y=torch.tensor(a['target.image'],device='cuda')
        with torch.no_grad():fx=model.net.forward(model.scaling_layer(2*x-1));fy=model.net.forward(model.scaling_layer(2*y-1))
        leaves=[v.detach().clone().requires_grad_(True) for v in fx];counter=[0];original=model.net.forward
        def fixed(_):
            counter[0]+=1
            if counter[0]>2:raise AssertionError('Unexpected VGG invocation')
            return leaves if counter[0]==1 else fy
        model.net.forward=fixed
        try:loss=model(x,y,normalize=True).mean();loss.backward()
        finally:model.net.forward=original
        data={'loss':np.asarray(loss.detach().cpu())}
        for i in range(5):data.update({f'x{i}':leaves[i].detach().cpu().numpy(),f'y{i}':fy[i].detach().cpu().numpy(),f'gradient{i}':leaves[i].grad.cpu().numpy()})
        path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**data);print('Exported five actual VGG feature levels and LPIPS head gradients',flush=True);return
    if not args.output:p.error('check needs --output')
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    import jittor as jt
    from noposplat_jittor.lpips import LPIPSVGG
    from tools.check_alignment import differences
    jt.flags.use_cuda=1;jt.flags.use_tensorcore=0;jt.flags.amp_level=0;model=LPIPSVGG('weights/lpips_vgg.npz')
    with np.load(path,allow_pickle=False) as a:data={key:a[key].copy() for key in a.files}
    xs=[jt.array(data[f'x{i}']).start_grad() for i in range(5)];ys=[jt.array(data[f'y{i}']).stop_grad() for i in range(5)];counter=[0];original=model._features
    def fixed(_):
        counter[0]+=1
        if counter[0]>2:raise AssertionError('Unexpected VGG invocation')
        return xs if counter[0]==1 else ys
    model._features=fixed
    try:
        dummy=jt.zeros((1,3,8,8)).stop_grad();loss=model(dummy,dummy).mean();gradients=jt.grad(loss,xs)
        checks={'loss':differences(data['loss'],loss.numpy(),1e-5,1e-4)}
        for i,g in enumerate(gradients):checks[f'feature_gradient_{i}']=differences(data[f'gradient{i}'],g.numpy(),1e-10,1e-4)
    finally:model._features=original
    report={'scope':__doc__,'checks':checks,'pass':all(x['pass'] for x in checks.values())}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
