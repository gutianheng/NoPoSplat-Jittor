"""PyTorch: author's renderer/Adam/SE3 with exact LPIPS weights, all target views."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def load_lpips(path):
    import torch
    from lpips import LPIPS
    # Build the architecture without downloading the unused VGG classifier;
    # EVERY convolution and calibration weight is then overwritten from export.
    model=LPIPS(net='vgg',pnet_rand=True,pretrained=False,verbose=False).eval()
    loaded=0
    with np.load(path,allow_pickle=False) as a, torch.no_grad():
        assert a['__format__'].item()=='lpips_vgg_0.1'
        for i in range(1,6):
            for name,layer in getattr(model.net,f'slice{i}').named_children():
                if isinstance(layer,torch.nn.Conv2d):
                    for kind in ('weight','bias'):
                        value=a[f'features.{name}.{kind}'];param=getattr(layer,kind)
                        assert tuple(param.shape)==value.shape
                        param.copy_(torch.from_numpy(value));loaded+=1
        for i in range(5):
            layer=next(x for x in getattr(model,f'lin{i}').model if isinstance(x,torch.nn.Conv2d))
            value=a[f'linear.{i}'];assert tuple(layer.weight.shape)==value.shape
            layer.weight.copy_(torch.from_numpy(value));loaded+=1
    assert loaded==31,'Every VGG convolution and LPIPS calibration tensor must be loaded'
    return model.requires_grad_(False).cuda()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--weights',default='weights/lpips_vgg.npz');p.add_argument('--config',default='configs/re10k.yaml')
    p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    sys.path.insert(0,str(Path(args.official).resolve()))
    import torch
    from src.model.decoder.cuda_splatting import render_cuda
    from src.misc.cam_utils import update_pose
    from src.evaluation.metrics import compute_psnr,compute_ssim
    from noposplat_jittor.config import load_config
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=load_config(args.config);root=Path(args.reference);model=load_lpips(args.weights)
    with np.load(root/'batch.npz',allow_pickle=False) as a:
        target={k.split('.',1)[1]:torch.tensor(a[k][0],device='cuda') for k in a.files if k.startswith('target.')}
    with np.load(root/'forward.npz',allow_pickle=False) as a:
        gs=[torch.tensor(a[k],device='cuda') for k in ('means','covariances','harmonics','opacities')]
    views=len(target['image']);pose=target['extrinsics'].clone()
    theta=torch.nn.Parameter(torch.zeros((views,3),device='cuda'))
    rho=torch.nn.Parameter(torch.zeros((views,3),device='cuda'))
    optimizer=torch.optim.Adam([{'params':[theta],'lr':cfg['test']['pose_lr']},{'params':[rho],'lr':cfg['test']['pose_lr']}])
    def render(pose_value,with_grad=False):
        expanded=[x.expand(views,*x.shape[1:]) for x in gs]
        return render_cuda(pose_value,target['intrinsics'],target['near'],target['far'],tuple(cfg['data']['image_shape']),
            torch.zeros((views,3),device='cuda'),*expanded,scale_invariant=cfg['renderer']['scale_invariant'],
            cam_rot_delta=theta if with_grad else None,cam_trans_delta=rho if with_grad else None)[0]
    with torch.no_grad():initial=render(pose).cpu().numpy()
    losses=[];norms=[];pose_history=[pose.detach().cpu().numpy().copy()];gradient_history=[];start=time.perf_counter()
    for _ in range(cfg['test']['pose_steps']):
        optimizer.zero_grad();color=render(pose,True)
        loss=((color-target['image'])**2).mean()+cfg['train']['lpips_weight']*model(color,target['image'],normalize=True).mean()
        loss.backward();losses.append(loss.item());norms.append([theta.grad.norm().item(),rho.grad.norm().item()])
        gradient_history.append(np.stack([theta.grad.detach().cpu().numpy(),rho.grad.detach().cpu().numpy()]))
        with torch.no_grad():
            optimizer.step();pose=update_pose(rho,theta,pose);theta.zero_();rho.zero_()
            pose_history.append(pose.detach().cpu().numpy().copy())
    with torch.no_grad():
        final=render(pose)
        values={'psnr':compute_psnr(target['image'],final).cpu().numpy(),
                'ssim':compute_ssim(target['image'],final).cpu().numpy(),
                'lpips':model(target['image'],final,normalize=True).reshape(-1).cpu().numpy()}
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(out,color_initial=initial,color_final=final.detach().cpu().numpy(),poses_final=pose.detach().cpu().numpy(),
             loss_history=np.asarray(losses),gradient_norms=np.asarray(norms),pose_history=np.stack(pose_history),
             gradient_history=np.stack(gradient_history),**values)
    report={'status':'exported','views':views,'steps':cfg['test']['pose_steps'],'seconds':time.perf_counter()-start,
            'initial_loss':losses[0],'last_loss':losses[-1],'metrics':{k:v.tolist() for k,v in values.items()},
            'lpips':'All 31 exported convolution/calibration tensors loaded; no random weights used',
            'source':'Original render_cuda and update_pose, actual PyTorch Adam; fixed original-model Gaussian tensors'}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
