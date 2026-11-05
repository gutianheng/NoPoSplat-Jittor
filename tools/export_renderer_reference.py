"""PyTorch environment: export the original pose-aware CUDA renderer fixture."""
from pathlib import Path
import argparse
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--official',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    sys.path.insert(0,str(Path(args.official).resolve()))
    import torch
    from src.model.decoder.cuda_splatting import render_cuda
    torch.backends.cuda.matmul.allow_tf32=False
    rng=np.random.default_rng(52);n=5
    means=rng.uniform(-.3,.3,(1,n,3)).astype(np.float32);means[...,2]+=2
    cov=np.broadcast_to(np.eye(3,dtype=np.float32)*.01,(1,n,3,3)).copy()
    sh=rng.uniform(-.5,.5,(1,n,3,25)).astype(np.float32)
    opa=rng.uniform(.2,.7,(1,n)).astype(np.float32)
    data={k:v for k,v in zip(('means','covariances','harmonics','opacities'),(means,cov,sh,opa))}
    tensors=[torch.tensor(x,device='cuda',requires_grad=True) for x in (means,cov,sh,opa)]
    pose=np.eye(4,dtype=np.float32)[None];k=np.array([[[.8,0,.5],[0,.8,.5],[0,0,1]]],np.float32)
    theta=torch.zeros((1,3),device='cuda',requires_grad=True);rho=torch.zeros((1,3),device='cuda',requires_grad=True)
    color,depth=render_cuda(torch.tensor(pose,device='cuda'),torch.tensor(k,device='cuda'),torch.tensor([.1],device='cuda'),torch.tensor([100.],device='cuda'),(16,16),torch.zeros((1,3),device='cuda'),*tensors,scale_invariant=False,cam_rot_delta=theta,cam_trans_delta=rho)
    mask=torch.linspace(.1,1,256,device='cuda').reshape(1,1,16,16)
    loss=(color*mask).sum()+.01*(depth*mask[:,0]).sum();loss.backward()
    data.update(color=color.detach().cpu().numpy(),depth=depth.detach().cpu().numpy(),pose=pose,intrinsics=k)
    data.update({f'grad_{key}':x.grad.cpu().numpy() for key,x in zip(('means','covariances','harmonics','opacities','theta','rho'),tensors+[theta,rho])})
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);np.savez(out,**data)
    print('Original CUDA renderer forward + Gaussian/camera gradients exported.')


if __name__=='__main__':main()

