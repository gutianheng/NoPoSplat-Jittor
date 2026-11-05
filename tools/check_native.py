"""Real native CUDA forward/finite-difference tests. Requires Jittor + built .so.

This is NOT a replacement renderer. It tests the actual vendored CUDA backend.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/native_check.json');args=p.parse_args()
    import jittor as jt
    from noposplat_jittor.model import Gaussians
    from noposplat_jittor.renderer import render_one
    from noposplat_jittor.rasterizer import release_states
    from noposplat_jittor.geometry_numpy import retract_pose
    jt.flags.use_cuda=1
    means=np.array([[[.1,.05,2.],[-.2,.1,2.5]]],np.float32)
    cov=np.stack([np.eye(3,dtype=np.float32)*.02]*2)[None]
    sh=np.array([[[[.8],[.2],[.1]],[[.2],[.7],[.3]]]],np.float32)
    opacity=np.array([[.5,.6]],np.float32)
    arrays=[means,cov,sh,opacity]
    params=[jt.array(x).start_grad() for x in arrays]
    gaussians=Gaussians(*params)
    k=np.array([[.8,0,.5],[0,.8,.5],[0,0,1]],np.float32);pose=np.eye(4,dtype=np.float32)
    cfg={'scale_invariant':False,'background':[0,0,0]};theta=jt.zeros(3).start_grad();rho=jt.zeros(3).start_grad()
    try:
        image,depth=render_one(gaussians,pose,k,.1,100.,(16,16),cfg,theta,rho)
        mask=jt.array(np.linspace(.1,1,16*16,dtype=np.float32).reshape(1,16,16)).stop_grad()
        loss=(image*mask).sum();grads=jt.grad(loss,params+[theta,rho]);values=[g.numpy() for g in grads]
        if not all(np.isfinite(x).all() for x in values):raise AssertionError('Nonfinite native gradient')
        baseline=image.numpy();release_states()
        def measure(pose_value=pose,mean_value=means):
            with jt.no_grad():
                g=Gaussians(jt.array(mean_value),jt.array(cov),jt.array(sh),jt.array(opacity))
                im,_=render_one(g,pose_value,k,.1,100.,(16,16),cfg)
                # Accumulate finite-difference probes in float64 on the CPU;
                # the renderer and its analytical backward remain FP32.
                result=float((im.numpy().astype(np.float64)*mask.numpy()).sum())
            release_states();return result
        # The previous 1e-3 probe crosses the original kernel's alpha=1/255
        # cutoff at pixel (y=11,x=9). A central difference across that jump
        # cannot verify the derivative within the current smooth branch.
        # Require agreement at BOTH smaller steps; keep the original tolerances.
        checks=[]
        for eps in (3e-4,1e-4):
            plus=means.copy();minus=means.copy();plus[0,0,0]+=eps;minus[0,0,0]-=eps
            fd_mean=(measure(mean_value=plus)-measure(mean_value=minus))/(2*eps)
            fd_rho=(measure(retract_pose(pose,[eps,0,0],[0,0,0]))-measure(retract_pose(pose,[-eps,0,0],[0,0,0])))/(2*eps)
            checks.append({'eps':eps,'mean_x_fd':fd_mean,'rho_x_fd':fd_rho,
                'pass':bool(np.isclose(values[0][0,0,0],fd_mean,atol=.03,rtol=.05) and np.isclose(values[-1][0],fd_rho,atol=.03,rtol=.05))})
        report={'finite':True,'nonzero_image':bool(np.max(baseline)>0),'mean_x_analytic':float(values[0][0,0,0]),'mean_x_fd':fd_mean,'rho_x_analytic':float(values[-1][0]),'rho_x_fd':fd_rho,
                'gradient_norms':[float(np.linalg.norm(v)) for v in values]}
        report['finite_difference_checks']=checks
        report['tolerances']={'atol':.03,'rtol':.05}
        report['note']='Both probes stay within the current alpha-cutoff branch; original CUDA kernels unchanged.'
        report['pass']=bool(report['nonzero_image'] and all(item['pass'] for item in checks))
    finally:release_states()
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()

