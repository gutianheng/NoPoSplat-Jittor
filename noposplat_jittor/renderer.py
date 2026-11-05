"""Camera preprocessing is static; gradients flow through the CUDA Function.

Target poses are c2w. Camera optimization uses explicit theta/rho Jacobians,
followed by an SE(3) retraction outside the graph, as the original implementation.
"""
import numpy as np
import jittor as jt
from .geometry_numpy import projection_matrices
from .rasterizer import Rasterize


def render_one(gaussians,c2w,k,near,far,shape,cfg,theta=None,rho=None,batch=0):
    c2w=np.asarray(c2w,dtype=np.float32).copy()
    # Match the author's tensor arithmetic: the reciprocal, its square and
    # scaled bounds are rounded in FP32, not first evaluated in Python FP64.
    scale=np.float32(1.)/np.float32(near) if cfg['scale_invariant'] else np.float32(1.)
    c2w[:3,3]*=scale
    view,proj,raw,tx,ty=projection_matrices(c2w,k,np.float32(near)*scale,np.float32(far)*scale)
    camera={'scale':scale,'scale_squared':np.float32(scale*scale),'view':view,'projection':proj,
            'projection_raw':raw,'tanx':tx,'tany':ty,'position':c2w[:3,3],'background':cfg['background']}
    return render_with_camera(gaussians,camera,shape,theta,rho,batch)


def render_with_camera(gaussians,camera,shape,theta=None,rho=None,batch=0):
    """Native render at explicit static camera values.

    Production render_one supplies its normal NumPy camera. Diagnostic tools may
    supply original-camera fixtures to isolate the rasterizer from preprocessing;
    such fixtures are never substituted into production training/evaluation.
    """
    # These values change with the sampled baseline/near plane. Runtime tensor
    # inputs avoid specializing an otherwise identical CUDA graph per camera.
    scales=jt.array(np.asarray([camera['scale'],camera['scale_squared']],np.float32)).stop_grad()
    means=gaussians.means[batch]*scales[0]
    cov=gaussians.covariances[batch]*scales[1]
    cov=jt.stack([cov[:,i,j] for i,j in ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))],dim=-1)
    sh=gaussians.harmonics[batch].permute(0,2,1)
    degree=round(sh.shape[1]**.5)-1
    const=lambda x:jt.array(np.asarray(x,np.float32)).stop_grad()
    theta=jt.zeros(3).stop_grad() if theta is None else theta
    rho=jt.zeros(3).stop_grad() if rho is None else rho
    return Rasterize(int(shape[0]),int(shape[1]),float(camera['tanx']),float(camera['tany']),degree)(means,cov,sh,gaussians.opacities[batch].reshape(-1,1),const(camera['view']),const(camera['projection']),const(camera['projection_raw']),const(camera['position']),const(camera['background']),theta,rho)
