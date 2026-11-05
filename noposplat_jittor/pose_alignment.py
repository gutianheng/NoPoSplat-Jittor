"""Joint target-view Adam pose alignment, matching ModelWrapper.test_step_align."""
import numpy as np
import jittor as jt
from .geometry_numpy import retract_pose
from .optim import AdamW
from .renderer import render_one
from .rasterizer import release_states
from .losses import mse_loss


def align_target_poses(gaussians,target,cfg,lpips,trace=None):
    """Keep geometry fixed; optimize all target camera deltas as one batch."""
    poses=target['extrinsics'].copy()
    views=len(target['image'])
    gt=jt.array(np.asarray(target['image'],np.float32)).stop_grad()
    theta=jt.zeros((views,3)).start_grad()
    rho=jt.zeros((views,3)).start_grad()
    optimizer=AdamW([('theta',theta),('rho',rho)],cfg['test']['pose_lr'],(.9,.999),weight_decay=0.)
    if trace is not None:trace.update(poses=[poses.copy()],gradients=[],losses=[])
    try:
        for _ in range(cfg['test']['pose_steps']):
            images=[render_one(gaussians,poses[v],target['intrinsics'][v],target['near'][v],target['far'][v],
                     cfg['data']['image_shape'],cfg['renderer'],theta[v],rho[v])[0] for v in range(views)]
            prediction=jt.stack(images)
            loss=mse_loss(prediction,gt)+cfg['train']['lpips_weight']*lpips(prediction,gt).mean()
            gradients=jt.grad(loss,[theta,rho])
            optimizer.apply(gradients)
            jt.sync_all()
            if trace is not None:
                trace['gradients'].append(np.stack([g.numpy().copy() for g in gradients]))
                trace['losses'].append(float(loss.item()))
            delta_theta,delta_rho=theta.numpy(),rho.numpy()
            poses=np.stack([retract_pose(p,dr,dt) for p,dr,dt in zip(poses,delta_rho,delta_theta)])
            if trace is not None:trace['poses'].append(poses.copy())
            theta.assign(jt.zeros_like(theta));rho.assign(jt.zeros_like(rho))
            theta.start_grad();rho.start_grad()
            release_states()
    finally:release_states()
    return poses
