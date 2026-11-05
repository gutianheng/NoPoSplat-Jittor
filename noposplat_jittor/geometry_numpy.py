"""Static camera/data geometry; no framework import and no training graph."""
import numpy as np


def cameras_from_vectors(poses):
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 18:
        raise ValueError("Expected cameras [N,18]")
    k = np.broadcast_to(np.eye(3, dtype=np.float32), (len(poses), 3, 3)).copy()
    k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2] = poses[:, :4].T
    w2c = np.broadcast_to(np.eye(4, dtype=np.float32), (len(poses), 4, 4)).copy()
    w2c[:, :3] = poses[:, 6:].reshape(-1, 3, 4)
    return np.linalg.inv(w2c).astype(np.float32), k


def get_fov(k):
    inv = np.linalg.inv(k)
    rays = []
    for v in ([0, .5, 1], [1, .5, 1], [.5, 0, 1], [.5, 1, 1]):
        x = inv @ np.asarray(v, dtype=np.float32)
        rays.append(x / np.linalg.norm(x, axis=-1, keepdims=True))
    return np.stack([np.arccos(np.clip((rays[a]*rays[b]).sum(-1), -1, 1)) for a,b in ((0,1),(2,3))], -1)


def normalize_cameras(c2w, context_indices, near=.1, far=100., baseline_min=.001, baseline_max=1e10):
    c2w = np.asarray(c2w, dtype=np.float32).copy()
    baseline = np.linalg.norm(c2w[context_indices[0], :3, 3] - c2w[context_indices[-1], :3, 3])
    if not baseline_min <= baseline <= baseline_max:
        raise ValueError(f"Baseline outside allowed range: {baseline}")
    c2w[:, :3, 3] /= baseline
    c2w = np.linalg.inv(c2w[context_indices[0]])[None] @ c2w
    return c2w.astype(np.float32), float(near/baseline), float(far/baseline)


def quaternion_xyzw_to_matrix(q, eps=1e-8):
    q = np.asarray(q)
    i,j,k,r = np.moveaxis(q,-1,0)
    s = 2 / ((q*q).sum(-1) + eps)
    return np.stack((1-s*(j*j+k*k),s*(i*j-k*r),s*(i*k+j*r),s*(i*j+k*r),1-s*(i*i+k*k),s*(j*k-i*r),s*(i*k-j*r),s*(j*k+i*r),1-s*(i*i+j*j)), -1).reshape(*q.shape[:-1],3,3)


def skew(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=np.float64)


def retract_pose(c2w, rho, theta):
    """Author convention: exp([rho,theta]) LEFT-multiplies w2c."""
    theta = np.asarray(theta, dtype=np.float64)
    t = np.linalg.norm(theta)
    K = skew(theta)
    if t < 1e-5:
        A = 1-t*t/6+t**4/120
        B = .5-t*t/24+t**4/720
        C = 1/6-t*t/120+t**4/5040
    else:
        A,B,C = np.sin(t)/t,(1-np.cos(t))/t**2,(t-np.sin(t))/t**3
    R = np.eye(3)+A*K+B*(K@K)
    V = np.eye(3)+B*K+C*(K@K)
    delta = np.eye(4)
    delta[:3,:3], delta[:3,3] = R,V@np.asarray(rho)
    return np.linalg.inv(delta@np.linalg.inv(c2w)).astype(np.float32)


def projection_matrices(c2w, k, near, far):
    fx,fy = np.tan(get_fov(np.asarray(k)[None])[0]/2)
    p = np.zeros((4,4),np.float32)
    p[0,0],p[1,1],p[3,2] = 1/fx,1/fy,1
    p[2,2],p[2,3] = far/(far-near),-far*near/(far-near)
    view = np.linalg.inv(c2w).T.copy()
    proj = p.T.copy()
    return view, view@proj, proj, float(fx),float(fy)

