"""Inspect finite-difference stability without changing the renderer or tolerances."""
from pathlib import Path
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jittor as jt
from noposplat_jittor.model import Gaussians
from noposplat_jittor.renderer import render_one
from noposplat_jittor.rasterizer import release_states
from noposplat_jittor.geometry_numpy import retract_pose

jt.flags.use_cuda = 1
means = np.array([[[.1,.05,2.],[-.2,.1,2.5]]], np.float32)
cov = np.stack([np.eye(3,dtype=np.float32)*.02]*2)[None]
sh = np.array([[[[.8],[.2],[.1]],[[.2],[.7],[.3]]]], np.float32)
opa = np.array([[.5,.6]], np.float32)
params = [jt.array(x).start_grad() for x in (means,cov,sh,opa)]
k = np.array([[.8,0,.5],[0,.8,.5],[0,0,1]],np.float32)
pose = np.eye(4,dtype=np.float32)
theta, rho = jt.zeros(3).start_grad(), jt.zeros(3).start_grad()
cfg = {'scale_invariant':False,'background':[0,0,0]}
mask_np = np.linspace(.1,1,256,dtype=np.float32).reshape(1,16,16)
mask = jt.array(mask_np).stop_grad()
im,_ = render_one(Gaussians(*params),pose,k,.1,100.,(16,16),cfg,theta,rho)
values = [g.numpy() for g in jt.grad((im*mask).sum(), params+[theta,rho])]
release_states()

def measure(m=means, p=pose):
    with jt.no_grad():
        g = Gaussians(*[jt.array(x) for x in (m,cov,sh,opa)])
        image,_ = render_one(g,p,k,.1,100.,(16,16),cfg)
        # Float64 accumulation separates render precision from reduction noise.
        array = image.numpy().copy()
        loss = float((array.astype(np.float64)*mask_np).sum())
    release_states()
    return loss

report = {'analytic_mean_x':float(values[0][0,0,0]),
          'analytic_rho_x':float(values[-1][0]), 'sweep':[]}
for eps in (1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5):
    plus, minus = means.copy(), means.copy()
    plus[0,0,0] += eps; minus[0,0,0] -= eps
    fd_m = (measure(plus)-measure(minus))/(2*eps)
    fd_r = (measure(p=retract_pose(pose,[eps,0,0],[0,0,0]))-
            measure(p=retract_pose(pose,[-eps,0,0],[0,0,0])))/(2*eps)
    report['sweep'].append({'eps':eps,'mean_x_fd':fd_m,'rho_x_fd':fd_r})
path = Path('runs/native_fd_diagnostic_001.json')
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
