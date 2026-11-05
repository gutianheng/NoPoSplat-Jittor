"""Isolate diagnostic Gaussian squared-mean adjoints at identical stored outputs."""
from pathlib import Path
import argparse,json
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.reference);out=Path(a.output)
if out.exists():raise FileExistsError(out)
import jittor as jt
jt.flags.use_cuda=1;jt.flags.use_tensorcore=0;jt.flags.amp_level=0
with np.load(root/'forward.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in ('means','covariances','harmonics','opacities')}
checks={}
with np.load(root/'gaussian_gradients.npz',allow_pickle=False) as expected:
    for mode in ('pow_mean','multiply_mean','multiply_sum_divide'):
        variables={k:jt.array(v).start_grad() for k,v in arrays.items()}
        loss=sum((v**2).mean() if mode=='pow_mean' else (v*v).mean() if mode=='multiply_mean' else (v*v).sum()/v.numel() for v in variables.values())
        gradients=jt.grad(loss,list(variables.values()))
        for (key,_),gradient in zip(variables.items(),gradients):
            value=gradient.numpy();target=expected[key];diff=np.abs(value-target)
            checks[mode+':'+key]={'max_abs':float(diff.max()),'relative_l2':float(np.linalg.norm(diff.ravel())/(np.linalg.norm(target.ravel())+1e-30)),
                                   'exact':bool(np.array_equal(value,target)),'allclose_1e-10_1e-5':bool(np.allclose(value,target,atol=1e-10,rtol=1e-5))}
report={'scope':__doc__,'checks':checks};out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
