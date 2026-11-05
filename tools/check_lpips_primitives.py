"""Compare actual CUDA ReLU-at-zero and max-pool tie subgradients."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('export','check'));p.add_argument('--reference',required=True);p.add_argument('--output');a=p.parse_args();ref=Path(a.reference)
    if a.mode=='export':
        if ref.exists():raise FileExistsError(ref)
        import torch
        rng=np.random.default_rng(18);cases=[('relu',np.array([-1.,-0.,0.,1.,1e-7],dtype='float32')),
            ('pool',np.ones((1,2,8,8),dtype='float32')),('pool',np.zeros((1,2,8,8),dtype='float32')),
            ('pool',np.tile(np.array([[1,1,0,0],[1,1,0,0],[3,3,2,2],[3,3,2,2]],dtype='float32'),(1,2,2,2))),
            ('pool',rng.normal(size=(2,3,7,9)).astype('float32'))];data={'kinds':np.asarray([c[0] for c in cases])}
        for i,(kind,x) in enumerate(cases):
            tx=torch.tensor(x,device='cuda',requires_grad=True);y=torch.nn.functional.relu(tx) if kind=='relu' else torch.nn.functional.max_pool2d(tx,2,2)
            u=rng.normal(size=tuple(y.shape)).astype('float32');(y*torch.tensor(u,device='cuda')).sum().backward()
            data.update({f'input{i}':x,f'output{i}':y.detach().cpu().numpy(),f'upstream{i}':u,f'gradient{i}':tx.grad.cpu().numpy()})
        ref.parent.mkdir(parents=True,exist_ok=True);np.savez(ref,**data);print('Exported CUDA primitive reference',flush=True);return
    if not a.output:p.error('check requires --output')
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    import jittor as jt
    jt.flags.use_cuda=1;jt.flags.use_tensorcore=0;jt.flags.amp_level=0
    checks=[]
    with np.load(ref,allow_pickle=False) as data:
        for i,kind in enumerate(data['kinds']):
            x=jt.array(data[f'input{i}']).start_grad();y=jt.nn.ReLU()(x) if kind=='relu' else jt.nn.MaxPool2d(2,stride=2)(x)
            gradient=jt.grad((y*jt.array(data[f'upstream{i}']).stop_grad()).sum(),x)
            forward=y.numpy();backward=gradient.numpy();expected=data[f'gradient{i}']
            checks.append({'case':i,'kind':str(kind),'shape':list(x.shape),'forward_exact':bool(np.array_equal(forward,data[f'output{i}'])),
                'gradient_exact':bool(np.array_equal(backward,expected)),'gradient_max_abs':float(np.max(np.abs(backward-expected)))})
    report={'scope':'Actual CUDA ReLU/pool primitives on five synthetic fixtures, including positive ties. Not a complete VGG alignment test.',
        'checks':checks,'pass':all(c['forward_exact'] and c['gradient_exact'] for c in checks)}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
