"""Isolate encoder Jacobian using the EXACT same upstream photometric gradient.

This diagnostic replaces the loss gradient at Gaussian outputs with an exported
constant. It is NOT an end-to-end renderer/loss parity check or a training path.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default='configs/re10k.yaml');args=p.parse_args()
    root=Path(args.reference);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    import jittor as jt
    from noposplat_jittor.config import load_config
    from noposplat_jittor.model import NoPoSplat
    from noposplat_jittor.weights import load_weights
    from noposplat_jittor.optim import active_parameters
    from tools.check_alignment import differences
    jt.flags.use_cuda=1;jt.flags.use_tensorcore=0;jt.flags.amp_level=0
    model=NoPoSplat(load_config(args.config)['model']);load_weights(model,root/'encoder.npz',strict=True);model.eval()
    # Jittor eval() stops parameter gradients; restore them explicitly, just as
    # check_alignment --grad does. This differs from PyTorch eval() semantics.
    for parameter in model.parameters():parameter.start_grad()
    with np.load(root/'batch.npz',allow_pickle=False) as a:
        context={k.split('.',1)[1]:jt.array(a[k]).stop_grad() for k in a.files if k.startswith('context.')}
    g=model(context)
    with np.load(root/'gaussian_gradients.npz',allow_pickle=False) as a:
        loss=sum((getattr(g,key)*jt.array(a[key]).stop_grad()).sum() for key in a.files)
    params=active_parameters(model)
    with np.load(root/'gradients.npz',allow_pickle=False) as expected:
        if {key for key,_ in params}!=set(expected.files):raise AssertionError('Active parameter keys differ before gradient execution')
    grads=jt.grad(loss,[p for _,p in params]);jt.sync_all();checks={}
    with np.load(root/'gradients.npz',allow_pickle=False) as a:
        for (key,_),gradient in zip(params,grads):
            item=differences(a[key],gradient.numpy(),1e-6,.01)
            item['pass']=bool(item['pass'] and item['relative_l2']<=.02);checks[key]=item
        if set(checks)!=set(a.files):raise AssertionError('Parameter key coverage mismatch')
    report={'scope':__doc__,'thresholds':{'atol':1e-6,'rtol':.01,'relative_l2':.02},'checks':checks,'pass':all(x['pass'] for x in checks.values())}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps({'checks':len(checks),'failed':[key for key,x in checks.items() if not x['pass']],
        'max_relative_l2':max(x['relative_l2'] for x in checks.values()),'report':str(out)}))
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
