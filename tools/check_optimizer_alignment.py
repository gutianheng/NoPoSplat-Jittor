from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',default='reference/optimizer.npz');p.add_argument('--output',default='runs/optimizer_alignment.json');p.add_argument('--cpu',action='store_true');args=p.parse_args()
    import jittor as jt
    from noposplat_jittor.optim import AdamW,learning_rates
    jt.flags.use_cuda=0 if args.cpu else 1
    cfg={'lr':2e-4,'backbone_lr_multiplier':.1,'warmup':125,'max_steps':18751}
    errors=[];passes=[]
    with np.load(args.reference,allow_pickle=False) as ref:
        a=jt.array(ref['initial_a']).start_grad();b=jt.array(ref['initial_b']).start_grad()
        opt=AdamW([('gaussian_param_head.weight',a),('backbone.weight',b)])
        for i,(x,y) in enumerate(zip(ref['grads_a'],ref['grads_b'])):
            opt.apply([jt.array(x),jt.array(y)],learning_rates(i,cfg),.5);jt.sync_all()
            aa,bb=a.numpy(),b.numpy()
            errors.append(max(float(np.max(np.abs(aa-ref['parameters_a'][i]))),float(np.max(np.abs(bb-ref['parameters_b'][i])))))
            passes.append(np.allclose(aa,ref['parameters_a'][i],atol=1e-6,rtol=1e-5) and np.allclose(bb,ref['parameters_b'][i],atol=1e-6,rtol=1e-5))
        autograd_checks=[]
        for case,decay,betas,reset in [('train',.05,(.9,.95),False),('pose',0.,(.9,.999),True)]:
            key=f'autograd_{case}_initial'
            if key not in ref:raise ValueError('Reference lacks consecutive-autograd fixtures; re-export it')
            param=jt.array(ref[key]).start_grad();optimizer=AdamW([('parameter',param)],lr=.005,betas=betas,weight_decay=decay)
            case_errors=[];case_passes=[]
            for index,target in enumerate(ref[f'autograd_{case}_targets']):
                gradient=jt.grad(((param-jt.array(target).stop_grad())**2).mean(),param)
                actual_gradient=gradient.numpy()
                optimizer.apply([gradient]);jt.sync_all();actual=param.numpy()
                if param.is_stop_grad():raise AssertionError('Optimizer stopped an active parameter')
                case_errors.append(float(np.max(np.abs(actual-ref[f'autograd_{case}_parameters'][index]))))
                case_passes.append(np.allclose(actual,ref[f'autograd_{case}_parameters'][index],atol=1e-6,rtol=1e-5) and
                                   np.allclose(actual_gradient,ref[f'autograd_{case}_gradients'][index],atol=1e-6,rtol=1e-5))
                if reset:param.assign(jt.zeros_like(param));param.start_grad()
            autograd_checks.append({'case':case,'steps':len(case_errors),'max_abs':max(case_errors),'pass':bool(all(case_passes))})
    report={'steps':len(errors),'max_abs':max(errors),'autograd_checks':autograd_checks,
            'pass':bool(all(passes) and all(x['pass'] for x in autograd_checks)),
            'scope':'AdamW two-group 130-step fixture plus consecutive backward/update and Adam camera-reset fixtures'}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()

