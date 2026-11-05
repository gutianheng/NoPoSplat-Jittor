"""Check the actual Jittor model/AdamW save-load-resume path against a fixture."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',default='reference/optimizer_002.npz');p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    import jittor as jt
    from noposplat_jittor.optim import AdamW,learning_rates
    jt.flags.use_cuda=1;jt.flags.use_tensorcore=0;jt.flags.amp_level=0
    cfg={'lr':2e-4,'backbone_lr_multiplier':.1,'warmup':125,'max_steps':18751}
    with np.load(args.reference,allow_pickle=False) as data:ref={k:data[k].copy() for k in data.files}
    class Toy(jt.nn.Module):
        def __init__(self):super().__init__();self.a=jt.array(ref['initial_a']).start_grad();self.b=jt.array(ref['initial_b']).start_grad()
    def optimizer(model):return AdamW([('gaussian_param_head.weight',model.a),('backbone.weight',model.b)])
    def run(model,opt,start,stop):
        for i in range(start,stop):opt.apply([jt.array(ref['grads_a'][i]),jt.array(ref['grads_b'][i])],learning_rates(i,cfg),.5);jt.sync_all()
    original=Toy();opt=optimizer(original);run(original,opt,0,65)
    rng=np.random.default_rng(322);rng.random(19)
    state={'model':original.state_dict(),'optimizer':opt.state_dict(),'step':65,'config':{'train':cfg},'numpy_rng':rng.bit_generator.state}
    jt.save(state,str(out/'step_000065.pkl'));run(original,opt,65,130);expected=original.state_dict()
    loaded=jt.load(str(out/'step_000065.pkl'));resumed=Toy();resumed_opt=optimizer(resumed)
    resumed.load_state_dict(loaded['model']);resumed_opt.load_state_dict(loaded['optimizer']);resumed.train()
    identity=all(a is b for a,b in zip((resumed.a,resumed.b),resumed_opt.parameters))
    if not identity:raise AssertionError('Model loading replaced parameters referenced by optimizer')
    restored=np.random.default_rng();restored.bit_generator.state=loaded['numpy_rng'];rng_equal=np.array_equal(rng.random(20),restored.random(20))
    run(resumed,resumed_opt,loaded['step'],130)
    checks={}
    for key,value in resumed.state_dict().items():
        aa=value.numpy();bb=expected[key].numpy();checks[key]={'max_abs':float(np.max(np.abs(aa-bb))),'pass':bool(np.allclose(aa,bb,atol=1e-7,rtol=1e-6))}
    torch_agreement=bool(np.allclose(resumed.a.numpy(),ref['parameters_a'][-1],atol=1e-6,rtol=1e-5) and np.allclose(resumed.b.numpy(),ref['parameters_b'][-1],atol=1e-6,rtol=1e-5))
    moment_equal=all(np.allclose(a.numpy(),b.numpy(),atol=1e-7,rtol=1e-6) for kind in ('m','v') for a,b in zip(getattr(opt,kind),getattr(resumed_opt,kind)))
    report={'scope':'Small actual model/optimizer checkpoint and resume; no dropout RNG or full-model resource claim',
        'resumed_step':loaded['step'],'final_optimizer_step':resumed_opt.t,'optimizer_parameter_identity':identity,
        'numpy_rng_restored':rng_equal,'moments_match':moment_equal,'torch_fixture_agreement':torch_agreement,'parameters':checks,
        'pass':identity and rng_equal and moment_equal and torch_agreement and all(x['pass'] for x in checks.values())}
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
