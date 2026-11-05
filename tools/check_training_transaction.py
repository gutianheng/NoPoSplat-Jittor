"""Exercise the actual CLI transaction with a toy renderer and a Torch oracle.

Checks accumulation, clipping, parameter groups, CSV, checkpoint, RNG and resume.
The small neural model, synthetic stream and renderer substitute are explicitly
test doubles; this is NOT a Gaussian renderer, dataset or full-model test.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,csv,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('export','check'));parser.add_argument('--reference',required=True);parser.add_argument('--output');args=parser.parse_args()
    from noposplat_jittor.config import load_config
    from noposplat_jittor.optim import learning_rates
    cfg=load_config('configs/re10k.yaml');cfg['train'].update(batch_size=1,accumulate=4,max_steps=4,warmup=2,checkpoint_every=2,log_every=1)
    c=cfg['train'];p=Path(args.reference)
    if args.mode=='export':
        if p.exists():raise FileExistsError(p)
        import torch
        rng=np.random.default_rng(71);data={'x':rng.normal(size=(8,12)).astype('float32'),'y':rng.normal(size=(8,12)).astype('float32')}
        model=torch.nn.Module();model.backbone=torch.nn.Linear(12,12);model.gaussian_param_head=torch.nn.Linear(12,12,bias=False)
        with torch.no_grad():
            for name,param in model.named_parameters():data['initial:'+name]=rng.normal(scale=.1,size=tuple(param.shape)).astype('float32');param.copy_(torch.tensor(data['initial:'+name]))
        groups=[{'params':list(model.gaussian_param_head.parameters())},{'params':list(model.backbone.parameters())}]
        opt=torch.optim.AdamW(groups,lr=c['lr'],betas=c['betas'],eps=1e-8,weight_decay=c['weight_decay']);sampler=np.random.default_rng(cfg['seed']);losses=[]
        for step in range(c['max_steps']):
            indices=sampler.integers(0,8,size=4);x=torch.tensor(data['x'][indices]);y=torch.tensor(data['y'][indices]);opt.zero_grad()
            prediction=model.gaussian_param_head(torch.tanh(model.backbone(x)));loss=torch.nn.functional.mse_loss(prediction,y);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
            for group,rate in zip(opt.param_groups,learning_rates(step,c)):group['lr']=rate
            opt.step();losses.append(loss.item())
        for name,param in model.named_parameters():data['final:'+name]=param.detach().numpy()
        data['losses']=np.asarray(losses);p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,**data);print('Exported Torch true batch-4 updates for the CLI accumulation test',flush=True);return
    if not args.output:parser.error('check requires --output')
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    import jittor as jt
    import noposplat_jittor.data as data_module
    import noposplat_jittor.renderer as renderer_module
    import noposplat_jittor.rasterizer as rasterizer_module
    from noposplat_jittor.cli import train
    jt.flags.use_cuda=1;jt.flags.use_tensorcore=0;jt.flags.amp_level=0
    with np.load(p,allow_pickle=False) as a:data={k:a[k].copy() for k in a.files}
    class Toy(jt.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=jt.nn.Linear(12,12);self.gaussian_param_head=jt.nn.Linear(12,12,bias=False)
            self.load_state_dict({k.removeprefix('initial:'):jt.array(v) for k,v in data.items() if k.startswith('initial:')})
            for v in self.parameters():v.start_grad()
        def execute(self,context,step=0):return self.gaussian_param_head(jt.tanh(self.backbone(context['image'][:,0].reshape((-1,12))))).reshape((-1,3,2,2))
    class Stream:
        def __init__(self,root,split,config,seed,index=None,limit=None):assert split=='train';self.rng=np.random.default_rng(seed)
        def state_dict(self):return {'kind':'transaction_toy','rng':self.rng.bit_generator.state}
        def load_state_dict(self,state):
            if state['kind']!='transaction_toy':raise ValueError('Wrong toy stream state')
            self.rng.bit_generator.state=state['rng']
        def close(self):pass
        def train_example(self,step):
            i=int(self.rng.integers(0,8));return {'context':{'image':data['x'][i].reshape((1,3,2,2))},'target':{
                'image':data['y'][i].reshape((1,3,2,2)),'extrinsics':np.eye(4,dtype='float32')[None],
                'intrinsics':np.eye(3,dtype='float32')[None],'near':np.array([.1],dtype='float32'),'far':np.array([100],dtype='float32')}}
    class ZeroPerceptual:
        def __call__(self,prediction,truth):return (prediction*0).sum((1,2,3),keepdims=True)
    def renderer(g,*unused,batch=0,**kwargs):return g[batch],None
    original=(data_module.SceneStream,renderer_module.render_one,rasterizer_module.release_states)
    data_module.SceneStream=Stream;renderer_module.render_one=renderer;rasterizer_module.release_states=lambda:jt.sync_all()
    try:
        continuous=Toy();train(SimpleNamespace(output=str(out/'continuous'),resume=None),cfg,continuous,ZeroPerceptual(),jt)
        resumed=Toy();train(SimpleNamespace(output=str(out/'resumed'),resume=str(out/'continuous/step_000002.pkl')),cfg,resumed,ZeroPerceptual(),jt)
        checks={}
        for name,param in continuous.state_dict().items():
            a=param.numpy();b=resumed.state_dict()[name].numpy();ref=data['final:'+name]
            checks[name]={'torch_max_abs':float(np.max(np.abs(a-ref))),'resume_max_abs':float(np.max(np.abs(a-b))),
                'torch_pass':bool(np.allclose(a,ref,atol=1e-6,rtol=1e-5)),'resume_pass':bool(np.array_equal(a,b))}
        def rows(path):
            with path.open() as f:return list(csv.DictReader(f))
        full=rows(out/'continuous/train.csv');partial=rows(out/'resumed/train.csv')
        loss_pass=bool(np.allclose([float(r['loss']) for r in full],data['losses'],atol=1e-6,rtol=1e-5))
        resume_loss=all(float(a['loss'])==float(b['loss']) for a,b in zip(full[2:],partial)) and len(partial)==2
        batch_pass=all(int(x['effective_batch'])==4 for x in full+partial)
        report={'scope':__doc__,'parameters':checks,'torch_batch4_loss_agreement':loss_pass,'resume_losses_exact':resume_loss,'effective_batch_recorded':batch_pass,
            'pass':loss_pass and resume_loss and batch_pass and all(x['torch_pass'] and x['resume_pass'] for x in checks.values())}
        (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:data_module.SceneStream,renderer_module.render_one,rasterizer_module.release_states=original
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
