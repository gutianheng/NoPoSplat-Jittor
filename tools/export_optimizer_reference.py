from pathlib import Path
import argparse
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='reference/optimizer.npz');args=p.parse_args()
    import torch
    from noposplat_jittor.optim import learning_rates
    rng=np.random.default_rng(29);initial=[rng.normal(size=(3,4)).astype(np.float32),rng.normal(size=(5,)).astype(np.float32)]
    a,b=[torch.nn.Parameter(torch.tensor(x)) for x in initial]
    optimizer=torch.optim.AdamW([{'params':[a],'lr':2e-4},{'params':[b],'lr':2e-5}],betas=(.9,.95),weight_decay=.05,eps=1e-8)
    cfg={'lr':2e-4,'backbone_lr_multiplier':.1,'warmup':125,'max_steps':18751}
    arrays={'initial_a':initial[0],'initial_b':initial[1]};pa=[];pb=[];ga=[];gb=[]
    for i in range(130):
        x=rng.normal(size=(3,4)).astype(np.float32);y=rng.normal(size=5).astype(np.float32)
        ga.append(x);gb.append(y);a.grad=torch.tensor(x);b.grad=torch.tensor(y)
        for group,lr in zip(optimizer.param_groups,learning_rates(i,cfg)):group['lr']=lr
        torch.nn.utils.clip_grad_norm_([a,b],.5)
        optimizer.step();pa.append(a.detach().numpy().copy());pb.append(b.detach().numpy().copy())
    arrays.update(grads_a=np.stack(ga),grads_b=np.stack(gb),parameters_a=np.stack(pa),parameters_b=np.stack(pb))
    # Consecutive real backward passes catch parameter stop-gradient bugs that
    # a fixture of externally supplied gradients cannot detect.
    for case,decay,betas,reset in [('train',.05,(.9,.95),False),('pose',0.,(.9,.999),True)]:
        initial=rng.normal(size=(3,3)).astype(np.float32) if not reset else np.zeros((3,3),np.float32)
        param=torch.nn.Parameter(torch.tensor(initial));targets=[];states=[];gradients=[]
        opt=(torch.optim.Adam if reset else torch.optim.AdamW)([param],lr=.005,betas=betas,weight_decay=decay,eps=1e-8)
        for _ in range(20):
            target=rng.normal(size=(3,3)).astype(np.float32);targets.append(target)
            opt.zero_grad();((param-torch.tensor(target))**2).mean().backward()
            gradients.append(param.grad.detach().numpy().copy());opt.step();states.append(param.detach().numpy().copy())
            if reset:
                with torch.no_grad():param.zero_()
        arrays.update({f'autograd_{case}_initial':initial,f'autograd_{case}_targets':np.stack(targets),
                       f'autograd_{case}_parameters':np.stack(states),f'autograd_{case}_gradients':np.stack(gradients)})
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);np.savez(out,**arrays);print(out)


if __name__=='__main__':main()

