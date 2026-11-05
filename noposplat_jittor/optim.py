"""Explicit AdamW to expose update/epsilon/decay conventions for parity tests."""
import math
import numpy as np


def learning_rates(step,cfg):
    base=cfg['lr'];warm=cfg['warmup'];small=base*cfg['backbone_lr_multiplier']
    if step<warm:
        f=1/warm+(1-1/warm)*step/warm
        return base*f,small*f
    t=step-warm;eta=base*.1
    cosine=(1+math.cos(math.pi*t/cfg['max_steps']))/2
    return eta+(base-eta)*cosine,eta+(small-eta)*cosine


class AdamW:
    def __init__(self,named_parameters,lr=2e-4,betas=(.9,.95),eps=1e-8,weight_decay=.05):
        import jittor as jt
        self.jt=jt;self.named=list(named_parameters);self.lr=lr;self.betas=betas;self.eps=eps;self.decay=weight_decay;self.t=0
        self.m=[jt.zeros_like(p).stop_grad() for _,p in self.named]
        self.v=[jt.zeros_like(p).stop_grad() for _,p in self.named]
    @property
    def parameters(self):return [p for _,p in self.named]
    def apply(self,grads,rates=None,clip=None):
        jt=self.jt
        grads=[g.detach() for g in grads]
        if len(grads)!=len(self.named):raise ValueError('Optimizer gradient coverage differs')
        norm=jt.sqrt(sum((g*g).sum() for g in grads))
        if not np.isfinite(float(norm.item())):
            raise FloatingPointError('Nonfinite gradient norm: AdamW update refused before modifying parameters or moments')
        self.t+=1;b1,b2=self.betas
        # Keep update-dependent coefficients as runtime tensor inputs. Python
        # literals otherwise become CUDA source constants and recompile the
        # same update kernel at every warmup/cosine/bias-correction step.
        pair=(self.lr,self.lr) if rates is None else rates
        coefficients=jt.array(np.asarray([1-b1**self.t,1-b2**self.t,
            pair[0],1-pair[0]*self.decay,pair[1],1-pair[1]*self.decay],dtype=np.float32)).stop_grad()
        if clip:
            coef=jt.minimum(float(clip)/(norm+1e-6),1.)
            grads=[g*coef for g in grads]
        for i,((name,p),g) in enumerate(zip(self.named,grads)):
            group=0 if ('gaussian_param_head' in name or 'intrinsic_encoder' in name) else 1
            lr,decay=coefficients[2+2*group],coefficients[3+2*group]
            with jt.no_grad():
                m=b1*self.m[i]+(1-b1)*g;v=b2*self.v[i]+(1-b2)*(g*g)
                self.m[i].update(m);self.v[i].update(v)
                mhat=m/coefficients[0];vhat=v/coefficients[1]
                p.update(p*decay-lr*mhat/(jt.sqrt(vhat)+self.eps))
            # Jittor update() copies the no_grad flag from its new value.
            # Keep active optimizer parameters differentiable on the next step.
            p.start_grad()
    def state_dict(self):
        return {'step':self.t,'names':[n for n,_ in self.named],'m':[v.numpy() for v in self.m],'v':[v.numpy() for v in self.v]}
    def load_state_dict(self,state):
        if state['names']!=[n for n,_ in self.named]:raise ValueError('Optimizer parameter order mismatch')
        self.t=state['step']
        for group in ('m','v'):
            for p,a in zip(getattr(self,group),state[group]):p.assign(self.jt.array(a))


def active_parameters(model):
    # These parameters never participate in the default two-view forward. Torch
    # gives them grad=None, so AdamW must not decay them as if zero gradients.
    result=[];seen=set()
    for name,p in model.named_parameters():
        if p.is_stop_grad() or id(p) in seen:continue
        seen.add(id(p))
        if name=='backbone.mask_token' or '.refinenet4.resConfUnit1.' in name:continue
        if getattr(model,'cfg',{}).get('intrinsics_mode','token')=='none' and name.startswith('backbone.intrinsic_encoder.'):continue
        result.append((name,p))
    return result
