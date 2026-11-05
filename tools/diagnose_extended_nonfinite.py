"""Reproduce the frozen default prefix, abort before any nonfinite update."""
import sys,json,time,os
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jittor as jt
from noposplat_jittor.optim import AdamW
from noposplat_jittor.config import load_config
from tools.reduced_training import jittor_train
from types import SimpleNamespace

root=Path(__file__).resolve().parents[1]
out=root/('runs/extended_20261103/'+os.environ.get('NPS_DIAGNOSTIC_NAME','nonfinite_diagnostic_001'))
out.mkdir(exist_ok=True)
if os.environ.get('NPS_DIAGNOSTIC_NO_CHECKPOINT')=='1':
    import noposplat_jittor.checkpointing as checkpointing
    def finite_only(state,path,**kwargs):
        count=0
        for key,value in state['model'].items():
            assert np.isfinite(value.numpy()).all(),key
            count+=1
        (out/'finite_model_check.json').write_text(json.dumps({'step':state['step'],'finite_state_arrays':count,
            'checkpoint_writes_suppressed_in_diagnostic_only':True,'recent_opacity':recent},indent=2))
    checkpointing.save_checkpoint=finite_only
    checkpointing.retain_owned_checkpoints=lambda *args:None
original=AdamW.apply
from noposplat_jittor.model import NoPoSplat
if os.environ.get('NPS_DIAGNOSTIC_OLD_OPACITY')=='1':
    import noposplat_jittor.model as model_module
    model_module.map_opacity=lambda p,e:.5*(1-(1-p)**e+p**(1/e))
execute=NoPoSplat.execute
recent=[]
def traced(self,context,global_step=0,trace=None):
    record={} if trace is None else trace
    result=execute(self,context,global_step,record)
    row={'update':global_step+1}
    for key in ('raw_gs1','raw_gs2'):
        raw=record[key][:,0]
        opacity=jt.sigmoid(raw)
        row[key]={'sigmoid_ones':int((opacity==1).sum().item()),'sigmoid_zeros':int((opacity==0).sum().item()),'raw_max':float(raw.max().item()),'raw_min':float(raw.min().item())}
    recent.append(row)
    if len(recent)>8:recent.pop(0)
    return result
NoPoSplat.execute=traced
def checked(self,grads,rates=None,clip=None):
    squared=sum((g*g).sum() for g in grads)
    value=float(squared.item())
    if not np.isfinite(value):
        rows=[]
        for (name,p),g in zip(self.named,grads):
            a=g.numpy();b=p.numpy()
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                row={'name':name,'gradient_nonfinite':int((~np.isfinite(a)).sum()),'parameter_nonfinite':int((~np.isfinite(b)).sum()),'shape':list(a.shape)}
                if name.endswith('dpt.head.4.bias'):row['nonfinite_gradient_indices']=np.flatnonzero(~np.isfinite(a)).tolist()
                rows.append(row)
        (out/'nonfinite.json').write_text(json.dumps({'attempted_update':self.t+1,'gradient_square_sum':str(value),'parameters':rows,'recent_opacity':recent,'unix':time.time()},indent=2))
        raise FloatingPointError('Nonfinite gradient at attempted update '+str(self.t+1))
    if self.t%25==0:print('FINITE_GRADIENT',self.t+1,value,flush=True)
    return original(self,grads,rates,clip)
AdamW.apply=checked
cfg=load_config(root/'configs/extended_20261103/runtime/default.yaml')
cfg['train']['checkpoint_every']=450
resume=os.environ.get('NPS_DIAGNOSTIC_RESUME')
jittor_train(SimpleNamespace(output=str(out),resume=resume,stop_step=int(os.environ.get('NPS_DIAGNOSTIC_STOP','480')),skip_weight_export=True,weights=str(root/'reference/mast3r_initialization_round2d_001/encoder.npz'),lpips_weights=str(root/'weights/lpips_vgg.npz')),cfg)
