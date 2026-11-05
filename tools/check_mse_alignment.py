"""Real PyTorch vs Jittor scalar-loss and gradient comparison across RGB cases."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('export','check'));p.add_argument('--reference',required=True);p.add_argument('--output');a=p.parse_args()
    path=Path(a.reference)
    if a.mode=='export':
        if path.exists():raise FileExistsError(path)
        import torch
        rng=np.random.default_rng(704);arrays={};i=0
        for shape in ((1,3,32,32),(1,3,256,256),(3,3,256,256)):
            for pattern in ('random','near_equal','mixed_residual'):
                y=rng.uniform(size=shape).astype(np.float32)
                if pattern=='random':x=rng.uniform(size=shape).astype(np.float32)
                else:
                    residual=rng.normal(0,1e-3 if pattern=='near_equal' else 1e-5,size=shape).astype(np.float32)
                    if pattern=='mixed_residual':residual.reshape(-1)[::37]+=rng.normal(size=residual.reshape(-1)[::37].size).astype(np.float32)
                    x=y+residual
                xx=torch.tensor(x,device='cuda',requires_grad=True);yy=torch.tensor(y,device='cuda');loss=torch.nn.functional.mse_loss(xx,yy);loss.backward()
                arrays.update({f'{i}.x':x,f'{i}.y':y,f'{i}.gradient':xx.grad.cpu().numpy(),f'{i}.loss':np.asarray(loss.detach().cpu()),f'{i}.pattern':np.asarray(pattern)});i+=1
        arrays['count']=np.asarray(i);path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**arrays);print('Exported '+str(i)+' real PyTorch MSE cases',flush=True);return
    if not a.output:p.error('check requires --output')
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    import jittor as jt
    from noposplat_jittor.losses import mse_loss
    from tools.check_alignment import differences
    jt.flags.use_cuda=1;jt.flags.use_tensorcore=0;jt.flags.amp_level=0;cases=[]
    with np.load(path,allow_pickle=False) as ref:
        for i in range(int(ref['count'])):
            x=jt.array(ref[f'{i}.x']).start_grad();y=jt.array(ref[f'{i}.y']).stop_grad();loss=mse_loss(x,y);gradient=jt.grad(loss,x)
            scalar=differences(ref[f'{i}.loss'],loss.numpy(),1e-8,1e-6);derivative=differences(ref[f'{i}.gradient'],gradient.numpy(),1e-10,1e-5)
            cases.append({'shape':list(x.shape),'pattern':ref[f'{i}.pattern'].item(),'loss':scalar,'gradient':derivative,'pass':scalar['pass'] and derivative['pass']})
    report={'scope':__doc__,'cases':cases,'pass':all(x['pass'] for x in cases),'pixel_dtype':'float32','scalar_accumulation':'float64','returned_loss_dtype':'float32'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
