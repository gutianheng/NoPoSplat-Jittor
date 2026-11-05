from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',default='runs/renderer_alignment.json');args=p.parse_args()
    import jittor as jt
    from noposplat_jittor.model import Gaussians
    from noposplat_jittor.renderer import render_one
    from noposplat_jittor.rasterizer import release_states
    from tools.check_alignment import differences
    jt.flags.use_cuda=1
    try:
        with np.load(args.reference,allow_pickle=False) as a:
            keys=('means','covariances','harmonics','opacities')
            params=[jt.array(a[k]).start_grad() for k in keys]
            theta=jt.zeros(3).start_grad();rho=jt.zeros(3).start_grad()
            color,depth=render_one(Gaussians(*params),a['pose'][0],a['intrinsics'][0],.1,100.,(16,16),{'scale_invariant':False,'background':[0,0,0]},theta,rho)
            mask=jt.array(np.linspace(.1,1,256,dtype=np.float32).reshape(1,16,16)).stop_grad()
            loss=(color*mask).sum()+.01*(depth*mask).sum()
            grads=jt.grad(loss,params+[theta,rho])
            report={'color':differences(a['color'][0],color.numpy(),1e-5,1e-4),'depth':differences(a['depth'][0],depth.numpy()[0],1e-5,1e-4)}
            for key,g in zip(keys+('theta','rho'),grads):
                ref=a['grad_'+key]
                actual=g.numpy().reshape(ref.shape)
                item=differences(ref,actual,1e-5,1e-3)
                if np.linalg.norm(ref.reshape(-1))>1e-12:item['pass']=bool(item['pass'] and item['relative_l2']<.01)
                report['grad_'+key]=item
    finally:release_states()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)
    if not all(x['pass'] for x in report.values()):raise SystemExit(1)


if __name__=='__main__':main()

