from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',default='weights/lpips_vgg.npz');p.add_argument('--reference',default='weights/lpips_reference.npz');p.add_argument('--output',default='runs/lpips_alignment.json');p.add_argument('--cpu',action='store_true');args=p.parse_args()
    import jittor as jt
    from noposplat_jittor.lpips import LPIPSVGG
    jt.flags.use_cuda=0 if args.cpu else 1
    model=LPIPSVGG(args.weights)
    with np.load(args.reference,allow_pickle=False) as a:
        x=jt.array(a['x']).start_grad();y=jt.array(a['y']).stop_grad()
        loss=model(x,y).mean();grad=jt.grad(loss,x)
        report={'loss_abs_error':float(abs(loss.item()-a['loss'].item())),'gradient_max_abs_error':float(np.max(np.abs(grad.numpy()-a['grad'])))}
        report['pass']=bool(np.allclose(loss.item(),a['loss'].item(),atol=1e-5,rtol=1e-4) and np.allclose(grad.numpy(),a['grad'],atol=1e-5,rtol=1e-4))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()

