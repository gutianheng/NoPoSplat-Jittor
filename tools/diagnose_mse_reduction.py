"""Compare pixelwise square and MSE reduction at identical saved RGB tensors.

Read-only numerical diagnostic; alternative reductions do not modify training.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    import jittor as jt
    jt.flags.use_cuda=1;jt.flags.amp_level=0;jt.flags.use_tensorcore=0
    with np.load(args.reference,allow_pickle=False) as a:x_np=a['color'].copy();y_np=a['target.image'].copy();ref=float(a['mse'])
    x=jt.array(x_np).start_grad();y=jt.array(y_np).stop_grad();squared=(x-y)**2
    squared_np=squared.numpy();expected=(x_np-y_np)**2
    cases={'native_mean':squared.mean(),'native_sum_divide':squared.sum()/x_np.size,'float64_reduction_only':squared.float64().mean().float32()}
    expected_gradient=(np.float32(2)/np.float32(x_np.size))*(x_np-y_np)
    result={'scope':__doc__,'shape':list(x_np.shape),'torch_reference':ref,
        'numpy_float32_mean':float(expected.mean()),'numpy_float64_accumulation':float(expected.mean(dtype=np.float64)),
        'square_element_max_abs':float(np.max(np.abs(squared_np-expected))),'cases':{}}
    for name,loss in cases.items():
        value=float(loss.item());gradient=jt.grad(loss,x).numpy()
        result['cases'][name]={'value':value,'abs_error_to_torch':abs(value-ref),
            'gradient_max_abs':float(np.max(np.abs(gradient-expected_gradient))),
            'gradient_allclose':bool(np.allclose(gradient,expected_gradient,atol=1e-10,rtol=1e-5)),
            'finite':bool(np.isfinite(value) and np.isfinite(gradient).all())}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if not all(x['finite'] for x in result['cases'].values()):raise SystemExit(1)


if __name__=='__main__':main()
