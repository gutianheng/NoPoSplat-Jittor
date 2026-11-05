"""Compare actual saved full-model states; does not replace test-set metrics."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from noposplat_jittor.dataset_conversion import atomic_json


def energy(x):return float(np.einsum('i,i->',x.ravel(),x.ravel(),dtype=np.float64))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--torch',required=True);p.add_argument('--jittor',required=True);p.add_argument('--initial',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    records=[];totals=dict(reference=0.,difference=0.,torch_update=0.,jittor_update=0.);elements=0
    with np.load(a.torch,allow_pickle=False) as left,np.load(a.jittor,allow_pickle=False) as right,np.load(a.initial,allow_pickle=False) as initial:
        if set(left.files)!=set(right.files) or set(left.files)!=set(initial.files):raise ValueError('Model state key sets differ')
        for key in left.files:
            t,j,b=left[key],right[key],initial[key]
            if t.shape!=j.shape or t.shape!=b.shape:raise ValueError('Shape mismatch '+key)
            if t.dtype!=np.float32 or j.dtype!=np.float32 or not np.isfinite(t).all() or not np.isfinite(j).all():raise ValueError('Invalid saved parameter '+key)
            d=j-t;et=energy(t);ed=energy(d);elements+=t.size
            totals['reference']+=et;totals['difference']+=ed;totals['torch_update']+=energy(t-b);totals['jittor_update']+=energy(j-b)
            records.append({'key':key,'shape':list(t.shape),'max_abs_difference':float(np.max(abs(d),initial=0)),'relative_l2':float(np.sqrt(ed/(et+1e-30)))})
    report={'scope':__doc__,'state_arrays':len(records),'float32_state_elements':elements,'all_keys_shapes_dtypes_finite_pass':True,
        'whole_state_relative_l2':float(np.sqrt(totals['difference']/totals['reference'])),'difference_norm_relative_to_torch_training_update':float(np.sqrt(totals['difference']/totals['torch_update'])),
        'torch_training_update_norm':float(np.sqrt(totals['torch_update'])),'jittor_training_update_norm':float(np.sqrt(totals['jittor_update'])),
        'largest_parameter_relative_differences':sorted(records,key=lambda x:x['relative_l2'],reverse=True)[:20],
        'note':'Learned weights can diverge under nonlinear training. These values are descriptive; no post-hoc pass threshold is applied.'}
    atomic_json(a.output,report);print(json.dumps({k:v for k,v in report.items() if k!='largest_parameter_relative_differences'}))


if __name__=='__main__':main()
