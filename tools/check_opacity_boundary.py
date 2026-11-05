"""Author opacity identity forward/backward at saturation, plus fail-closed AdamW."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jittor as jt
from noposplat_jittor.model import map_opacity
from noposplat_jittor.optim import AdamW
jt.flags.use_cuda=1
rows=[]
for values,expected in [([0.,1e-8,.1,.5,.9,1.],np.ones(6,np.float32))]:
    x=jt.array(np.asarray(values,np.float32)).start_grad();y=map_opacity(x,1.)
    a=y.numpy();g=jt.grad(y.sum(),x).numpy()
    v=np.asarray(values,np.float32);forward=np.float32(.5)*(np.float32(1)-(np.float32(1)-v)+v)
    np.testing.assert_array_equal(a,forward);np.testing.assert_array_equal(g,expected)
    rows.append({'case':'probability_endpoints_and_interiors','forward':a.tolist(),'gradient':g.tolist(),'passed':True})
x=jt.array(np.asarray([-100.,-20.,0.,20.,100.],np.float32)).start_grad()
p=jt.sigmoid(x);y=map_opacity(p,1.);g=jt.grad(y.sum(),x).numpy()
assert np.isfinite(y.numpy()).all() and np.isfinite(g).all()
np.testing.assert_allclose(g,p.numpy()*(1-p.numpy()),atol=1e-7,rtol=1e-6)
rows.append({'case':'sigmoid_saturation','forward':y.numpy().tolist(),'gradient':g.tolist(),'passed':True})
p=jt.array([1.,2.]).start_grad();opt=AdamW([('test',p)]);before=p.numpy().copy()
try:opt.apply([jt.array([np.nan,0.])])
except FloatingPointError:pass
else:raise AssertionError('Nonfinite gradient accepted')
np.testing.assert_array_equal(p.numpy(),before)
assert opt.t==0
assert all(np.count_nonzero(v.numpy())==0 for v in opt.m+opt.v)
rows.append({'case':'nonfinite_update_refused_without_state_change','passed':True})
record={'status':'passed','checks':rows,'scope':'Boundary correction only; not full model gradient parity or long training acceptance.'}
Path('reports/extended_opacity_boundary_check.json').write_text(json.dumps(record,indent=2,allow_nan=False))
print(json.dumps(record,indent=2,allow_nan=False))
