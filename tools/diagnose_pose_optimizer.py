"""Check repeated-gradient and reset semantics of the actual Jittor optimizer."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jittor as jt
from noposplat_jittor.optim import AdamW
jt.flags.use_cuda=1
rows=[]
param=jt.zeros((3,3)).start_grad()
optimizer=AdamW([('theta',param)],lr=.005,betas=(.9,.999),weight_decay=0.)
for i in range(3):
    row={'step':i,'stopped_before':bool(param.is_stop_grad()),'slice_stopped':bool(param[0].is_stop_grad())}
    gradient=jt.grad(((param-1)**2).sum(),param)
    row['gradient']=gradient.numpy().tolist()
    optimizer.apply([gradient]);jt.sync_all()
    row.update(stopped_after_update=bool(param.is_stop_grad()),value=param.numpy().tolist())
    param.assign(jt.zeros_like(param))
    row['stopped_after_reset']=bool(param.is_stop_grad())
    rows.append(row)
Path('runs/pose_optimizer_gradient_diagnostic.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
