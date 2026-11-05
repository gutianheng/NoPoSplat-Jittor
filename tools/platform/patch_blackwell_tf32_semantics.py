"""Match PyTorch convolution TF32 permission without FP16 down-conversion.

Jittor's global flag==1 selects CUBLAS_COMPUTE_32F_FAST_TF32 for matmuls, but
its cuDNN code additionally allowed FP32->FP16 conversion. Keep flag==0 and
other modes unchanged; mode 1 instead uses CUDNN_DEFAULT_MATH, as PyTorch.
"""
from pathlib import Path
import hashlib,json

root=Path('/root/nps-blackwell');base=root/'venv/lib/python3.10/site-packages/jittor/extern/cuda/cudnn/ops'
report={'scope':__doc__,'files':[],'qualification':'Patched source only; numerical and throughput checks still required.'}
for path in sorted(base.glob('*.cc')):
    text=path.read_text()
    old='CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION'
    replacement='(use_tensorcore == 1 ? CUDNN_DEFAULT_MATH : CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)'
    if old not in text:continue
    if replacement in text:raise RuntimeError('Already patched: '+str(path))
    saved=root/'cudnn_tf32_original'/path.name;saved.parent.mkdir(exist_ok=True)
    if saved.exists():raise FileExistsError(saved)
    saved.write_bytes(path.read_bytes());updated=text.replace(old,replacement);path.write_text(updated)
    report['files'].append({'path':str(path),'before_sha256':hashlib.sha256(saved.read_bytes()).hexdigest(),'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
if not report['files']:raise RuntimeError('Expected cuDNN math-type call sites not found')
(root/'jittor_tf32_semantics_patch.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
