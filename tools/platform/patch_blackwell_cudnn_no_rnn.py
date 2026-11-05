import os
"""Do not build cuDNN APIs removed in v9 and unused by NoPoSplat.

This explicitly disables legacy cuDNN RNN operators in this private environment.
It does not implement substitute RNNs or alter convolution/NoPoSplat math.
"""
from pathlib import Path
import hashlib,json

root=Path(os.environ.get('NPS_BLACKWELL_PREFIX','/root/nps-blackwell'));p=root/'venv/lib/python3.10/site-packages/jittor/compile_extern.py';s=p.read_text()
needle='''        for fname in f:
            culib_src_files.append(os.path.join(r, fname))'''
replacement='''        for fname in f:
            # Private NoPoSplat environment: cuDNN9 removed legacy RNN APIs.
            # NoPoSplat has no RNN/GRU/LSTM; keep convolution operators intact.
            if lib_name == "cudnn" and locals().get("major", 8) >= 9 and fname.startswith("cudnn_rnn"):
                continue
            culib_src_files.append(os.path.join(r, fname))'''
if s.count(needle)!=1:raise ValueError('Unexpected compile_extern source')
backup=root/'jittor_compile_extern_before_rnn_filter.py';backup.write_text(s)
new=s.replace(needle,replacement);compile(new,str(p),'exec');p.write_text(new)
record={'patch':'Exclude cuDNN v6-era RNN operators only when actual cuDNN major >=9','unsupported_in_this_environment':'Legacy cuDNN RNN ops; not used by NoPoSplat',
        'before_sha256':hashlib.sha256(s.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest(),'model_math_modified':False,'gpu_validated':False}
(root/'jittor_cudnn9_rnn_filter.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
