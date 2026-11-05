import os
"""Honor nvcc12.8 SM120 support instead of Jittor's hardcoded SM90 ceiling."""
from pathlib import Path
import hashlib,json,shutil,subprocess

root=Path(os.environ.get('NPS_BLACKWELL_PREFIX','/root/nps-blackwell'));p=root/'venv/lib/python3.10/site-packages/jittor/compiler.py';s=p.read_text()
codes=subprocess.check_output([str(root/'cuda-12.8/bin/nvcc'),'--list-gpu-code'],text=True).split()
if 'sm_120' not in codes:raise ValueError('Compiler cannot build SM120')
needle='''    max_arch = 90
''';replacement='''    # Private Blackwell compatibility: nvcc12.8 adds native SM120 support.
    max_arch = 120 if nvcc_version >= [12, 8] else 90
'''
if s.count(needle)!=1:raise ValueError('Unexpected Jittor architecture selection source')
(root/'jittor_compiler_before_sm120.py').write_text(s);new=s.replace(needle,replacement);compile(new,str(p),'exec');p.write_text(new)
record={'patch':'Raise native SM ceiling to120 only with nvcc>=12.8; actual nvcc --list-gpu-code contains sm_120',
    'before_sha256':hashlib.sha256(s.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest(),'gpu_validated':False}
(root/'jittor_sm120_patch.json').write_text(json.dumps(record,indent=2))
# Only identical third-party SOURCE/download directories are copied; no old
# architecture-dependent core/ops/shared-library caches are used.
for folder in ('cutt','cutlass','mkl'):
    source=root/'jittor-home/.cache/jittor'/folder;target=root/'jittor-home-sm120/.cache/jittor'/folder
    if source.is_dir() and not target.exists():shutil.copytree(source,target)
print(json.dumps(record),flush=True)
