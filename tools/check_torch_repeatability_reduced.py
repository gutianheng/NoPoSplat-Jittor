"""One-batch author-versus-author control after the fixed GPU evaluation suite.

This measures repeatability, not a substitute for cross-framework acceptance.
It never labels Jittor residuals as inherent nondeterminism without evidence.
"""
from pathlib import Path
import json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from noposplat_jittor.dataset_conversion import atomic_json
from tools.check_alignment import differences


def main():
    root=Path(__file__).resolve().parents[1]
    runs=root/'runs/reduced_20261103';report=runs/'torch_repeatability.json'
    if report.exists():raise FileExistsError('Preserve previous repeatability control')
    state={'status':'waiting_for_fixed_evaluations','started_unix':time.time()};atomic_json(report,state)
    try:
        while True:
            suite=json.loads((runs/'evaluation_suite.json').read_text())
            if suite['status']=='failed':raise RuntimeError('Main evaluation failed; control not run')
            if suite['status']=='all_specified_evaluations_executed':break
            time.sleep(10)
        original=root/'reference/reduced_gpu_photo_20261103'
        repeat=root/'reference/reduced_torch_repeat_20261103'
        if repeat.exists():raise FileExistsError('Use a fresh repeat output')
        env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
        env.update(LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',NVIDIA_TF32_OVERRIDE='0',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        command=['/root/nps-blackwell/refvenv/bin/python','tools/export_reference.py','--official','../NoPoSplat-original',
            '--mast3r','weights/initialization/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
            '--batch',str(original/'batch.npz'),'--output',str(repeat),'--seed','111123','--grad','--photometric','--trusted-input','--math-mode','fp32']
        state['status']='running_author_reference';atomic_json(report,state)
        with (runs/'torch_repeatability.log').open('wb') as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        with np.load(original/'encoder.npz',allow_pickle=False) as a,np.load(repeat/'encoder.npz',allow_pickle=False) as b:
            if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):
                raise ValueError('Initialization differs; this is not a same-state control')
        checks={};energy=0.;error_energy=0.
        for file,is_gradient in [('forward.npz',False),('gradients.npz',True)]:
            with np.load(original/file,allow_pickle=False) as a,np.load(repeat/file,allow_pickle=False) as b:
                if set(a.files)!=set(b.files):raise ValueError('Trace keys differ')
                for key in a.files:
                    left,right=a[key],b[key]
                    row=differences(left,right,1e-6 if is_gradient else 3e-4,.01 if is_gradient else 3e-4)
                    if is_gradient:
                        e=float(np.einsum('i,i->',left.ravel(),left.ravel(),dtype=np.float64))
                        d=right-left;energy+=e;error_energy+=float(np.einsum('i,i->',d.ravel(),d.ravel(),dtype=np.float64))
                        if e>1e-24:row['pass']=bool(row['pass'] and row['relative_l2']<=.02)
                    checks[('grad:' if is_gradient else '')+key]=row
        atomic_json(repeat/'author_repeat_comparison.json',checks)
        state.update(status='completed',identical_initial_state=True,checks=len(checks),
            strict_failed_count=sum(not r['pass'] for r in checks.values()),
            failed_forward=[k for k,r in checks.items() if not k.startswith('grad:') and not r['pass']],
            max_gradient_relative_l2=max(r['relative_l2'] for k,r in checks.items() if k.startswith('grad:')),
            all_parameter_gradient_weighted_relative_l2=float(np.sqrt(error_energy/max(energy,1e-30))),
            note='Single frozen batch, same author framework and initialization. Does not establish a universal nondeterminism bound or waive Jittor failures.')
    except Exception as error:state.update(status='failed',error=type(error).__name__+': '+str(error))
    state['finished_unix']=time.time();atomic_json(report,state);print(json.dumps(state),flush=True)
    if state['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
