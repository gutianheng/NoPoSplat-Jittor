"""Diagnostic same-short-trained-state tests; primary results stay unchanged.

Two new fixed-index tests distinguish learned-state differences from evaluation
runtime differences. They are not selected reruns of the primary experiment.
"""
from pathlib import Path
import json,os,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from noposplat_jittor.dataset_conversion import atomic_json
from tools.compare_reduced_runs import compare_evaluation


def main():
    root=Path(__file__).resolve().parents[1];runs=root/'runs/reduced_20261103';out=runs/'evaluation_controls';out.mkdir(exist_ok=True)
    status=out/'cross_runtime_status.json'
    if status.exists():raise FileExistsError('Preserve completed/failed control; explicit recovery required')
    state={'status':'running','started_unix':time.time(),'primary_results_unchanged':True,
           'question':'Same learned weights across evaluation runtimes, not independent training parity',
           'predeclared_same_state_mean_gap_tolerances':{'psnr':.1,'ssim':.002,'lpips':.005},'stages':[]}
    atomic_json(status,state)
    env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
    env.update(LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',NVIDIA_TF32_OVERRIDE='0',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',use_parallel_op_compiler='0')
    try:
        for source,runtime in [('torch','jittor'),('jittor','torch')]:
            name='default_'+source+'_weights_'+runtime;folder=out/name;state['active']=name;atomic_json(status,state)
            tool='tools/evaluate_resumable.py' if runtime=='jittor' else 'tools/evaluate_torch_resumable.py'
            args=[tool,'--config','configs/reduced_20261103/default.yaml','--weights',str(runs/('default_'+source)/'encoder_000500.npz'),
                  '--weights-role','own_reduced_training','--output',str(folder),'--save-scene-limit','12']
            command=['bash','-c','source scripts/blackwell_env.sh && python '+' '.join(args)] if runtime=='jittor' else ['/root/nps-blackwell/refvenv/bin/python',*args]
            tick=time.monotonic()
            with (out/(name+'.log')).open('wb') as log:
                subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
            torch_dir=folder if runtime=='torch' else runs/'evaluation/default_torch'
            jittor_dir=folder if runtime=='jittor' else runs/'evaluation/default_jittor'
            value=compare_evaluation(torch_dir,jittor_dir,state['predeclared_same_state_mean_gap_tolerances'])
            value.update(question=state['question'],same_weight_source=source,
                note='Additional diagnostic after a primary short-training LPIPS failure; does not replace that failed primary comparison.')
            atomic_json(out/('same_'+source+'_weights_comparison.json'),value)
            state['stages'].append({'name':name,'seconds':time.monotonic()-tick,'aggregate_acceptance_pass':value['aggregate_acceptance_pass']});atomic_json(status,state)
        state.update(status='completed',active=None)
    except Exception as error:state.update(status='failed',error=type(error).__name__+': '+str(error))
    state['finished_unix']=time.time();atomic_json(status,state);print(json.dumps(state),flush=True)
    if state['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
