"""Two independent evaluator processes; unchanged fixed-index mathematics.

Parallel execution is an operational choice, never a change to test coverage.
Only allocation failures may retry serially from the evaluator's commit journal.
"""
from pathlib import Path
import subprocess,os,signal,time,shlex,shutil
from tools.run_extended_suite import ROOT,OUT,PREFIX,load
from tools.compare_reduced_runs import compare_evaluation
from noposplat_jittor.dataset_conversion import atomic_json

def run_pair(suite,name,case,step=None,dataset='re10k'):
    suite.assert_core();suite.budget(2100,reserve=1.)
    base=OUT/'evaluation';base.mkdir(exist_ok=True)
    role='official_pretrained' if step is None else 'own_reduced_training'
    configuration='official_'+dataset if step is None else case
    jobs={};failures=[]
    def completed(framework):
        folder=base/(name+'_'+framework);file=folder/'results.json'
        if not file.exists():return False
        result=load(file)
        if result.get('scene_count')!=500 or result.get('target_count')!=1500:raise ValueError('Existing result has incomplete coverage')
        manifest=load(folder/'evaluation_manifest.json')
        if len(manifest['scene_keys'])!=500 or len(list((folder/'completed_scenes').glob('*.json')))!=500:raise ValueError('Existing result journal differs')
        return True
    existing=[fw for fw in ('torch','jittor') if completed(fw)] if suite.state.get('allow_budget_evaluation_resume') else []
    pending=[fw for fw in ('torch','jittor') if fw not in existing]
    parallel=len(pending)>1 and not suite.state.get('evaluation_resource_serial_fallback',False)
    def launch(framework,retry=False):
        folder=base/(name+'_'+framework)
        weight=f'reference/{dataset}_forward_001/encoder.npz' if step is None else str(OUT/(case+'_'+framework)/f'encoder_{step:06d}.npz')
        tool='tools/evaluate_torch_resumable.py' if framework=='torch' else 'tools/evaluate_resumable.py'
        args=[tool,'--config',f'configs/extended_20261103/runtime/{configuration}.yaml','--weights',weight,
            '--weights-role',role,'--output',str(folder),'--save-scene-limit','12',
            '--save-extra-scenes',f'configs/extended_20261103/qualitative_extra_{dataset}.json']
        if (folder/'binding.json').exists() or (folder/'journal.json').exists() or (folder/'manifest.json').exists():
            args.append('--resume-evaluation')
        # EvaluationJournal's actual immutable binding filename.
        if folder.exists() and any(folder.iterdir()) and '--resume-evaluation' not in args:args.append('--resume-evaluation')
        log=OUT/(name+'_'+framework+('_serial_retry' if retry else '')+'.log')
        if log.exists():
            if not suite.state.get('allow_budget_evaluation_resume') or not (folder/'evaluation_manifest.json').exists():
                raise FileExistsError('Preserve an existing evaluator log: '+str(log))
            for number in range(1,1000):
                candidate=log.with_name(log.stem+f'_budget_resume_{number:03d}.log')
                if not candidate.exists():log=candidate;break
            else:raise RuntimeError('Too many owned evaluator resumes')
        stream=log.open('wb')
        command=[str(PREFIX/'refvenv/bin/python'),*args] if framework=='torch' else ['bash','-c','source scripts/blackwell_env.sh && exec python '+shlex.join(args)]
        child=subprocess.Popen(command,cwd=ROOT,env=suite.env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        return {'process':child,'stream':stream,'start':time.monotonic(),'log':log,'retry':retry}
    def await_jobs(active):
        completed={}
        try:
            while len(completed)<len(active):
                suite.state.update(active='evaluation_pair_'+name,active_pid=None,
                    active_evaluation_pids={fw:j['process'].pid for fw,j in active.items() if fw not in completed},
                    active_evaluation_names=[name+'_'+fw for fw in active],parallel_evaluator_processes=len(active))
                suite.save();meter=suite.guard_meter()
                if meter['estimated_spend_cny']>meter['soft_cap_cny']:raise TimeoutError('Evaluation budget guard reached')
                for fw,j in active.items():
                    if fw in completed:continue
                    code=j['process'].poll()
                    if code is None:
                        if time.monotonic()-j['start']>5400:raise TimeoutError('Evaluator timeout '+fw)
                        continue
                    seconds=time.monotonic()-j['start'];j['stream'].close();completed[fw]=code
                    suite.state['stages'].append({'name':name+'_'+fw,'framework':fw,'seconds':seconds,'returncode':code,
                        'independent_parallel_process':len(active)>1,'serial_resource_retry':j['retry']});suite.save()
                time.sleep(2)
        finally:
            for job in active.values():
                if job['process'].poll() is None:
                    os.killpg(job['process'].pid,signal.SIGTERM);job['process'].wait(timeout=30)
                if not job['stream'].closed:job['stream'].close()
        return completed
    codes={fw:0 for fw in existing}
    if parallel:
        try:
            for fw in pending:jobs[fw]=launch(fw)
            codes.update(await_jobs(jobs))
        except Exception:
            for job in jobs.values():
                if job['process'].poll() is None:os.killpg(job['process'].pid,signal.SIGTERM)
                if not job['stream'].closed:job['stream'].close()
            raise
    else:
        for fw in pending:
            suite.budget(2100,reserve=1.);jobs[fw]=launch(fw);codes.update(await_jobs({fw:jobs[fw]}))
    for fw,code in codes.items():
        if code==0:continue
        log=jobs[fw]['log'].read_text(errors='replace')
        if not parallel or not any(x in log for x in ('Unable to alloc cuda device memory','CUDA out of memory')):
            raise RuntimeError('Independent evaluator failed: '+name+'_'+fw+'; see '+str(jobs[fw]['log']))
        failures.append({'framework':fw,'log':str(jobs[fw]['log']),'reason':'Allocation failure only; retry same immutable journal serially'})
        suite.state['evaluation_resource_serial_fallback']=True;suite.save()
        retry=launch(fw,True);result=await_jobs({fw:retry})
        if result[fw]:raise RuntimeError('Serial allocation retry failed: '+fw)
    for fw in ('torch','jittor'):
        result=load(base/(name+'_'+fw)/'results.json')
        if result['scene_count']!=500 or result['target_count']!=1500:raise ValueError('Incomplete fixed-index coverage')
    tolerance={'psnr':.1,'ssim':.002,'lpips':.005} if step is None else {'psnr':.3,'ssim':.01,'lpips':.01}
    comparison=compare_evaluation(base/(name+'_torch'),base/(name+'_jittor'),tolerance)
    comparison['execution_resource_schedule']={'parallel_independent_evaluators':parallel,'resource_failures':failures,'reused_complete_frameworks':existing,
        'mathematical_settings_and_test_coverage_unchanged':True}
    atomic_json(base/(name+'_comparison.json'),comparison)
    suite.state.pop('active_evaluation_pids',None);suite.state.pop('active_evaluation_names',None);suite.save()
