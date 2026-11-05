"""Full fixed-index NVS with per-scene atomic recovery; no training claims.

Each scene uses the existing evaluator unchanged (including target-pose Adam
alignment). A completed scene is journaled only after every target is written.
Create OUTPUT/PAUSE_REQUESTED to stop at a scene boundary before a restart.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,csv,json,os,sys,time,uuid
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    p.add_argument('--lpips-weights',default='weights/lpips_vgg.npz');p.add_argument('--resume-evaluation',action='store_true');p.add_argument('--limit',type=int)
    p.add_argument('--save-scene-limit',type=int,default=24)
    p.add_argument('--save-extra-scenes',help='JSON list of predeclared fixed scene IDs for matched qualitative panels')
    p.add_argument('--weights-role',choices=['specified','official_pretrained','own_reduced_training'],default='specified')
    p.add_argument('--pause-after-scenes',type=int,help='Controlled pause diagnostic; never changes the required index');a=p.parse_args()
    if a.pause_after_scenes is not None and a.pause_after_scenes<1:p.error('Pause count must be positive')
    from noposplat_jittor.config import load_config
    from noposplat_jittor.data import make_scene_stream
    from noposplat_jittor.dataset_conversion import atomic_json,plan_digest,sha256_file
    from noposplat_jittor.evaluation_journal import EvaluationJournal
    cfg=load_config(a.config);mode=cfg.get('runtime',{}).get('math_mode','fp32')
    if mode=='fp32':os.environ['NVIDIA_TF32_OVERRIDE']='0'
    stream=make_scene_stream(cfg['data'],'test',cfg['seed'],cfg['data']['eval_index'],a.limit);out=Path(a.output).resolve();started=time.monotonic()
    try:
        binding={'model_weights_sha256':sha256_file(a.weights),'weights_role':a.weights_role,'lpips_weights_sha256':sha256_file(a.lpips_weights),'model':cfg['model'],'renderer':cfg['renderer'],
            'math_mode':mode,'test':{k:v for k,v in cfg['test'].items() if k!='save_images'},'lpips_weight':cfg['train']['lpips_weight'],
            'data':{k:v for k,v in cfg['data'].items() if k not in ('root','eval_index','chunk_stream','source_plans')},
            'fixed_index_sha256':plan_digest(stream.fixed),'source_plan_sha256':getattr(stream,'binding',None),'limit':a.limit}
        journal=EvaluationJournal(out,binding,stream.keys,resume=a.resume_evaluation);journal.progress('initializing')
        all_keys=list(stream.keys);stream.keys=[k for k in all_keys if k not in journal.completed]
        from noposplat_jittor.evaluation_summary import overlap_metadata
        save_keys=set(all_keys[:a.save_scene_limit])
        if a.save_scene_limit:
            for group in ('small','medium','large'):
                candidates=[k for k in all_keys if overlap_metadata(stream.fixed[k])['overlap_group']==group]
                if candidates:save_keys.add(min(candidates))
        if a.save_extra_scenes:
            extra=json.loads(Path(a.save_extra_scenes).read_text())
            if not isinstance(extra,list) or any(not isinstance(k,str) for k in extra) or not set(extra)<=set(all_keys):raise ValueError('Extra qualitative scene IDs are not in the fixed evaluation index')
            save_keys.update(extra)
        import jittor as jt
        from noposplat_jittor.precision import configure_math
        from noposplat_jittor.model import NoPoSplat
        from noposplat_jittor.weights import load_weights
        from noposplat_jittor.lpips import LPIPSVGG
        from noposplat_jittor.cli import _evaluate_stream,write_image
        jt.flags.use_cuda=1;configure_math(jt,mode);jt.set_global_seed(cfg['seed'])
        model=NoPoSplat(cfg['model']);load_weights(model,a.weights,strict=True);perceptual=LPIPSVGG(a.lpips_weights)
        atomic_json(out/'runtime.json',{'jittor':jt.__version__,'use_cuda':jt.flags.use_cuda,'use_tensorcore':jt.flags.use_tensorcore,'amp_level':jt.flags.amp_level,'math_mode':mode,'torch_imported':'torch' in sys.modules,'saved_scene_keys':sorted(save_keys),'image_selection':'first fixed-index scene IDs plus smallest ID in each overlap group; never selected by metrics'})
        if 'torch' in sys.modules:raise RuntimeError('Torch unexpectedly imported into Jittor evaluation process')
        iterator=iter(stream.test_examples())
        while not (out/'PAUSE_REQUESTED').exists():
            try:example=next(iterator)
            except StopIteration:break
            key=example['scene'];attempt=out/'attempts'/(plan_digest({'scene':key})[:16]+'-'+uuid.uuid4().hex)
            if key in save_keys:
                attempt.mkdir(parents=True)
                for i,image in enumerate(example['context']['image']):write_image(attempt/f'context_{i}.png',image*.5+.5)
                atomic_json(attempt/'context_frames.json',{'context_indices':example['context']['index'].tolist(),'scene':key})
            local_cfg=copy.deepcopy(cfg);local_cfg['test']['save_images']=key in save_keys
            one=SimpleNamespace(test_examples=lambda:iter([example]));options=SimpleNamespace(output=str(attempt),no_align=False,limit=1)
            _evaluate_stream(options,local_cfg,model,perceptual,jt,one)
            rows=[]
            with (attempt/'per_target.csv').open(newline='',encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    for k in ('psnr','ssim','lpips','overlap','encode_seconds','render_align_metric_seconds'):row[k]=float(row[k])
                    row['target']=int(row['target']);row['align_pose']=row['align_pose']=='True'
                    if 'target_slot' in row:row['target_slot']=int(row['target_slot'])
                    rows.append(row)
            journal.commit(key,rows,example['target']['index'],attempt.relative_to(out));print(json.dumps(journal.progress('running')),flush=True)
            if a.pause_after_scenes is not None and len(journal.completed)>=a.pause_after_scenes:atomic_json(out/'PAUSE_REQUESTED',{'reason':'explicit controlled-pause diagnostic'})
        progress=journal.progress('completed' if len(journal.completed)==len(all_keys) else 'paused')
        if progress['complete']:
            from noposplat_jittor.evaluation_summary import summarize_nvs
            rows=journal.rows();atomic_json(out/'results.json',{'status':'evaluation_completed','scene_count':len(all_keys),'target_count':len(rows),'binding':binding,
                **summarize_nvs(rows),'scope':'explicit_subset' if a.limit else 'entire_specified_fixed_index','elapsed_this_invocation_seconds':time.monotonic()-started,
                'weights_role':a.weights_role,'claim_note':'Official pretrained evaluation and own reduced-data training are distinct experiments; neither is a full paper-training claim.'})
            temporary=out/'.per_target.csv.tmp'
            with temporary.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);f.flush();os.fsync(f.fileno())
            os.replace(temporary,out/'per_target.csv')
        print(json.dumps(progress),flush=True)
    except Exception as e:
        if out.is_dir():atomic_json(out/'failure.json',{'status':'failed','error':type(e).__name__+': '+str(e),'note':'Previously committed scenes are retained; a failure is not completion.'})
        raise
    finally:stream.close()

if __name__=='__main__':main()
