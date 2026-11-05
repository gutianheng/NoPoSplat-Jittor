"""Compare actual original preprocessing and LR schedule on CPU (no Jittor).

Only selected trusted pure functions are extracted from the supplied source;
no project installation, model construction, downloads or CUDA are needed.
"""
from pathlib import Path
import argparse
import ast
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--official',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    import torch
    from PIL import Image
    from einops import rearrange
    from noposplat_jittor.data import crop_resize
    from noposplat_jittor.optim import learning_rates
    from noposplat_jittor.config import load_config
    root=Path(args.official)
    tree=ast.parse((root/'src/dataset/shims/crop_shim.py').read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('rescale','center_crop','rescale_and_crop')]
    for node in selected:
        node.returns=None
        for arg in node.args.args:arg.annotation=None
    namespace={'np':np,'torch':torch,'Image':Image,'rearrange':rearrange}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<original-crop-functions>','exec'),namespace)
    rng=np.random.default_rng(123)
    images=rng.integers(0,256,(2,3,360,640),dtype=np.uint8).astype(np.float32)/255
    k=np.array([[[.8,0,.5],[0,.9,.5],[0,0,1]],[[1,0,.4],[0,1.2,.6],[0,0,1]]],np.float32)
    xref,kref=namespace['rescale_and_crop'](torch.tensor(images),torch.tensor(k),(256,256))
    actual,kactual=crop_resize(images,k,(256,256))
    crop={'image_max_abs':float(np.max(np.abs(xref.numpy()-actual))),'k_max_abs':float(np.max(np.abs(kref.numpy()-kactual)))}
    crop['pass']=bool(np.array_equal(xref.numpy(),actual) and np.array_equal(kref.numpy(),kactual))
    a=torch.nn.Parameter(torch.zeros(1));b=torch.nn.Parameter(torch.zeros(1))
    opt=torch.optim.AdamW([{'params':[a],'lr':2e-4},{'params':[b],'lr':2e-5}],lr=2e-4)
    warm=torch.optim.lr_scheduler.LinearLR(opt,1/125,1,total_iters=125)
    cosine=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=18751,eta_min=2e-5)
    scheduler=torch.optim.lr_scheduler.SequentialLR(opt,[warm,cosine],milestones=[125])
    cfg={'lr':2e-4,'backbone_lr_multiplier':.1,'warmup':125,'max_steps':18751}
    errors=[]
    for step in range(18751):
        errors.append(max(abs(x['lr']-y) for x,y in zip(opt.param_groups,learning_rates(step,cfg))))
        opt.step();scheduler.step()
    lr={'max_abs':max(errors),'steps':len(errors),'pass':max(errors)<1e-12}
    data_tree=ast.parse((root/'src/dataset/dataset_re10k.py').read_text(encoding='utf-8'))
    bounds={n.target.id:ast.literal_eval(n.value) for n in ast.walk(data_tree) if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id in ('near','far') and n.value is not None}
    actual_config=load_config(Path(__file__).resolve().parents[1]/'configs/re10k.yaml')
    bounds['pass']=all(actual_config['data'][k]==bounds[k] for k in ('near','far'))
    # Compare against the author's actual scalar tagging code, including float32
    # boundary behavior, on every official evaluation entry and boundary probes.
    from collections import Counter
    from noposplat_jittor.evaluation_summary import overlap_metadata
    tag_tree=ast.parse((root/'src/misc/utils.py').read_text(encoding='utf-8'))
    tag_node=next(n for n in tag_tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_overlap_tag')
    tag_namespace={};exec(compile(ast.Module(body=[tag_node],type_ignores=[]),'<original-overlap>', 'exec'),tag_namespace)
    overlap_report={};overlap_ok=True
    for name in ('re10k','acid'):
        entries=json.loads((root/'assets'/f'evaluation_index_{name}.json').read_text(encoding='utf-8'))
        counts=Counter();mismatches=[]
        for key,entry in entries.items():
            if entry is None:continue
            value=entry.get('overlap');value=value if isinstance(value,float) else .75 if value=='large' else .25
            expected=tag_namespace['get_overlap_tag'](torch.tensor([value],dtype=torch.float32))
            actual=overlap_metadata(entry)['overlap_group'];counts[actual]+=1
            if actual!=expected:mismatches.append(key)
        overlap_report[name]={'groups':dict(counts),'mismatches':mismatches,'pass':not mismatches}
        overlap_ok=overlap_ok and not mismatches
    # Validate the newly supplied single-GPU profile against the real YAML,
    # rather than claiming the old batch=1 smoke profile matches paper training.
    import yaml
    original=yaml.safe_load((root/'config/experiment/re10k_1x8.yaml').read_text(encoding='utf-8'))
    profile=load_config(Path(__file__).resolve().parents[1]/'configs/re10k_official_1x8.yaml')
    pairs={'batch_size':(original['data_loader']['train']['batch_size'],profile['train']['batch_size']),
           'max_steps':(original['trainer']['max_steps'],profile['train']['max_steps']),
           'lr':(float(original['optimizer']['lr']),profile['train']['lr']),
           'warmup':(original['optimizer']['warm_up_steps'],profile['train']['warmup']),
           'backbone_lr_multiplier':(original['optimizer']['backbone_lr_multiplier'],profile['train']['backbone_lr_multiplier']),
           'gap_warmup':(original['dataset']['re10k']['view_sampler']['warm_up_steps'],profile['data']['gap_warmup']),
           'checkpoint_every':(original['checkpointing']['every_n_train_steps'],profile['train']['checkpoint_every'])}
    profile_report={'values_original_and_port':pairs,'pass':all(a==b for a,b in pairs.values()),'execution_status':'NOT_RUN'}
    report={'torch_version':torch.__version__,'crop':crop,'lr_schedule':lr,'original_bounds':bounds,
            'overlap_against_actual_original_function':overlap_report,'single_gpu_config_values':profile_report,
            'scope':'CPU preprocessing/LR/bounds/grouping/config declarations only; NOT a Jittor model or optimizer execution test.'}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
    if not (crop['pass'] and lr['pass'] and bounds['pass'] and overlap_ok and profile_report['pass']):raise SystemExit(1)


if __name__=='__main__':main()
