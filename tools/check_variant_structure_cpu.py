"""Compare actual author and port constructor schemas using Torch meta tensors.

The Jittor module is represented by a constructor-only adapter. No Jittor
operation, numerical forward or backward is executed by this check.
"""
from pathlib import Path
import argparse,copy,json,sys,types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--official',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    import torch
    from hydra import compose,initialize_config_dir
    sys.path.insert(0,str(Path(a.official).resolve()))
    # Encoder constructors do not require dataset loaders/Lightning. Avoid their
    # eager package initializer; child shim/type modules remain the real files.
    dataset_package=types.ModuleType('src.dataset');dataset_package.__path__=[str(Path(a.official).resolve()/'src/dataset')]
    sys.modules['src.dataset']=dataset_package
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder
    from noposplat_jittor.config import load_config
    from noposplat_jittor.weights import canonical_key
    class ConstructorModule(torch.nn.Module):
        def __setattr__(self,name,value):
            if isinstance(value,torch.Tensor) and not isinstance(value,torch.nn.Parameter):value=torch.nn.Parameter(value)
            super().__setattr__(name,value)
    fake=types.ModuleType('jittor');fake.Var=torch.Tensor;fake.ones=torch.ones;fake.zeros=torch.zeros
    fake_nn=types.ModuleType('jittor.nn')
    for name in ('Linear','Conv2d','ConvTranspose2d','Identity','ReLU','Dropout'):setattr(fake_nn,name,getattr(torch.nn,name))
    fake_nn.Module=ConstructorModule;fake.nn=fake_nn;sys.modules['jittor']=fake;sys.modules['jittor.nn']=fake_nn
    from noposplat_jittor.model import NoPoSplat
    cases=[('default',{},[]),('no_intrinsics',{'intrinsics_mode':'none'},['model.encoder.backbone.intrinsics_embed_loc=none']),
        ('global_add',{'intrinsics_mode':'linear'},['model.encoder.backbone.intrinsics_embed_type=linear']),
        ('dense_intrinsics',{'intrinsics_mode':'pixelwise'},['model.encoder.backbone.intrinsics_embed_type=pixelwise']),
        ('no_rgb_shortcut',{'gs_head':'dpt'},['model.encoder.gs_params_head_type=dpt']),
        ('multi_view',{'multi_view':True},['model.encoder.name=noposplat_multi','model.encoder.backbone.name=croco_multi'])]
    results={}
    for name,port_overrides,original_overrides in cases:
        with initialize_config_dir(version_base=None,config_dir=str(Path(a.official).resolve()/'config')):raw=compose(config_name='main',overrides=['+experiment=re10k',*original_overrides])
        set_cfg(raw);cfg=raw
        port=load_config('configs/re10k.yaml')['model'];port.update(port_overrides)
        with torch.device('meta'):original,_=get_encoder(cfg.model.encoder);candidate=NoPoSplat(port)
        def schema(model):return {canonical_key(k):list(v.shape) for k,v in model.state_dict().items()}
        left,right=schema(original),schema(candidate);missing=sorted(set(left)-set(right));extra=sorted(set(right)-set(left))
        mismatched={k:[left[k],right[k]] for k in left.keys()&right.keys() if left[k]!=right[k]}
        results[name]={'author_keys':len(left),'port_keys':len(right),'missing':missing,'unexpected':extra,'shape_mismatches':mismatched,
            'original_overrides':original_overrides,'port_overrides':port_overrides,'pass':not(missing or extra or mismatched)}
        print(json.dumps({'case':name,**results[name]}),flush=True)
        del original,candidate
    report={'scope':__doc__,'jittor_executed':False,'numerical_forward_backward_executed':False,'cases':results,'pass':all(x['pass'] for x in results.values())}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
