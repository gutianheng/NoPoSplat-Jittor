"""Compare checkpoint names/shapes against the real original constructor on meta.

This is a STATIC PARAMETER-SCHEMA test. A tiny constructor-only Jittor facade
uses PyTorch meta tensors to collect the port's declarations. It does NOT run
Jittor, any forward/backward, a renderer, or numerical equivalence tests.
"""
import argparse
from pathlib import Path
import importlib.util
import json
import sys
import types

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(args.official).resolve()));sys.path.append(str(ROOT))
    import torch
    from hydra import initialize_config_dir,compose
    # Bypass dataset/encoder package __init__ registries, which import unrelated
    # video and CUDA evaluation dependencies. Leaf modules remain ORIGINAL files.
    official=Path(args.official).resolve()
    for name in ('src.dataset','src.model.encoder'):
        package=types.ModuleType(name);package.__path__=[str(official/Path(*name.split('.')))]
        sys.modules[name]=package
    from dacite import from_dict,Config
    from omegaconf import OmegaConf
    from src.global_cfg import set_cfg
    from src.model.encoder.encoder_noposplat import EncoderNoPoSplat,EncoderNoPoSplatCfg
    from noposplat_jittor.config import load_config
    from noposplat_jittor.weights import canonical_key
    with initialize_config_dir(version_base=None,config_dir=str(Path(args.official).resolve()/'config')):
        raw=compose(config_name='main',overrides=['+experiment=re10k'])
    set_cfg(raw);cfg=from_dict(EncoderNoPoSplatCfg,OmegaConf.to_container(raw.model.encoder,resolve=True),Config(cast=[tuple]))
    with torch.device('meta'):
        reference=EncoderNoPoSplat(cfg)
    expected={canonical_key(k):tuple(v.shape) for k,v in reference.state_dict().items()}
    facade=types.ModuleType('jittor')
    facade.nn=torch.nn;facade.Var=torch.Tensor
    facade.zeros=lambda shape,**kw:torch.nn.Parameter(torch.zeros(shape,device='meta'))
    facade.ones=lambda shape,**kw:torch.nn.Parameter(torch.ones(shape,device='meta'))
    previous=sys.modules.get('jittor');sys.modules['jittor']=facade
    try:
        spec=importlib.util.spec_from_file_location('_nps_schema_only',ROOT/'noposplat_jittor/model.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        with torch.device('meta'):ported=module.NoPoSplat(load_config(ROOT/'configs/re10k.yaml')['model'])
        actual={canonical_key(k):tuple(v.shape) for k,v in ported.state_dict().items()}
    finally:
        if previous is None:sys.modules.pop('jittor',None)
        else:sys.modules['jittor']=previous
    missing=sorted(set(expected)-set(actual));unexpected=sorted(set(actual)-set(expected))
    mismatched={k:{'original':expected[k],'port':actual[k]} for k in set(expected)&set(actual) if expected[k]!=actual[k]}
    report={'scope':'constructor/meta parameter names and shapes ONLY','jittor_executed':False,
            'forward_backward_executed':False,'torch_version':torch.__version__,
            'original_canonical_arrays':len(expected),'port_canonical_arrays':len(actual),
            'original_declared_elements':sum(__import__('math').prod(x) for x in expected.values()),
            'missing':missing,'unexpected':unexpected,'mismatched':mismatched,'pass':not(missing or unexpected or mismatched)}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
