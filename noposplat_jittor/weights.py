from pathlib import Path
import json
import re
import numpy as np


def canonical_key(key):
    key=re.sub(r'^module\.', '', key)
    key=re.sub(r'^encoder\.', '', key)
    key=re.sub(r'\.scratch\.layer_rn\.(\d)\.',lambda m:f'.scratch.layer{int(m[1])+1}_rn.',key)
    return key


def read_arrays(path):
    result={}
    with np.load(path,allow_pickle=False) as f:
        for key in f.files:
            if key.startswith('__'):continue
            name=canonical_key(key)
            if name in result and not np.array_equal(result[name],f[key]):raise ValueError(f'Conflicting tied/alias weights: {key}')
            result[name]=f[key]
    return result


def load_weights(model,path,strict=True,report_path=None):
    import jittor as jt
    weights=read_arrays(path);params=model.state_dict()
    missing=[];shape_errors=[];used=set()
    for name,param in params.items():
        key=canonical_key(name)
        if key not in weights:missing.append(name);continue
        array=weights[key]
        if tuple(array.shape)!=tuple(param.shape):shape_errors.append([name,list(array.shape),list(param.shape)]);continue
        param.assign(jt.array(array));used.add(key)
    unexpected=sorted(set(weights)-used)
    report={'missing':missing,'shape_errors':shape_errors,'unexpected':unexpected,'loaded':len(used),'source':str(path)}
    if report_path:Path(report_path).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if shape_errors or (strict and (missing or unexpected)):raise ValueError(f'Checkpoint is not an exact match: {report}')
    return report


def save_arrays(model,path):
    arrays={k:v.numpy() for k,v in model.state_dict().items()}
    for key,array in arrays.items():
        if not np.isfinite(array).all():raise FloatingPointError('Nonfinite model export refused: '+key)
    np.savez(path,**arrays)

