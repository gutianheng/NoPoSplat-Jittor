"""Trusted full NoPoSplat encoder checkpoint -> framework-neutral named NPZ.

MASt3R initialization is handled by export_reference.py, which calls the original
checkpoint_filter_fn. This converter never invents mappings for incompatible heads.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from noposplat_jittor.weights import canonical_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--trusted-input',action='store_true')
    args=p.parse_args()
    if not args.trusted_input:p.error('Checkpoint deserialization is only safe for trusted files')
    import torch
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    state=torch.load(args.input,map_location='cpu',weights_only=False)
    state=state.get('state_dict',state)
    has_encoder=any(k.startswith(('encoder.','module.encoder.')) for k in state)
    arrays={}
    for key,value in state.items():
        if not isinstance(value,torch.Tensor):continue
        if has_encoder and not key.startswith(('encoder.','module.encoder.')):continue
        name=canonical_key(key)
        a=value.detach().float().cpu().numpy()
        if name in arrays and not np.array_equal(arrays[name],a):raise ValueError(f'Alias mismatch: {key}')
        arrays[name]=a
    if not any(k.startswith('backbone.') for k in arrays):raise ValueError('Not a full NoPoSplat encoder state_dict; use the MASt3R initialization exporter instead')
    output.parent.mkdir(parents=True,exist_ok=True);np.savez(output,**arrays)
    print(f'Converted {len(arrays)} named arrays; validate with strict Jittor loading before use.')


if __name__=='__main__':main()

