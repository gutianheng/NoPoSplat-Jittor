from pathlib import Path
import argparse
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from noposplat_jittor.config import load_config
from noposplat_jittor.data import SceneStream,collate_numpy


def main():
    p=argparse.ArgumentParser(description='Freeze one preprocessed batch for both frameworks')
    p.add_argument('--config',default='configs/re10k.yaml');p.add_argument('--index',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();cfg=load_config(args.config);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    stream=SceneStream(cfg['data']['root'],'test',cfg['data'],index=args.index)
    example=next(stream.test_examples());batch=collate_numpy([example])
    arrays={f'{kind}.{k}':v for kind,d in batch.items() for k,v in d.items()}
    arrays['scene']=np.asarray(example['scene']);out.parent.mkdir(parents=True,exist_ok=True);np.savez(out,**arrays)
    print(out,example['scene'])


if __name__=='__main__':main()

