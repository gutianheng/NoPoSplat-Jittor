"""Real-source, CPU-only streaming/resume diagnostic, not paper accuracy."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--cache',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--prefetch',type=int,default=8);p.add_argument('--download-workers',type=int,default=12);a=p.parse_args()
    from noposplat_jittor.config import load_config
    from noposplat_jittor.chunk_stream import ChunkSceneStream
    from noposplat_jittor.dataset_conversion import atomic_json
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);cfg=load_config('configs/re10k.yaml')['data'];cfg['source_plans']={'train':str(Path(a.plan).resolve())}
    cfg['chunk_stream']={'cache_root':str(Path(a.cache).resolve()),'cache_gib':40,'reserve_gib':10,'converter_python':str(Path('.refvenv/bin/python').absolute()),
        'network':'hf_mirror','prefetch':a.prefetch,'download_workers':a.download_workers}
    report={'scope':__doc__,'status':'started','config':cfg};started=time.monotonic();stream=None
    try:
        stream=ChunkSceneStream(cfg,'train',111123)
        for i in range(8):
            item=stream.train_example(0);print(json.dumps({'sample':i,'scene':item['scene'],'elapsed_seconds':time.monotonic()-started}),flush=True)
        state=stream.state_dict();atomic_json(out/'data_cursor.json',state)
        expected=[stream.train_example(0) for _ in range(8)];stream.close();stream=None
        resumed=ChunkSceneStream(cfg,'train',999);stream=resumed;resumed.load_state_dict(state)
        for original in expected:
            actual=resumed.train_example(0)
            if actual['scene']!=original['scene']:raise AssertionError('Scene changed after resume')
            for kind in ('context','target'):
                for key in original[kind]:np.testing.assert_array_equal(actual[kind][key],original[kind][key])
        report.update(status='passed',verified_next_examples=8,source_scenes=len(resumed.scene_chunk),torch_imported='torch' in sys.modules,cursor=resumed.state_dict())
        if report['torch_imported']:raise AssertionError('Torch imported into the framework-neutral reader')
    except Exception as e:report.update(status='failed',error=type(e).__name__+': '+str(e));raise
    finally:
        if stream is not None:stream.close()
        report['seconds']=time.monotonic()-started;atomic_json(out/'summary.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('config','cursor')}),flush=True)

if __name__=='__main__':main()
