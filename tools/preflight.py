"""Read-only inventory before renting compute; no tensor unpickling/downloads.

Chunk existence is NOT scene-payload verification. Missing assets are reported;
this script cannot certify CUDA/Jittor correctness or guarantee cloud spending.
"""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import platform
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def safe_child(root,name):
    root=root.resolve();candidate=(root/name).resolve()
    if candidate!=root and root not in candidate.parents:raise ValueError(f'Index path escapes stage directory: {name}')
    return candidate


def inspect_dataset(root,evaluation_index):
    root=Path(root);fixed=read_json(Path(evaluation_index))
    requested={k:v for k,v in fixed.items() if v is not None}
    result={'root':str(root),'official_index':str(evaluation_index),
            'official_index_sha256':hashlib.sha256(Path(evaluation_index).read_bytes()).hexdigest(),
            'official_index_keys':len(fixed),'requested_non_null_scenes':len(requested),
            'context_view_histogram':dict(Counter(str(len(v['context'])) for v in requested.values())),
            'target_view_histogram':dict(Counter(str(len(v['target'])) for v in requested.values())),
            'stages':{}}
    for stage in ('train','test'):
        folder=root/stage;indexfile=folder/'index.json'
        files=list(folder.glob('*.torch')) if folder.exists() else []
        info={'directory_exists':folder.exists(),'index_exists':indexfile.is_file(),
              'chunk_files':len(files),'chunk_bytes':sum(x.stat().st_size for x in files)}
        if indexfile.is_file():
            index=read_json(indexfile);info['indexed_scenes']=len(index)
            by_chunk={name:safe_child(folder,name).is_file() for name in set(index.values())}
            present={k for k,name in index.items() if by_chunk[name]}
            info.update(indexed_chunks=len(by_chunk),missing_indexed_chunks=sum(not x for x in by_chunk.values()),
                        scenes_with_existing_chunk=len(present))
            if stage=='test':
                covered=set(requested)&present
                info.update(requested_scenes_with_existing_chunk=len(covered),requested_scenes_missing=len(set(requested)-present),
                            full_requested_coverage=len(covered)==len(requested),
                            note='Existence and index membership only. Payload, frame bounds, images and camera filters NOT checked.')
        result['stages'][stage]=info
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--budget-cny',type=float,default=100.)
    args=parser.parse_args()
    if args.budget_cny<=0:parser.error('Budget must be positive')
    official=Path(args.official).resolve()
    disk=shutil.disk_usage(ROOT)
    datasets={name:inspect_dataset(official/'datasets'/name,official/'assets'/f'evaluation_index_{name}.json') for name in ('re10k','acid')}
    weights=[]
    for folder in (official/'pretrained_weights',ROOT/'weights'):
        if folder.is_dir():
            weights.extend({'path':str(p),'bytes':p.stat().st_size} for p in folder.rglob('*') if p.is_file() and p.suffix in ('.pth','.ckpt','.npz','.pkl'))
    syntax=[]
    for path in sorted((ROOT/'noposplat_jittor').glob('*.py')):
        import ast
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path));syntax.append(str(path.relative_to(ROOT)))
    report={'generated_at':datetime.now().astimezone().isoformat(),'read_only_inventory':True,
            'official_source':str(official),'host':{'os':platform.platform(),'python':sys.version,'executable':sys.executable,
            'jittor_find_spec':importlib.util.find_spec('jittor') is not None},
            'disk':{'path':str(ROOT.anchor),'free_bytes':disk.free,'free_gib':disk.free/(1024**3)},
            'budget_cny':args.budget_cny,'rental_started_by_this_audit':False,'spend_by_this_audit_cny':0,
            'datasets':datasets,'weights_in_expected_directories':weights,'runtime_python_files_syntax_checked':syntax,
            'gates':{'cuda_build':'NOT_RUN','jittor_forward_backward':'NOT_RUN','pytorch_jittor_alignment':'NOT_RUN',
                     'full_paper_variants':'INCOMPLETE','paper_training_protocol':'NOT_ALIGNED',
                     'full_datasets':'INCOMPLETE','all_experiment_budget':'UNVERIFIED'},
            'safe_to_start_long_training':False,
            'conclusion':'Local package is an experimental port. Do not start paper training until runtime, scope, data and budget gates pass.'}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
