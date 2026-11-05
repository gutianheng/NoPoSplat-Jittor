"""Prepare private AutoDL upload: sources + three diagnostic scenes/dataset.

Run in a PyTorch reference environment. Reads trusted user-supplied .torch
chunks, preserves originals, excludes weights/binaries/caches, rents nothing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from noposplat_jittor.evaluation_summary import overlap_metadata

EXCLUDED={'.git','.venv','.refvenv','.cache','.audit_cache','__pycache__','.test_tmp','data','weights','runs','reference','build','dist'}
SOURCE_SUFFIXES={'.py','.yaml','.yml','.json','.md','.txt','.toml','.cu','.cuh','.h','.hpp','.inl','.cpp','.cc','.c','.cmake','.ipp','.inc','.sh'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def copy_sources(source,destination,allowed_roots=None):
    for path in source.rglob('*'):
        if not path.is_file():continue
        rel=path.relative_to(source)
        if set(rel.parts)&EXCLUDED or any(p.endswith('.egg-info') for p in rel.parts):continue
        if allowed_roots is not None and rel.parts[0] not in allowed_roots:continue
        if path.suffix not in SOURCE_SUFFIXES and path.name not in ('LICENSE','COPYING'):continue
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official',required=True);p.add_argument('--rasterizer',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();official=Path(args.official).resolve();raster=Path(args.rasterizer).resolve();out=Path(args.output).resolve()
    if out.exists() or out.with_suffix('.zip').exists():raise FileExistsError('Choose a new output directory; no originals or old packages are overwritten')
    if ROOT==out or ROOT in out.parents:raise ValueError('Place output outside the project to prevent recursive copying')
    out.mkdir(parents=True);project=out/'NoPoSplat-Jittor'
    copy_sources(ROOT,project)
    copy_sources(official,out/'NoPoSplat-original',{'src','config','README.md','DATASETS.md','LICENSE','pyproject.toml','requirements.txt'})
    copy_sources(raster,out/'rasterizer-original')
    for name in ('re10k','acid','dtu','scannetpp'):
        src=official/'assets'/f'evaluation_index_{name}.json'
        dest=out/'NoPoSplat-original/assets'/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    records={};env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'}
    for name in ('re10k','acid'):
        index=json.loads((official/'datasets'/name/'test/index.json').read_text(encoding='utf-8'))
        fixed=json.loads((official/'assets'/f'evaluation_index_{name}.json').read_text(encoding='utf-8'))
        chosen={};entries={}
        for key in sorted(index):
            if fixed.get(key) is None:continue
            tag=overlap_metadata(fixed[key])['overlap_group']
            if tag in ('small','medium','large') and tag not in chosen:
                chosen[tag]=key;entries[key]=fixed[key]
        if len(chosen)!=3:raise ValueError(f'{name}: need one available scene from each overlap group')
        subset=project/'configs'/f'evaluation_index_{name}_diagnostic3.json'
        subset.write_text(json.dumps(entries,indent=2),encoding='utf-8')
        command=[sys.executable,'-B',str(ROOT/'tools/convert_dataset.py'),'--source',str(official/'datasets'/name),
                 '--output',str(project/'data'/name),'--stage','test','--evaluation-index',str(subset),'--trusted-input']
        run=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        logs=out/'local_preparation_logs';logs.mkdir(exist_ok=True)
        (logs/f'{name}_conversion.log').write_text(run.stdout+run.stderr,encoding='utf-8')
        if run.returncode:raise RuntimeError(f'{name} conversion failed: {run.stderr}')
        # Freeze a real preprocessed batch; this decodes images and checks shape,
        # camera convention and baseline filters before any paid cloud session.
        run=subprocess.run([sys.executable,'-B',str(project/'tools/prepare_batch.py'),'--config',f'configs/{name}.yaml',
                            '--index',f'configs/evaluation_index_{name}_diagnostic3.json','--output',f'fixtures/batch_{name}.npz'],
                           cwd=project,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (logs/f'{name}_prepare_batch.log').write_text(run.stdout+run.stderr,encoding='utf-8')
        if run.returncode:raise RuntimeError(f'{name} prepare_batch failed: {run.stderr}')
        from noposplat_jittor.config import load_config
        from noposplat_jittor.data import SceneStream
        cfg=load_config(project/f'configs/{name}.yaml')['data'];cfg['root']=str(project/'data'/name)
        stream=SceneStream(cfg['root'],'test',cfg,index=subset)
        verified=[example['scene'] for example in stream.test_examples()]
        records[name]={'purpose':'DIAGNOSTIC ONLY, not full-test reproduction','groups_to_scene':chosen,
                       'decoded_and_camera_checked':verified,'target_count':sum(len(x['target']) for x in entries.values())}
    (out/'diagnostic_data_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    manifest=[{'path':str(p.relative_to(out)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]
    manifest_path=out/'UPLOAD_MANIFEST.json';manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8');files.append(manifest_path)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:
            name=str(path.relative_to(out)).replace('\\','/')
            info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o755 if path.suffix=='.sh' else 0o644)<<16
            z.writestr(info,path.read_bytes())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP CRC check failed')
        for item in manifest:
            if hashlib.sha256(z.read(item['path'])).hexdigest()!=item['sha256']:raise RuntimeError('ZIP manifest check failed')
    sha=digest(archive);archive.with_suffix('.zip.sha256').write_text(sha+'  '+archive.name+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha,'files':len(files),'diagnostic_data':records,
                      'gpu_executed':False,'rental_started':False},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
