"""Publish a completed checkpoint; interrupted writes keep prior files intact."""
from pathlib import Path
import os
import uuid


def save_checkpoint(state,path,*,writer=None):
    """Single-writer save to an adjacent temporary file, then atomic rename.

    The installed Jittor safepickle writes directly to its destination. Keep a
    partial multi-GB dump out of the normal step_*.pkl checkpoint namespace.
    """
    path=Path(path)
    if path.exists():raise FileExistsError(f'Refusing to replace checkpoint: {path}')
    if writer is None:
        import jittor as jt
        writer=jt.save
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp.pkl')
    try:
        writer(state,str(temporary))
        with temporary.open('rb+') as file:os.fsync(file.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def retain_owned_checkpoints(new_path,keep):
    """Prune only exact files registered after successful publication here.

    Existing unrelated checkpoints are never discovered/deleted by globbing.
    A rolling two-checkpoint policy still retains the final training checkpoint.
    """
    from .dataset_conversion import atomic_json,sha256_file
    import json
    path=Path(new_path).absolute();directory=path.parent.resolve();keep=int(keep)
    if keep<2:raise ValueError('Keep at least two complete checkpoints for recovery')
    if path.is_symlink() or path.resolve().parent!=directory or not path.is_file():raise ValueError('Checkpoint is not a published regular file')
    ledger=directory/'checkpoint_retention.json'
    state=json.loads(ledger.read_text()) if ledger.exists() else {'version':1,'registered':[],'removed':[]}
    if state['version']!=1:raise ValueError('Unexpected checkpoint ownership ledger')
    row={'file':path.name,'bytes':path.stat().st_size,'sha256':sha256_file(path)}
    previous=next((x for x in state['registered'] if x['file']==path.name),None)
    if previous and previous!=row:raise ValueError('Registered checkpoint changed')
    if previous is None:state['registered'].append(row)
    atomic_json(ledger,state)
    while len(state['registered'])>keep:
        old=state['registered'][0];candidate=directory/old['file']
        if candidate.is_symlink() or candidate.resolve().parent!=directory:raise ValueError('Owned checkpoint path escaped run directory')
        if candidate.exists():
            if candidate.stat().st_size!=old['bytes'] or sha256_file(candidate)!=old['sha256']:raise ValueError('Refusing to delete a modified registered checkpoint')
            candidate.unlink()
        state['removed'].append(state['registered'].pop(0));atomic_json(ledger,state)
