"""Atomic scene commits for long evaluations; partial scenes never count."""
from pathlib import Path
import json
from .dataset_conversion import atomic_json,plan_digest


class EvaluationJournal:
    def __init__(self,root,binding,keys,*,resume=False):
        self.root=Path(root).resolve();self.keys=list(keys);self.required=set(keys)
        if not self.keys or len(self.required)!=len(self.keys):raise ValueError('Expected unique nonempty scene keys')
        self.root.mkdir(parents=True,exist_ok=True);manifest=self.root/'evaluation_manifest.json'
        expected={'version':1,'binding':binding,'scene_keys':self.keys}
        if manifest.exists():
            if not resume:raise FileExistsError('Use explicit resume or a new evaluation output')
            if json.loads(manifest.read_text())!=expected:raise ValueError('Evaluation model/protocol/source/index changed; refusing mixed results')
        else:
            if resume:raise FileNotFoundError('No evaluation manifest to resume')
            if any(self.root.iterdir()):raise FileExistsError('Refusing an existing unowned evaluation directory')
            atomic_json(manifest,expected)
        self.records=self.root/'completed_scenes';self.records.mkdir(exist_ok=True);self.completed={}
        for file in self.records.glob('*.json'):
            record=json.loads(file.read_text());key=record['scene']
            if key not in self.required or key in self.completed or file.name!=plan_digest({'scene':key})+'.json':raise ValueError('Unexpected/duplicate completed scene record')
            self._validate(record);self.completed[key]=record

    def _validate(self,record):
        import math
        if record['scene'] not in self.required or not record['rows']:raise ValueError('Invalid completed scene')
        rows=record['rows']
        if [int(x['target']) for x in rows]!=record['expected_targets']:raise ValueError('Partial or reordered target views cannot be committed')
        if any(x['scene']!=record['scene'] or any(not math.isfinite(float(x[k])) for k in ('psnr','ssim','lpips')) for x in rows):raise ValueError('Invalid/nonfinite evaluation record')

    def commit(self,key,rows,expected_targets,attempt):
        if key in self.completed:raise FileExistsError('Scene already committed; refusing duplicate metrics')
        record={'scene':key,'rows':rows,'expected_targets':list(map(int,expected_targets)),'attempt':str(attempt)};self._validate(record)
        atomic_json(self.records/(plan_digest({'scene':key})+'.json'),record);self.completed[key]=record

    def rows(self):return [row for key in self.keys if key in self.completed for row in self.completed[key]['rows']]
    def progress(self,status):
        record={'status':status,'completed_scenes':len(self.completed),'required_scenes':len(self.keys),'completed_targets':len(self.rows()),'complete':len(self.completed)==len(self.keys)}
        atomic_json(self.root/'progress.json',record);return record
