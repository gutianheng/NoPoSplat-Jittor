"""Record genuinely unreadable test candidates, preserving frozen reserve order."""
import json,pickle,zipfile,time,hashlib
from pathlib import Path
from noposplat_jittor.dataset_conversion import ChunkConversion,atomic_json,sha256_file,scene_arrays,write_scene
from tools.read_selected_torch_archive import selected_scenes

class CheckedCandidateConversion(ChunkConversion):
    def publish(self):
        index={key:row['file'] for chunk in self.state['chunks'].values() for key,row in chunk['scenes'].items()}
        unavailable={key:reason for chunk in self.state['chunks'].values() for key,reason in chunk.get('unavailable_scenes',{}).items()}
        if set(index)&set(unavailable) or not (set(index)|set(unavailable))<=self.required:raise ValueError('Invalid candidate coverage')
        if unavailable and self.plan['stage']!='test':raise ValueError('Training scenes may not be silently excluded')
        complete=(set(index)|set(unavailable))==self.required and len(self.state['chunks'])==len(self.plan['chunks'])
        atomic_json(self.root/'index.json',index)
        atomic_json(self.root/'unavailable_source_candidates.json',unavailable)
        atomic_json(self.root/'conversion.json',{'status':'complete' if complete else 'partial','count':len(index),
            'checked_scene_count':len(index)+len(unavailable),'unavailable_source_count':len(unavailable),
            'required_scene_count':len(self.required),'completed_chunks':len(self.state['chunks']),'required_chunks':len(self.plan['chunks']),
            'plan_sha256':self.state['plan_sha256'],'source_repository':self.plan.get('repository'),'source_commit':self.plan.get('commit'),
            'note':'Count is genuine readable payloads; checked coverage includes explicitly recorded corrupt-source candidates, without fake archives.',
            'source_sha256':{k:v['source_sha256'] for k,v in self.state['chunks'].items()}})
    def convert(self,row,source,loader):
        try:return super().convert(row,source,loader)
        except (pickle.UnpicklingError,RuntimeError,zipfile.BadZipFile) as error:
            if self.plan['stage']!='test':raise
            source=Path(source)
            if source.stat().st_size!=row['bytes'] or sha256_file(source)!=row['sha256']:raise ValueError('Source integrity mismatch; not an input-validity exclusion')
            # A format/CRC failure must be demonstrated independently of Torch.
            try:
                with zipfile.ZipFile(source) as z:bad=z.testzip()
            except zipfile.BadZipFile as crc_error:bad=str(crc_error)
            if bad is None:raise
            converted={};unavailable={}
            for key in sorted(row['scenes']):
                try:
                    loaded=list(selected_scenes(source,{key}))
                    if len(loaded)!=1:raise ValueError('Selected metadata coverage differs')
                    _,cameras,images=loaded[0]
                    name=hashlib.sha256(key.encode()).hexdigest()+'.npz'
                    converted[key]=write_scene(self.root/name,scene_arrays(key,cameras,images))
                except zipfile.BadZipFile as selected_error:
                    unavailable[key]={'reason':'Unreadable pinned source ZIP member: '+str(selected_error),
                        'source_chunk':row['path'],'source_sha256':row['sha256'],'independently_observed_bad_crc_member':bad,
                        'does_not_use_model_scores':True,'unix':time.time()}
            if set(converted)|set(unavailable)!=set(row['scenes']):raise ValueError('Incomplete source inspection')
            self.state['chunks'][row['path']]={'source_sha256':row['sha256'],'scenes':converted,'unavailable_scenes':unavailable,
                'selected_storage_crc_checked':True,'preserved_corrupt_source':str(source),
                'initial_loader_error':type(error).__name__+': '+str(error)}
            atomic_json(self.ledger_path,self.state);self.publish()
            print(json.dumps({'source_integrity_inspection':row['path'],'valid_selected_scenes':list(converted),
                'unavailable_selected_scenes':list(unavailable),'original_source_preserved':True}),flush=True)
            return 'inspected_with_explicit_source_integrity_exclusions'
