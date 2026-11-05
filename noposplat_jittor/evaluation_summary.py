"""Overlap metadata and scene/target aggregation; no framework dependency."""
import math
import numpy as np

METRICS=('psnr','ssim','lpips')


def overlap_metadata(entry):
    if 'overlap' not in entry:return {'overlap':None,'overlap_group':'unknown'}
    value=entry['overlap']
    # Same string/None fallback as the author's ViewSamplerEvaluation.sample.
    value=value if isinstance(value,float) else .75 if value=='large' else .25
    value=np.float32(value)
    if not np.isfinite(value):raise ValueError('Nonfinite evaluation overlap')
    # Float32 thresholds mirror comparisons on the original torch.float32 tensor.
    # Preserve the author's <.05 -> medium quirk; report such indices in preflight.
    if np.float32(.05)<=value<=np.float32(.3):tag='small'
    elif value<=np.float32(.55):tag='medium'
    elif value<=np.float32(.8):tag='large'
    else:tag='ignore'
    return {'overlap':float(value),'overlap_group':tag}


def summarize_nvs(rows):
    if not rows:raise ValueError('No evaluated targets; cannot report metrics')
    per_scene={}
    seen=set()
    for row in rows:
        identity=(row['scene'],row.get('target_slot',row['target']))
        if identity in seen:raise ValueError(f'Duplicate evaluated target: {identity}')
        seen.add(identity)
        if not all(math.isfinite(row[k]) for k in METRICS):raise ValueError('Nonfinite evaluation metric')
        per_scene.setdefault(row['scene'],[]).append(row)
    scenes=[]
    for scene,targets in per_scene.items():
        if any('target_slot' in t for t in targets) and sorted(t.get('target_slot',-1) for t in targets)!=list(range(len(targets))):
            raise ValueError('Missing/duplicate target slots within scene')
        tags={t.get('overlap_group','unknown') for t in targets}
        if len(tags)!=1:raise ValueError('Inconsistent overlap group within one scene')
        scenes.append({'scene':scene,'targets':len(targets),'overlap':targets[0].get('overlap'),
                       'overlap_group':tags.pop(),**{k:float(np.mean([t[k] for t in targets])) for k in METRICS}})
    def means(scene_list,target_list):
        return {'scene_count':len(scene_list),'target_count':len(target_list),
                'scene_mean':{k:float(np.mean([r[k] for r in scene_list])) for k in METRICS},
                'image_mean':{k:float(np.mean([r[k] for r in target_list])) for k in METRICS}}
    groups={}
    for tag in ('small','medium','large','ignore','unknown'):
        ss=[s for s in scenes if s['overlap_group']==tag]
        rr=[r for r in rows if r.get('overlap_group','unknown')==tag]
        if ss:groups[tag]=means(ss,rr)
    return {**means(scenes,rows),'overlap_groups':groups,'per_scene':scenes,
            'aggregation_note':'scene_mean gives each scene equal weight; image_mean gives each target equal weight. No missing group is fabricated.'}
