"""CPU tests for audit and aggregation: never substitutes for Jittor execution."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'.test_tmp').mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
from noposplat_jittor.evaluation_summary import overlap_metadata,summarize_nvs
spec=importlib.util.spec_from_file_location('preflight',ROOT/'tools/preflight.py')
preflight=importlib.util.module_from_spec(spec);spec.loader.exec_module(preflight)


class Preparation(unittest.TestCase):
    def test_overlap_boundaries_float32(self):
        cases=[(.05,'small'),(.3,'small'),(.30001,'medium'),(.55,'medium'),(.55001,'large'),(.8,'large'),(.80001,'ignore')]
        for value,tag in cases:
            self.assertEqual(overlap_metadata({'overlap':float(np.float32(value))})['overlap_group'],tag)

    def test_overlap_original_fallback_and_quirk(self):
        for value,tag in [('large','large'),('small','small'),(None,'small'),(.01,'medium')]:
            self.assertEqual(overlap_metadata({'overlap':value})['overlap_group'],tag)
        self.assertEqual(overlap_metadata({})['overlap_group'],'unknown')

    def test_bad_overlap_refused(self):
        with self.assertRaises(ValueError):overlap_metadata({'overlap':float('nan')})

    def test_aggregation_distinguishes_scene_and_image(self):
        def row(scene,target,value,group):return dict(scene=scene,target=target,psnr=value,ssim=value,lpips=value,overlap_group=group)
        summary=summarize_nvs([row('a',0,1.,'small'),row('a',1,3.,'small'),row('b',0,8.,'large')])
        self.assertEqual(summary['scene_mean']['psnr'],5.)
        self.assertEqual(summary['image_mean']['psnr'],4.)
        self.assertEqual(summary['overlap_groups']['small']['scene_mean']['psnr'],2.)
        self.assertNotIn('medium',summary['overlap_groups'])

    def test_empty_duplicate_nonfinite_results_refused(self):
        row=dict(scene='a',target=0,psnr=1.,ssim=1.,lpips=1.)
        for rows in ([],[row,row],[{**row,'lpips':float('inf')}]):
            with self.assertRaises(ValueError):summarize_nvs(rows)

    def test_explicit_target_slots_preserve_repeated_official_frames(self):
        row=dict(scene='a',target=12,psnr=1.,ssim=1.,lpips=1.)
        rows=[{**row,'target_slot':0},{**row,'target_slot':1,'psnr':3.}]
        result=summarize_nvs(rows)
        self.assertEqual(result['target_count'],2)
        self.assertEqual(result['scene_mean']['psnr'],2.)
        with self.assertRaises(ValueError):summarize_nvs([rows[0],rows[0]])

    def test_inventory_checks_real_chunks_not_just_index_membership(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.test_tmp') as temp:
            root=Path(temp);(root/'test').mkdir()
            (root/'test/index.json').write_text(json.dumps({'a':'one.torch','b':'missing.torch'}))
            (root/'test/one.torch').write_bytes(b'not unpickled')
            fixed=root/'evaluation.json';fixed.write_text(json.dumps({'a':{'context':[0,1],'target':[2]},'b':{'context':[0,1],'target':[2]}}))
            report=preflight.inspect_dataset(root,fixed)
            self.assertEqual(report['stages']['test']['requested_scenes_with_existing_chunk'],1)
            self.assertEqual(report['stages']['test']['missing_indexed_chunks'],1)
            self.assertFalse(report['stages']['test']['full_requested_coverage'])

    def test_index_path_cannot_escape_dataset(self):
        with self.assertRaises(ValueError):preflight.safe_child(ROOT/'data','../outside.torch')


if __name__=='__main__':unittest.main(verbosity=2)
