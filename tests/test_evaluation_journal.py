import os
from pathlib import Path
import tempfile,unittest
from noposplat_jittor.evaluation_journal import EvaluationJournal

TEMP=Path(os.environ.get('NPS_TEST_TMP',str(Path(__file__).resolve().parents[1]/'.test_tmp')));TEMP.mkdir(exist_ok=True)

class EvaluationRecovery(unittest.TestCase):
    def rows(self,key,targets):return [dict(scene=key,target=t,psnr=25.,ssim=.8,lpips=.1) for t in targets]
    def test_partial_scene_is_not_completion_and_resume_preserves_order(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d)/'evaluation';binding={'weights_sha256':'immutable'};j=EvaluationJournal(root,binding,['b','a'])
            with self.assertRaises(ValueError):j.commit('a',self.rows('a',[1]),[1,2],'attempt')
            self.assertFalse(j.progress('running')['complete']);j.commit('a',self.rows('a',[1,2]),[1,2],'attempt')
            resumed=EvaluationJournal(root,binding,['b','a'],resume=True);self.assertEqual(len(resumed.completed),1)
            resumed.commit('b',self.rows('b',[3]),[3],'other');self.assertTrue(resumed.progress('completed')['complete']);self.assertEqual([r['scene'] for r in resumed.rows()],['b','a','a'])
    def test_different_weights_or_protocol_rejects_mixing_results(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d)/'evaluation';EvaluationJournal(root,{'weights':'one'},['scene'])
            with self.assertRaises(ValueError):EvaluationJournal(root,{'weights':'two'},['scene'],resume=True)
    def test_nonfinite_and_duplicate_results_are_refused(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            j=EvaluationJournal(Path(d)/'evaluation',{},['scene']);rows=self.rows('scene',[1]);rows[0]['lpips']=float('nan')
            with self.assertRaises(ValueError):j.commit('scene',rows,[1],'attempt')
            j.commit('scene',self.rows('scene',[1]),[1],'attempt')
            with self.assertRaises(FileExistsError):j.commit('scene',self.rows('scene',[1]),[1],'attempt')

if __name__=='__main__':unittest.main()
