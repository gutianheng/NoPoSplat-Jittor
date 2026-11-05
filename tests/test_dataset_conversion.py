import os
from pathlib import Path
import hashlib,json,tempfile,unittest
from unittest.mock import patch
import numpy as np
from noposplat_jittor.dataset_conversion import ChunkConversion,scene_arrays,write_scene

TEMP=Path(os.environ.get('NPS_TEST_TMP',str(Path(__file__).resolve().parents[1]/'.test_tmp')));TEMP.mkdir(exist_ok=True)


class StreamingConversion(unittest.TestCase):
    def fixture(self,root):
        paths=[];rows=[]
        for i in range(2):
            p=root/f'{i}.torch';p.write_bytes(f'encoded chunk {i}'.encode());paths.append(p)
            rows.append({'path':f'train/{i}.torch','bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scenes':[f'scene{i}']})
        plan={'repository':'test/fixture','commit':'test-only','stage':'train','chunks':rows}
        cams=np.zeros((2,18),np.float32);cams[:,:4]=[1,1,.5,.5];cams[:,6:]=np.eye(4,dtype=np.float32)[:3].reshape(-1)
        images=[np.array([255,216,1,2],dtype=np.uint8),np.array([255,216,3,4,5],dtype=np.uint8)]
        def loader(path):yield 'scene'+Path(path).stem,cams,images
        return paths,plan,cams,images,loader

    def test_resume_and_exact_payload(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);paths,plan,cameras,images,loader=self.fixture(root);out=root/'out';converter=ChunkConversion(out,plan)
            converter.convert(plan['chunks'][0],paths[0],loader);self.assertEqual(json.loads((out/'conversion.json').read_text())['status'],'partial')
            resumed=ChunkConversion(out,plan);self.assertTrue(resumed.verify_committed('train/0.torch'));resumed.convert(plan['chunks'][1],paths[1],loader)
            self.assertEqual(json.loads((out/'conversion.json').read_text())['status'],'complete')
            index=json.loads((out/'index.json').read_text());self.assertEqual(len(index),2)
            with np.load(out/index['scene0'],allow_pickle=False) as z:
                np.testing.assert_array_equal(z['cameras'],cameras);np.testing.assert_array_equal(z['image_bytes'],np.concatenate(images));np.testing.assert_array_equal(z['offsets'],[0,4,9])

    def test_corrupt_source_never_deserialized(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);paths,plan,*_=self.fixture(root);conversion=ChunkConversion(root/'out',plan);paths[0].write_bytes(b'wrong')
            with self.assertRaises(ValueError):conversion.convert(plan['chunks'][0],paths[0],lambda unused:self.fail('Must not deserialize corrupt source'))
            self.assertEqual(conversion.state['chunks'],{})

    def test_recover_payload_written_before_chunk_commit(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);paths,plan,cameras,images,loader=self.fixture(root);out=root/'out';conversion=ChunkConversion(out,plan)
            filename=hashlib.sha256(b'scene0').hexdigest()+'.npz';write_scene(out/filename,scene_arrays('scene0',cameras,images));before=(out/filename).read_bytes()
            recovered=ChunkConversion(out,plan);recovered.convert(plan['chunks'][0],paths[0],loader);self.assertEqual(before,(out/filename).read_bytes())

    def test_rebuild_index_from_committed_ledger(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);paths,plan,_,_,loader=self.fixture(root);out=root/'out';conversion=ChunkConversion(out,plan)
            original=conversion.publish
            with patch.object(conversion,'publish',side_effect=OSError('simulated failure after ledger commit')):
                with self.assertRaises(OSError):conversion.convert(plan['chunks'][0],paths[0],loader)
            recovered=ChunkConversion(out,plan);self.assertTrue(recovered.verify_committed('train/0.torch'));self.assertEqual(len(json.loads((out/'index.json').read_text())),1)

    def test_missing_scene_does_not_mark_chunk_complete(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);paths,plan,*_=self.fixture(root);conversion=ChunkConversion(root/'out',plan)
            with self.assertRaises(ValueError):conversion.convert(plan['chunks'][0],paths[0],lambda path:iter(()))
            self.assertFalse(conversion.verify_committed('train/0.torch'))

    def test_modified_payload_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);paths,plan,_,_,loader=self.fixture(root);out=root/'out';conversion=ChunkConversion(out,plan);conversion.convert(plan['chunks'][0],paths[0],loader)
            record=next(iter(conversion.state['chunks']['train/0.torch']['scenes'].values()));(out/record['file']).write_bytes(b'bad')
            with self.assertRaises(ValueError):ChunkConversion(out,plan).verify_committed('train/0.torch')

    def test_different_plan_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);_,plan,*_=self.fixture(root);ChunkConversion(root/'out',plan);plan['commit']='changed'
            with self.assertRaises(ValueError):ChunkConversion(root/'out',plan)


if __name__=='__main__':unittest.main()
