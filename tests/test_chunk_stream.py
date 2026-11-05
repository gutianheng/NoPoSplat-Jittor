import os
from pathlib import Path
from io import BytesIO
import copy,hashlib,json,tempfile,unittest
import numpy as np
from PIL import Image
from noposplat_jittor.chunk_stream import BoundedChunkCache,ChunkSceneStream,single_chunk_plan
from noposplat_jittor.dataset_conversion import ChunkConversion

TEMP=Path(os.environ.get('NPS_TEST_TMP',str(Path(__file__).resolve().parents[1]/'.test_tmp')));TEMP.mkdir(exist_ok=True)


class ChunkStreaming(unittest.TestCase):
    def fixture(self,root,scenes_per_chunk=1):
        rows=[];payloads={};sources={}
        for i in range(4):
            source=root/f'{i}.torch';source.write_bytes(bytes([i+1])*(180000*scenes_per_chunk));sources[i]=source
            keys=[f'scene{i}_{j}' for j in range(scenes_per_chunk)]
            rows.append({'path':f'train/{i}.torch','bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scenes':keys,'url':'https://huggingface.co/test/fixture'})
            for j,key in enumerate(keys):
                cameras=np.zeros((8,18),np.float32);cameras[:,:4]=[1,1,.5,.5]
                cameras[:,6:]=np.eye(4,dtype=np.float32)[:3].reshape(-1);cameras[:,9]=-np.arange(8)*.05
                images=[]
                for k in range(8):
                    image=np.full((32,48,3),[i*40+j*10,k*25,100],np.uint8);buffer=BytesIO();Image.fromarray(image).save(buffer,format='JPEG')
                    data=buffer.getvalue();images.append(np.frombuffer(data+b'\0'*(40000-len(data)),np.uint8))
                payloads[key]=(cameras,images)
        plan={'dataset':'fixture','stage':'train','repository':'test/fixture','commit':'immutable-test','selection':'all_fixture_scenes','chunks':rows}
        planfile=root/'plan.json';planfile.write_text(json.dumps(plan))
        def prepare(folder,whole,row):
            subset=single_chunk_plan(whole,row);conversion=ChunkConversion(folder/'converted'/whole['stage'],subset)
            def loader(_):
                for key in row['scenes']:yield key,*payloads[key]
            conversion.convert(row,sources[int(Path(row['path']).stem)],loader)
        cfg={'source_plans':{'train':str(planfile)},'chunk_stream':{'cache_root':str(root/'cache'),'prefetch':0},'near':.1,'far':100.,'baseline_min':.001,'baseline_max':1e10,
             'max_fov':100.,'original_shape':[32,48],'image_shape':[16,16],'gap_warmup':0,'gap_min':2,'gap_max':4,'initial_gap_min':2,'initial_gap_max':2,'target_views':1,'augment':True}
        return plan,cfg,prepare

    def test_lru_never_evicts_a_live_reader(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,_,prepare=self.fixture(root);cache=BoundedChunkCache(root/'cache',1000000,0,preparer=prepare);rows=plan['chunks']
            with cache.lease(plan,rows[0]) as first:
                cache.prepare(plan,rows[1]);cache.prepare(plan,rows[2])
                self.assertTrue(first.is_dir());self.assertNotIn(cache._id(plan,rows[1]),cache._state()['entries'])
                with cache.lease(plan,rows[2]):
                    with self.assertRaises(OSError):cache.prepare(plan,rows[3])
                self.assertLessEqual(sum(x['accounted_bytes'] for x in cache._state()['entries'].values()),1000000)
            cache.prepare(plan,rows[3]);self.assertTrue((root/'cache/audit.jsonl').exists())

    def test_changed_payload_cannot_be_leased(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,_,prepare=self.fixture(root);cache=BoundedChunkCache(root/'cache',2000000,0,preparer=prepare);row=plan['chunks'][0]
            path=cache.prepare(plan,row);next(path.glob('*.npz')).write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                with cache.lease(plan,row):self.fail('Corrupt cache must not reach the reader')

    def test_resume_preserves_next_scene_frames_and_augmentation(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,cfg,prepare=self.fixture(root,2);cache=BoundedChunkCache(root/'cache',5000000,0,preparer=prepare)
            cfg['chunk_stream']['prefetch']=2
            stream=ChunkSceneStream(cfg,'train',111123,cache=cache)
            try:
                for _ in range(3):stream.train_example(12000)
                state=stream.state_dict();expected=[stream.train_example(12000) for _ in range(12)]
            finally:stream.close()
            resumed=ChunkSceneStream(cfg,'train',99,cache=cache)
            try:
                resumed.load_state_dict(state);actual=[resumed.train_example(12000) for _ in range(12)]
            finally:resumed.close()
            for a,b in zip(expected,actual):
                self.assertEqual(a['scene'],b['scene'])
                for kind in ('context','target'):
                    for key in a[kind]:np.testing.assert_array_equal(a[kind][key],b[kind][key])
            self.assertEqual(len({x['scene'] for x in expected}),8)

    def test_resume_rejects_a_different_source_commit(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,cfg,prepare=self.fixture(root);cache=BoundedChunkCache(root/'cache',3000000,0,preparer=prepare)
            stream=ChunkSceneStream(cfg,'train',cache=cache);state=stream.state_dict();stream.close()
            plan['commit']='changed';Path(cfg['source_plans']['train']).write_text(json.dumps(plan));other=ChunkSceneStream(cfg,'train',cache=cache)
            try:
                with self.assertRaises(ValueError):other.load_state_dict(state)
            finally:other.close()

    def test_crash_after_conversion_can_recover_without_rewriting_payload(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,_,prepare=self.fixture(root);attempts=[]
            def interrupted(folder,whole,row):
                prepare(folder,whole,row);attempts.append(1)
                if len(attempts)==1:raise OSError('Simulated interruption after durable scene publication')
            cache=BoundedChunkCache(root/'cache',2000000,0,preparer=interrupted);row=plan['chunks'][0]
            with self.assertRaises(OSError):cache.prepare(plan,row)
            existing=next((cache._entry(cache._id(plan,row))/'converted/train').glob('*.npz'));before=existing.read_bytes()
            with cache.lease(plan,row):self.assertEqual(before,existing.read_bytes())

    def test_unowned_directory_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);(root/'keep.txt').write_text('user file')
            with self.assertRaises(FileExistsError):BoundedChunkCache(root,1000000,0)
            self.assertEqual((root/'keep.txt').read_text(),'user file')

    def test_missing_official_test_scene_is_not_skipped(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,cfg,prepare=self.fixture(root);plan['stage']='test'
            for row in plan['chunks']:row['path']=row['path'].replace('train/','test/')
            file=root/'test_plan.json';file.write_text(json.dumps(plan));cfg['source_plans']['test']=str(file)
            index=root/'fixed.json';index.write_text(json.dumps({'missing_scene':{'context':[0,2],'target':[1]}}))
            cache=BoundedChunkCache(root/'cache',3000000,0,preparer=prepare)
            with self.assertRaises(ValueError):ChunkSceneStream(cfg,'test',index=index,cache=cache)


if __name__=='__main__':unittest.main()
