import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy,hashlib,tempfile,time,unittest
from noposplat_jittor.source_store import allocate_sources,SourceStore

TEMP=Path(os.environ.get('NPS_TEST_TMP',str(Path(__file__).resolve().parents[1]/'.test_tmp')));TEMP.mkdir(exist_ok=True)

class PermanentSourceStore(unittest.TestCase):
    def fixture(self,root):
        chunks=[];payloads={}
        for i,size in enumerate((1000,800,600,500,400)):
            data=bytes([i+1])*size;name=f'train/{i}.torch';payloads[name]=data
            chunks.append({'path':name,'bytes':size,'sha256':hashlib.sha256(data).hexdigest(),'scenes':[str(i)],'url':'https://huggingface.co/example'})
        plan={'dataset':'fixture','repository':'test/fixture','commit':'fixed','stage':'train','chunks':chunks}
        stores={'local':{'root':str(root/'local'),'max_bytes':2000,'reserve_bytes':0},'shared':{'root':str(root/'shared'),'max_bytes':1800,'reserve_bytes':0}}
        allocation=allocate_sources([plan],stores)
        def downloader(row,path,**kwargs):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payloads[row['path']])
        return plan,allocation,downloader
    def test_all_sources_fit_and_survive_reopen_without_redownload(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,allocation,download=self.fixture(root);store=SourceStore(allocation)
            paths=[store.ensure(plan,row,downloader=download) for row in plan['chunks']]
            self.assertEqual(len(allocation['chunks']),5);self.assertEqual(sum(allocation['allocated_bytes'].values()),3300)
            again=SourceStore(allocation)
            for row,path in zip(plan['chunks'],paths):self.assertEqual(again.ensure(plan,row,downloader=lambda *a,**k:self.fail('No redownload')),path)
            self.assertTrue(all(path.is_file() for path in paths))
    def test_insufficient_total_capacity_does_not_publish_partial_selection(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,allocation,_=self.fixture(root);stores=copy.deepcopy(allocation['stores'])
            for x in stores.values():x['max_bytes']=1100
            with self.assertRaises(OSError):allocate_sources([plan],stores)
    def test_unknown_source_and_changed_payload_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,allocation,download=self.fixture(root);store=SourceStore(allocation);row=plan['chunks'][0];path=store.ensure(plan,row,downloader=download);path.write_bytes(b'bad')
            with self.assertRaises(ValueError):store.ensure(plan,row,downloader=download)
            changed=copy.deepcopy(plan);changed['commit']='different'
            with self.assertRaises(ValueError):store.ensure(changed,row,downloader=download)
    def test_unmounted_shared_root_is_not_written_to_system_disk(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);_,allocation,_=self.fixture(root);allocation['stores']['local']['required_mount']=str(root/'not-a-mount')
            with self.assertRaises(OSError):SourceStore(allocation)
            self.assertFalse((root/'local').exists())
    def test_unowned_store_directory_is_never_adopted(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);_,allocation,_=self.fixture(root);(root/'local').mkdir();(root/'local/keep').write_text('user')
            with self.assertRaises(FileExistsError):SourceStore(allocation)
            self.assertEqual((root/'local/keep').read_text(),'user')
    def test_concurrent_readers_download_one_source_once(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,allocation,download=self.fixture(root);store=SourceStore(allocation);calls=[]
            def delayed(row,path,**kwargs):calls.append(1);time.sleep(.05);download(row,path,**kwargs)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(lambda _:store.ensure(plan,plan['chunks'][0],downloader=delayed),range(2)))
            self.assertEqual(len(calls),1);self.assertEqual(results[0],results[1])
    def test_source_published_before_ledger_interruption_is_recovered(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,allocation,download=self.fixture(root);store=SourceStore(allocation);row=plan['chunks'][0]
            def interrupted(row,path,**kwargs):download(row,path,**kwargs);raise OSError('Simulated interruption before ledger commit')
            with self.assertRaises(OSError):store.ensure(plan,row,downloader=interrupted)
            path=store.ensure(plan,row,downloader=lambda *a,**k:self.fail('Already verified bytes must be reused'));self.assertTrue(path.is_file())
    def test_changed_transport_url_never_makes_a_request(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);plan,allocation,_=self.fixture(root);store=SourceStore(allocation);row=copy.deepcopy(plan['chunks'][0]);row['url']='http://untrusted.invalid'
            with self.assertRaises(ValueError):store.ensure(plan,row,downloader=lambda *a,**k:self.fail('Unpinned transport must not be requested'))

if __name__=='__main__':unittest.main()
