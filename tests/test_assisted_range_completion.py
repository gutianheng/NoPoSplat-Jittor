import os
from pathlib import Path
import hashlib,tempfile,unittest
from noposplat_jittor.verified_download import parallel_download

TEMP=Path(os.environ.get('NPS_TEST_TMP',str(Path(__file__).resolve().parents[1]/'.test_tmp')));TEMP.mkdir(exist_ok=True)


class AssistedRangeCompletion(unittest.TestCase):
    def test_full_verified_publication_supersedes_slow_partial_transport(self):
        raw=b'full pinned source'
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            target=Path(d)/'chunk'
            class Reply:
                status=206;headers={'Content-Range':f'bytes 0-3/{len(raw)}'}
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def read1(self,count):target.write_bytes(raw);return b''
            row={'url':'https://huggingface.co/public/fixed','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            parallel_download(row,target,workers=1,range_bytes=4,open_url=lambda *a,**k:Reply())
            self.assertEqual(target.read_bytes(),raw);self.assertFalse(target.with_name('chunk.ranges').exists())


if __name__=='__main__':unittest.main()
