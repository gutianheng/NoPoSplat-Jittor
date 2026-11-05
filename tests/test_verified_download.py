import os
from pathlib import Path
from io import BytesIO
import hashlib,json,tempfile,unittest
from noposplat_jittor.verified_download import parallel_download

TEMP=Path(os.environ.get('NPS_TEST_TMP',str(Path(__file__).resolve().parents[1]/'.test_tmp')));TEMP.mkdir(exist_ok=True)


class Response(BytesIO):
    status=206
    def __init__(self,data,header):super().__init__(data);self.headers={'Content-Range':header}


class RangeDownload(unittest.TestCase):
    def test_exact_source_and_completed_range_resume(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            payload=bytes(range(255))*19;path=Path(d)/'source.torch';row={'url':'https://huggingface.co/example','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()};calls=[]
            def fetch(request,timeout):
                lo,hi=map(int,request.headers['Range'].removeprefix('bytes=').split('-'));calls.append((lo,hi));return Response(payload[lo:hi+1],f'bytes {lo}-{hi}/{len(payload)}')
            parallel_download(row,path,workers=3,range_bytes=700,open_url=fetch)
            self.assertEqual(path.read_bytes(),payload);n=len(calls)
            parallel_download(row,path,workers=3,range_bytes=700,open_url=fetch);self.assertEqual(len(calls),n)

    def test_wrong_content_range_never_publishes(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            path=Path(d)/'source.torch';row={'url':'https://huggingface.co/example','bytes':8,'sha256':hashlib.sha256(b'abcdefgh').hexdigest()}
            with self.assertRaises(ValueError):parallel_download(row,path,workers=1,range_bytes=4,open_url=lambda *a,**k:Response(b'abcd','bytes 0-3/999'))
            self.assertFalse(path.exists())

    def test_full_hash_failure_never_publishes(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            path=Path(d)/'source.torch';row={'url':'https://huggingface.co/example','bytes':8,'sha256':hashlib.sha256(b'abcdefgh').hexdigest()}
            def wrong(request,timeout):
                lo,hi=map(int,request.headers['Range'].removeprefix('bytes=').split('-'));return Response(b'x'*(hi-lo+1),f'bytes {lo}-{hi}/8')
            with self.assertRaises(ValueError):parallel_download(row,path,workers=2,range_bytes=4,open_url=wrong)
            self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
