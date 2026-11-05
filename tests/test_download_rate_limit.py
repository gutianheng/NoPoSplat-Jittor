import os
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
import hashlib,tempfile,unittest
from noposplat_jittor.verified_download import parallel_download,retry_after_seconds

TEMP=Path(os.environ.get('NPS_TEST_TMP',str(Path(__file__).resolve().parents[1]/'.test_tmp')));TEMP.mkdir(exist_ok=True)


class RateLimitPolicy(unittest.TestCase):
    def test_server_retry_after_is_respected(self):
        self.assertEqual(retry_after_seconds({'Retry-After':'120'},0),120.)
        self.assertGreaterEqual(retry_after_seconds({},2),15.)

    def test_retry_then_full_source_hash_verification(self):
        raw=b'abcdefgh';calls=[]
        class Reply:
            status=206;headers={'Content-Range':'bytes 0-7/8'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,count):return raw
        def opener(request,timeout):
            calls.append(request)
            if len(calls)==1:raise HTTPError(request.full_url,429,'Too Many Requests',{'Retry-After':'2'},None)
            return Reply()
        with tempfile.TemporaryDirectory(dir=TEMP) as d,patch('noposplat_jittor.verified_download._set_network_pause') as pause:
            target=Path(d)/'chunk';row={'url':'https://huggingface.co/test/resolve/fixed/chunk','bytes':8,'sha256':hashlib.sha256(raw).hexdigest()}
            parallel_download(row,target,workers=1,range_bytes=8,open_url=opener)
            pause.assert_called_once_with(2.);self.assertEqual(target.read_bytes(),raw)


if __name__=='__main__':unittest.main()
