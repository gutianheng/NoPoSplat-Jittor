import os
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError,URLError
import hashlib,tempfile,unittest
from noposplat_jittor.verified_download import parallel_download,is_transient_download_error


class TransientDownloads(unittest.TestCase):
    def test_503_retries_honor_server_delay_then_verify_full_sha(self):
        raw=b'full pinned bytes';calls=[]
        class Reply:
            status=206;headers={'Content-Range':f'bytes 0-{len(raw)-1}/{len(raw)}'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,count):return raw
        def opener(request,timeout):
            calls.append(request)
            if len(calls)==1:raise HTTPError(request.full_url,503,'Unavailable',{'Retry-After':'45'},None)
            return Reply()
        root=Path(os.environ.get('NPS_TEST_TMP',str(Path(__file__).resolve().parents[1]/'.test_tmp')));root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as d,patch('noposplat_jittor.verified_download._set_network_pause') as pause:
            target=Path(d)/'chunk';row={'url':'https://huggingface.co/public/fixed','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            parallel_download(row,target,workers=1,range_bytes=len(raw),open_url=opener)
            pause.assert_called_once_with(45.);self.assertEqual(target.read_bytes(),raw)

    def test_protocol_integrity_and_storage_errors_are_not_retryable(self):
        for error in (ValueError('SHA mismatch'),OSError(28,'No space'),HTTPError('https://example.invalid',404,'Missing',{},None)):
            self.assertFalse(is_transient_download_error(error))
        self.assertTrue(is_transient_download_error(HTTPError('https://example.invalid',503,'Unavailable',{},None)))
        self.assertTrue(is_transient_download_error(URLError('Connection reset')))


if __name__=='__main__':unittest.main()
