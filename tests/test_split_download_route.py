from urllib.request import Request
import unittest
from noposplat_jittor.verified_download import open_public_cdn_range


class SplitRoute(unittest.TestCase):
    def test_signed_range_is_ephemeral_and_preserves_exact_request(self):
        final='https://us.aws.cdn.hf.co/public-file?ephemeral-signature=test'
        class Reply:
            status=206
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def geturl(self):return final
        called=[]
        def direct(request,timeout):
            called.append((request,timeout));return 'body-response'
        request=Request('https://huggingface.co/public/resolve/fixed/chunk',headers={'Range':'bytes=16-31'})
        actual=open_public_cdn_range(request,60,resolver=lambda *a,**k:Reply(),direct=direct)
        self.assertEqual(actual,'body-response');self.assertEqual(called[0][0].get_header('Range'),'bytes=16-31')
        self.assertEqual(called[0][0].full_url,final)

    def test_unexpected_redirect_is_rejected_before_fetch(self):
        class Reply:
            status=206
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def geturl(self):return 'https://unrelated.invalid/file'
        with self.assertRaisesRegex(ValueError,'redirect host'):
            open_public_cdn_range(Request('https://huggingface.co/public/file'),10,resolver=lambda *a,**k:Reply(),
                direct=lambda *a,**k:self.fail('Unexpected redirect must not be fetched'))

    def test_noncanonical_origin_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'canonical'):
            open_public_cdn_range(Request('https://unrelated.invalid/file'),10,resolver=lambda *a,**k:self.fail('Origin rejected first'))


if __name__=='__main__':unittest.main()
