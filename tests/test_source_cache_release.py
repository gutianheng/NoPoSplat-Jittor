import os
from pathlib import Path
import hashlib,tempfile,unittest
import numpy as np
from noposplat_jittor.dataset_conversion import ChunkConversion
from tools.prepare_reduced_experiments import release_verified_source

TEMP=Path(os.environ.get('NPS_TEST_TMP',str(Path(__file__).resolve().parents[1]/'.test_tmp')));TEMP.mkdir(exist_ok=True)


class SourceCacheRelease(unittest.TestCase):
    def fixture(self,base):
        cache=base/'cache';cache.mkdir();source=cache/'chunk.torch';source.write_bytes(b'verified raw bytes')
        row={'path':'test/chunk.torch','scenes':['scene'],'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        plan={'stage':'test','repository':'test','commit':'fixed','chunks':[row]};conversion=ChunkConversion(base/'converted',plan)
        def loader(file):yield 'scene',np.zeros((2,18),np.float32),[np.array([1,2],np.uint8),np.array([3],np.uint8)]
        return cache,source,row,conversion,loader

    def test_committed_archives_remain_valid_after_raw_release(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            cache,source,row,conversion,loader=self.fixture(Path(d));conversion.convert(row,source,loader)
            release_verified_source(source,row,conversion,cache)
            self.assertFalse(source.exists());self.assertTrue(conversion.verify_committed(row['path']))
            self.assertTrue((conversion.root/'source_cache_cleanup.json').exists())

    def test_uncommitted_or_changed_source_is_never_deleted(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            cache,source,row,conversion,loader=self.fixture(Path(d))
            with self.assertRaises(ValueError):release_verified_source(source,row,conversion,cache)
            self.assertTrue(source.exists());conversion.convert(row,source,loader)
            source.write_bytes(b'x'*row['bytes'])
            with self.assertRaises(ValueError):release_verified_source(source,row,conversion,cache)
            self.assertTrue(source.exists())

    def test_source_outside_cache_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            cache,source,row,conversion,loader=self.fixture(Path(d));conversion.convert(row,source,loader)
            other=Path(d)/'outside.torch';other.write_bytes(source.read_bytes())
            with self.assertRaises(ValueError):release_verified_source(other,row,conversion,cache)
            self.assertTrue(other.exists())


if __name__=='__main__':unittest.main()
