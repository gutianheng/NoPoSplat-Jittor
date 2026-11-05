import io,pickle,struct,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from tools.read_selected_torch_archive import MetadataOnly,selected_scenes
from tools.checked_candidate_conversion import CheckedCandidateConversion
from noposplat_jittor.dataset_conversion import sha256_file
try:import torch
except ImportError:torch=None

@unittest.skipIf(torch is None,'Torch is used only to create independent CPU archive fixtures')
class SelectedArchiveIntegrity(unittest.TestCase):
    def fixture(self,root):
        cameras=torch.arange(72,dtype=torch.float32).reshape(4,18)[::2]
        data=[{'key':'a','cameras':cameras,'images':[torch.arange(80,dtype=torch.uint8)[4:40:2],torch.arange(20,dtype=torch.uint8)]}]
        path=root/'source.torch';torch.save(data,path);return path,data
    def test_strided_storage_matches_torch_without_reencoding(self):
        with tempfile.TemporaryDirectory() as d:
            path,data=self.fixture(Path(d));key,cameras,images=list(selected_scenes(path,{'a'}))[0]
            self.assertEqual(key,'a');np.testing.assert_array_equal(cameras,data[0]['cameras'].numpy())
            for left,right in zip(images,data[0]['images']):self.assertEqual(left.tobytes(),right.numpy().tobytes())
    def test_unknown_pickle_global_refused(self):
        import os
        with self.assertRaises(pickle.UnpicklingError):MetadataOnly(io.BytesIO(pickle.dumps(os.system))).load()
    def test_corrupt_selected_member_is_explicit_unavailable_not_fake_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,data=self.fixture(root)
            with zipfile.ZipFile(path) as z:member=z.getinfo('source/data/1')
            with path.open('r+b') as f:
                f.seek(member.header_offset);head=f.read(30);n,e=struct.unpack_from('<HH',head,26)
                f.seek(member.header_offset+30+n+e+3);value=f.read(1);f.seek(-1,1);f.write(bytes([value[0]^1]))
            row={'path':'test/000001.torch','bytes':path.stat().st_size,'sha256':sha256_file(path),'scenes':['a']}
            plan={'dataset':'acid','stage':'test','commit':'fixture','repository':'fixture','chunks':[row]}
            converter=CheckedCandidateConversion(root/'out',plan)
            def refuses(_):raise pickle.UnpicklingError('Simulated Torch archive format failure')
            converter.convert(row,path,refuses)
            self.assertTrue(converter.verify_committed(row['path']))
            self.assertEqual(converter.state['chunks'][row['path']]['scenes'],{})
            self.assertIn('a',converter.state['chunks'][row['path']]['unavailable_scenes'])
            self.assertFalse(list((root/'out').glob('*.npz')));self.assertTrue(path.exists())

if __name__=='__main__':unittest.main()
