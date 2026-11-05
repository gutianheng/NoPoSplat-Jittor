import os
"""Cheap local tests: no Jittor, GPU, original weights or external downloads."""
import ast
from pathlib import Path
import sys
import unittest
import tempfile
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from noposplat_jittor.geometry_numpy import cameras_from_vectors,get_fov,normalize_cameras,quaternion_xyzw_to_matrix,retract_pose,projection_matrices
from noposplat_jittor.data import crop_resize,sample_indices,SceneStream
from noposplat_jittor.weights import canonical_key,read_arrays
from noposplat_jittor.metrics import psnr
from noposplat_jittor.optim import learning_rates
from noposplat_jittor.config import merge

TEMP_ROOT=Path(os.environ.get('NPS_TEST_TMP',str(Path(__file__).resolve().parents[1]/'.test_tmp')))
TEMP_ROOT.mkdir(exist_ok=True)


class Contracts(unittest.TestCase):
    def test_camera_vectors(self):
        poses=np.zeros((2,18),np.float32);poses[:,:4]=[.9,.9,.5,.5]
        poses[:,6:]=np.eye(4,dtype=np.float32)[:3].reshape(-1)
        poses[1,9]=-2
        ext,k=cameras_from_vectors(poses)
        self.assertAlmostEqual(float(ext[1,0,3]),2)
        np.testing.assert_allclose(k[0],[[.9,0,.5],[0,.9,.5],[0,0,1]])
    def test_baseline_canonical(self):
        ext=np.broadcast_to(np.eye(4),(3,4,4)).copy();ext[:,0,3]=[2,6,4]
        ext,n,f=normalize_cameras(ext,[0,1])
        np.testing.assert_allclose(ext[0],np.eye(4),atol=1e-7)
        np.testing.assert_allclose(ext[:,0,3],[0,1,.5]);self.assertAlmostEqual(n,.025);self.assertAlmostEqual(f,25.)
    def test_zero_baseline_rejected(self):
        with self.assertRaises(ValueError):normalize_cameras(np.stack([np.eye(4)]*2),[0,1])
    def test_xyzw(self):
        np.testing.assert_allclose(quaternion_xyzw_to_matrix([0,0,0,1]),np.eye(3))
        r=quaternion_xyzw_to_matrix(np.array([0,0,np.sqrt(.5),np.sqrt(.5)]))
        np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-7)
        cov=r@np.diag([1,4,9])@r.T
        np.testing.assert_allclose(np.linalg.eigvalsh(cov),[1,4,9],atol=1e-6)
    def test_fov(self):
        k=np.array([[[1,0,.5],[0,1,.5],[0,0,1]]],np.float32)
        np.testing.assert_allclose(get_fov(k),2*np.arctan(.5),atol=1e-7)
    def test_left_w2c_retraction(self):
        pose=retract_pose(np.eye(4),[.1,0,0],[0,0,0])
        np.testing.assert_allclose(pose[:3,3],[-.1,0,0])
        np.testing.assert_allclose(retract_pose(pose,[0,0,0],[0,0,0]),pose)
    def test_projection_storage(self):
        k=np.array([[1,0,.5],[0,1,.5],[0,0,1]],np.float32)
        view,full,p,tx,ty=projection_matrices(np.eye(4),k,1,10)
        np.testing.assert_allclose(view,np.eye(4));np.testing.assert_allclose(full,p)
        self.assertAlmostEqual(tx,.5,places=6);self.assertAlmostEqual(ty,.5,places=6)
        self.assertAlmostEqual(float(p[2,3]),1.)
    def test_crop_preserves_original_cx_convention(self):
        x=np.zeros((1,3,360,640),np.float32);k=np.array([[[1,0,.4],[0,1,.6],[0,0,1]]],np.float32)
        image,k=crop_resize(x,k,[256,256]);self.assertEqual(image.shape,(1,3,256,256))
        self.assertAlmostEqual(float(k[0,0,0]),455/256);self.assertAlmostEqual(float(k[0,1,1]),1.)
        self.assertAlmostEqual(float(k[0,0,2]),.4,places=6)
    def test_sampler_not_meters(self):
        cfg={'gap_warmup':9375,'initial_gap_min':25,'initial_gap_max':25,'gap_min':45,'gap_max':90,'target_views':1}
        rng=np.random.default_rng(1)
        ci,ti=sample_indices(100,0,cfg,rng);self.assertEqual(ci[1]-ci[0],25);self.assertTrue(ci[0]<=ti[0]<=ci[1])
        ci,ti=sample_indices(100,10000,cfg,rng);self.assertTrue(45<=ci[1]-ci[0]<=90)
    def test_weights_alias(self):
        self.assertEqual(canonical_key('module.encoder.downstream_head1.dpt.scratch.layer_rn.2.weight'),'downstream_head1.dpt.scratch.layer3_rn.weight')
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as d:
            path=Path(d)/'weights.npz';np.savez(path,**{'encoder.a':np.ones(2),'module.encoder.a':np.zeros(2)})
            with self.assertRaises(ValueError):read_arrays(path)
    def test_learning_rates(self):
        cfg={'lr':2e-4,'backbone_lr_multiplier':.1,'warmup':125,'max_steps':18751}
        a,b=learning_rates(0,cfg);self.assertAlmostEqual(a,2e-4/125);self.assertAlmostEqual(b,2e-5/125)
        a,b=learning_rates(125,cfg);self.assertAlmostEqual(a,2e-4);self.assertAlmostEqual(b,2e-5)
        a,b=learning_rates(125+18751,cfg);self.assertAlmostEqual(a,2e-5);self.assertAlmostEqual(b,2e-5)
    def test_psnr(self):
        np.testing.assert_allclose(psnr(np.zeros((1,3,4,4),np.float32),np.full((1,3,4,4),.1,np.float32)),20,atol=1e-5)
    def test_config_merge(self):
        a={'x':{'a':1,'b':2}};b=merge(a,{'x':{'b':3}});self.assertEqual(b,{'x':{'a':1,'b':3}});self.assertEqual(a['x']['b'],2)
    def test_missing_eval_scenes_fail(self):
        cfg={}
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as d:
            root=Path(d);(root/'test').mkdir();(root/'test/index.json').write_text('{"a":"a.npz"}')
            (root/'eval.json').write_text('{"b":{"context":[0,1],"target":[2]}}')
            with self.assertRaises(ValueError):SceneStream(root,'test',cfg,index=root/'eval.json')
    def test_package_source_syntax_and_framework_boundary(self):
        root=Path(__file__).resolve().parents[1]
        # Inspect owned project code, not reference virtualenvs/site-packages
        # (some third-party wheels legitimately ship Python-2 example files).
        sources=list(root.glob('*.py'))
        for folder in ('noposplat_jittor','tools','scripts','tests'):
            sources.extend((root/folder).rglob('*.py'))
        for path in sources:
            tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
            if path.parent.name=='noposplat_jittor':
                for node in ast.walk(tree):
                    if isinstance(node,ast.Import):self.assertTrue(all(x.name.split('.')[0]!='torch' for x in node.names),str(path))
                    if isinstance(node,ast.ImportFrom):self.assertNotEqual((node.module or '').split('.')[0],'torch',str(path))


if __name__=='__main__':unittest.main(verbosity=2)

