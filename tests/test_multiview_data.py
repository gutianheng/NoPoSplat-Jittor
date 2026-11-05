import unittest
import numpy as np
from noposplat_jittor.data import sample_indices,evaluation_context_indices


class MultiViewData(unittest.TestCase):
    def test_extra_training_contexts_are_distinct_and_interior(self):
        cfg={'gap_warmup':9375,'initial_gap_min':25,'initial_gap_max':25,'gap_min':45,'gap_max':90,'target_views':1}
        for count in range(2,7):
            ci,ti=sample_indices(100,0,{**cfg,'context_views':count},np.random.default_rng(count))
            self.assertEqual(len(set(ci)),count);self.assertEqual(ci[-1]-ci[0],25)
            self.assertTrue(np.all((ci[1:-1]>ci[0])&(ci[1:-1]<ci[-1])));self.assertTrue(ci[0]<=ti[0]<=ci[-1])

    def test_two_view_default_rng_unchanged(self):
        cfg={'gap_warmup':9375,'initial_gap_min':25,'initial_gap_max':25,'gap_min':45,'gap_max':90,'target_views':1}
        rng=np.random.default_rng(51);a,b=sample_indices(100,0,cfg,rng)
        reference=np.random.default_rng(51);gap=int(reference.integers(25,26));left=int(reference.integers(0,100-gap));target=reference.integers(left,left+gap+1,size=1)
        np.testing.assert_array_equal(a,[left,left+gap]);np.testing.assert_array_equal(b,target);self.assertEqual(rng.random(),reference.random())

    def test_evaluation_adds_views_without_changing_endpoints(self):
        np.testing.assert_array_equal(evaluation_context_indices([5,34],4),[5,14,24,34])
        np.testing.assert_array_equal(evaluation_context_indices([5,12,34],3),[5,12,34])


if __name__=='__main__':unittest.main()
