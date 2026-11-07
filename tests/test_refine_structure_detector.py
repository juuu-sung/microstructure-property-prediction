import unittest
import numpy as np
from scipy import ndimage as ndi
from skimage import draw
from scripts import refine_structure_detector as r


class RefinementTests(unittest.TestCase):
    def test_constant_and_white_noise_do_not_gain_edges(self):
        for seed in (None,19,71):
            a=np.full((256,256),140,np.uint8) if seed is None else np.clip(np.rint(140+np.random.default_rng(seed).normal(0,15,(256,256))),0,255).astype(np.uint8)
            q,m=r.detect(a)
            self.assertEqual(q['added_pixels'],0)
            self.assertFalse(q['training_approved']);self.assertIsNone(q['semantic_truth'])

    def test_weak_step_has_new_original_pixel_support(self):
        a=np.full((256,256),150,np.uint8);a[:,128:]=154
        q,m=r.detect(a)
        self.assertGreater(m['added'][10:-10,125:131].sum(),100)
        self.assertTrue(np.all(m['raw_step_contrast'][m['added']]>=.01))
        self.assertEqual(m['added'][:5].sum(),0)

    def test_round_deep_vs_shallow_polygon_are_partitioned(self):
        raw=np.full((256,256),.7);dark=np.zeros_like(raw,bool)
        rr,cc=draw.disk((70,70),9,shape=raw.shape);dark[rr,cc]=True;raw[rr,cc]=.12
        rr,cc=draw.polygon([130,142,155,146],[130,155,148,124],shape=raw.shape);dark[rr,cc]=True;raw[rr,cc]=.58
        compact=ndi.binary_opening(dark,structure=r.morphology.disk(2))
        certain,ambiguous,objects=r.classify_objects(raw,dark,compact,.02)
        self.assertTrue(certain[70,70]);self.assertTrue(ambiguous[142,140]);self.assertFalse((certain & ambiguous).any())
        np.testing.assert_array_equal(certain|ambiguous,compact)
        self.assertEqual(len(objects),2)

    def test_opening_cannot_hide_original_elongation(self):
        raw=np.full((256,256),.7);dark=np.zeros_like(raw,bool);dark[70:90,60:135]=True;raw[dark]=.1
        compact=np.zeros_like(dark);compact[72:88,85:102]=True
        certain,ambiguous,obj=r.classify_objects(raw,dark,compact,.01)
        self.assertFalse(certain.any());self.assertIn('elongated',obj[0]['reasons'])

    def test_original_boundary_is_retained_without_claiming_grain_truth(self):
        a=np.full((256,256),180,np.uint8);a[16::32]=40;a[:,16::32]=40
        q,m=r.detect(a);_,b=r.old.detect(a)
        np.testing.assert_array_equal(m['before'],b['boundary'])
        self.assertTrue(np.all(m['mixed_boundary'][m['before']]))
        self.assertFalse(r.PROFILE['training_approved'])


if __name__=='__main__':unittest.main()
