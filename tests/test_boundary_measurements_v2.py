import unittest
import numpy as np
from scripts import audit_boundary_measurements_v2 as b


class BoundaryTests(unittest.TestCase):
    def test_constant_is_abstention_with_null_not_zero(self):
        m,maps=b.detect(np.full((256,256),128,np.uint8))
        self.assertEqual(m['status'],'abstain');self.assertIsNone(m['reported_spacing'])
        self.assertIsNone(m['raw_spacing']);self.assertFalse(maps['boundary'].any())

    def test_step_boundaries_present_without_dark_valley(self):
        image=np.full((256,256),180,np.uint8);image[:,128:]=100
        m,maps=b.detect(image)
        self.assertGreater(maps['step'][:,124:132].sum(),200)
        self.assertEqual(maps['valley'].sum(),0)

    def test_grid_spacing_and_phase(self):
        for offset in (0,16):
            image=np.full((256,256),180,np.uint8);image[:,offset::32]=40;image[offset::32,:]=40
            m,_=b.detect(image)
            self.assertAlmostEqual(m['raw_spacing'],32,delta=1)
            self.assertGreater(m['intervals'],1000)

    def test_near_flanks_merge_and_border_censoring(self):
        row=np.zeros(64,bool);row[:2]=True;row[20]=True;row[23]=True;row[42]=True;row[-1]=True
        np.testing.assert_array_equal(b.centers(row),[21.5,42])

    def test_white_noise_must_not_report_grain_size(self):
        for seed in range(3):
            image=np.clip(np.rint(128+np.random.default_rng(seed).normal(0,12,(256,256))),0,255).astype(np.uint8)
            m,_=b.detect(image)
            self.assertEqual(m['status'],'abstain');self.assertIn('low_structure_noise_ratio',m['reasons'])
            self.assertIsNone(m['reported_spacing'])

    def test_quality_is_explicit_and_keeps_diagnostic_spacing(self):
        m={'intervals':200,'usable_transect_fraction':.9,'occupied_tiles':16,'scale_f1':.9,
           'structure_noise_ratio':4.,'boundary_area':.05,'raw_spacing':23.}
        self.assertEqual(b.quality(m)['reported_spacing'],23.)
        m['intervals']=1
        q=b.quality(m);self.assertIn('low_intervals',q['reasons']);self.assertIsNone(q['reported_spacing'])
        self.assertEqual(m['raw_spacing'],23.)

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):b.detect(np.zeros((128,128),np.uint8))


if __name__=='__main__':unittest.main()
