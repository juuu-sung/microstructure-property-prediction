import json
import unittest
import numpy as np
from scripts import auto_structure_review as a


class AutoReviewTests(unittest.TestCase):
    def test_empty_masks_not_perfect_and_no_physical_truth(self):
        z=np.zeros((256,256),bool)
        self.assertIsNone(a.overlap(z,z,'iou'))
        self.assertIsNone(a.overlap(z,z,'f1'))
        m={'boundary':z,'dark':z,'compact':z,'boundary_support':z.astype(float)}
        r=a.summarize_roi(m,[z]*6,[z]*6,[64,64,128,128])
        self.assertIsNone(r['boundary_f1_min']);self.assertFalse(r['ground_truth'])
        self.assertEqual(r['semantic_validity'],'unverified');self.assertEqual(len(r['review_flags']),2)

    def test_known_mask_shift_and_inverse_rotation(self):
        b=np.zeros((256,256),bool); b[50:200,100]=True
        shifted=np.roll(b,1,axis=1)
        self.assertAlmostEqual(a.overlap(b,shifted,'f1'),1)
        self.assertEqual(a.overlap(b,shifted,'iou'),0)
        np.testing.assert_array_equal(a.quality.detector.v1.align(np.rot90(b),'rot90'),b)

    def test_quality_anchor_and_proposal_source(self):
        g=np.full((256,256),180,np.uint8);g[16::32,:]=40;g[:,16::32]=40
        q,m,bp,dp=a.assess(g);anchor,am=a.quality.assess(g)
        for key in ('status','support_min','spacing_change_max','semantic_validity'):
            self.assertEqual(q[key],anchor[key])
        np.testing.assert_array_equal(m['boundary'],am['boundary'])
        self.assertEqual(q['status'],'repeatable_proxy');self.assertIsNone(q['physical_grain_size'])
        r=a.summarize_roi(m,bp,dp,[64,64,128,128]);json.dumps(r,allow_nan=False)
        self.assertFalse(a.PROFILE['training_approved']);self.assertFalse(a.PROFILE['ground_truth'])
        p=a.panels(g,m)
        for v in p.values(): self.assertEqual(v.shape,(256,256,3));self.assertEqual(v.dtype,np.uint8)

    def test_invalid_roi_and_non_native_input_rejected(self):
        with self.assertRaises(ValueError): a.summarize_roi({},[],[],[200,200,128,128])
        with self.assertRaises(ValueError): a.assess(np.zeros((128,128),np.uint8))


if __name__=='__main__':unittest.main()
