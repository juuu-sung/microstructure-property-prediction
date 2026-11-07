import copy
import json
import unittest
import numpy as np
from scripts import audit_boundary_quality_v3 as q


class QualityTests(unittest.TestCase):
    def fixture(self):
        base={'intervals':500,'usable_transect_fraction':.9,'occupied_tiles':16,'boundary_area':.05,
              'raw_spacing':32.,'scale_f1':.97,'structure_noise_ratio':1.49}
        probes=[{'name':n,'map_f1':.95,'spacing_change':.03,'raw_spacing':33.,'observation':{'available':True,'reasons':[]}} for n in q.INTERNAL]
        return base,probes,{'status':'structured','ratio':10.}

    def test_snr_threshold_cannot_flip_status_or_semantic_claim(self):
        b,p,s=self.fixture();before=q.decide(b,p,s)
        for snr in (.8,1.49,1.5,1.51,3.):
            b['structure_noise_ratio']=snr;after=q.decide(b,p,s)
            self.assertEqual(after['status'],before['status'])
            self.assertEqual(after['semantic_validity'],'unverified_mixed_boundaries')
            self.assertIsNone(after['physical_grain_size']);self.assertIsNone(after['pore_fraction'])

    def test_guard_band_around_previous_binary_cutoff(self):
        b,p,s=self.fixture()
        for value in (.79,.80,.81):
            p[0]['map_f1']=value
            self.assertEqual(q.decide(b,p,s)['status'],'borderline_proxy')
        p[0]['map_f1']=.60
        self.assertEqual(q.decide(b,p,s)['status'],'unstable_proxy')

    def test_empty_observation_not_rewarded_for_empty_map_agreement(self):
        b,p,s=self.fixture();b['raw_spacing']=None;b['intervals']=0
        result=q.decide(b,p,s)
        self.assertEqual(result['status'],'unavailable');self.assertIsNone(result['recommended_proxy_spacing'])

    def test_stable_null_like_map_cannot_be_usable_structure(self):
        b,p,s=self.fixture();s.update(status='null_like',ratio=1.1)
        result=q.decide(b,p,s)
        self.assertEqual(result['status'],'null_like');self.assertIsNone(result['observed_spacing'])

    def test_missing_probe_rejected(self):
        b,p,s=self.fixture()
        with self.assertRaises(ValueError):q.decide(b,p[:-1],s)

    def test_photometric_pixel_roundtrip_and_deterministic_noise(self):
        gray=np.arange(256,dtype=np.uint8)[None,:].repeat(256,axis=0)
        np.testing.assert_array_equal(q.perturb(gray,'noise11'),q.perturb(gray,'noise11'))
        np.testing.assert_array_equal(q.detector.v1.align(q.perturb(gray,'rot90'),'rot90'),gray)

    def test_real_constant_noise_and_shuffle_negative_controls(self):
        inputs=[np.full((256,256),128,np.uint8)]
        for seed in (51,97):
            inputs.append(np.clip(np.rint(128+np.random.default_rng(seed).normal(0,12,(256,256))),0,255).astype(np.uint8))
        grain=np.full((256,256),180,np.uint8);grain[::32,:]=40;grain[:,::32]=40
        inputs.append(np.random.default_rng(73).permutation(grain.ravel()).reshape(grain.shape))
        for gray in inputs:
            result,_=q.assess(gray)
            self.assertIn(result['status'],('unavailable','null_like'))
            self.assertIsNone(result['recommended_proxy_spacing'])

    def test_known_grid_repeatable_but_not_physical_truth(self):
        gray=np.full((256,256),180,np.uint8);gray[16::32,:]=40;gray[:,16::32]=40
        result,_=q.assess(gray)
        self.assertEqual(result['status'],'repeatable_proxy')
        self.assertAlmostEqual(result['raw_spacing'],32,delta=1)
        self.assertIsNone(result['physical_grain_size'])

    def test_transition_severity(self):
        self.assertEqual(q.transition('repeatable_proxy','borderline_proxy'),'adjacent')
        self.assertEqual(q.transition('repeatable_proxy','unstable_proxy'),'severe')
        self.assertEqual(q.transition('null_like','repeatable_proxy'),'severe')

    def test_complete_quality_record_is_json_serializable(self):
        gray=np.full((256,256),180,np.uint8);gray[16::32,:]=40;gray[:,16::32]=40
        result,_=q.assess(gray)
        restored=json.loads(json.dumps(result,allow_nan=False))
        self.assertEqual(restored['v2_metrics']['occupied_tiles'],16)
        self.assertEqual(restored['semantic_validity'],'unverified_mixed_boundaries')


if __name__=='__main__':unittest.main()
