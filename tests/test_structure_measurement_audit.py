"""Known geometry tests; fixtures are never used to train competition models."""
import unittest
import numpy as np
import pandas as pd
from scripts import audit_structure_measurements as a
from scripts.review_structure_measurements import all_offset_intercepts


class MeasurementTests(unittest.TestCase):
    def test_censored_runs_not_double_counted(self):
        line=np.zeros(64,bool);line[:3]=True;line[10:13]=True;line[25:27]=True;line[35:50]=True;line[62:]=True
        np.testing.assert_array_equal(a.runs_centers(line),[11,25.5])

    def test_grid_line_spacing_is_32(self):
        mask=np.zeros((256,256),bool);mask[:,16::32]=True;mask[16::32,:]=True
        stats=a.intercepts(mask)
        self.assertEqual(stats['spacing'],32.)
        self.assertEqual(stats['h_crossings'],48)
        self.assertEqual(stats['v_crossings'],48)

    def test_blank_is_missing_spacing_not_zero_grain_size(self):
        x,m=a.maps_and_metrics(np.full((256,256),128,np.uint8))
        self.assertIsNone(x['line_spacing']);self.assertEqual(x['compact_count'],0)
        self.assertEqual(x['dark_area'],0)
        self.assertEqual(x['ws1_regions'],1)

    def test_compact_disc_not_long_dark_stripe(self):
        y,x=np.mgrid[:256,:256];img=np.full((256,256),.8)
        disc=(x-80)**2+(y-80)**2<=6**2;img[disc]=.05;img[150:156,30:220]=.05
        mask,_,count=a.compact_mask(img)
        self.assertEqual(count,1);self.assertTrue(mask[80,80]);self.assertFalse(mask[152,100])

    def test_selection_has_no_label_dependency_and_frozen_split_sizes(self):
        rng=np.random.default_rng(72)
        df=pd.DataFrame(rng.random((100,6)),columns=a.COLS[1:]);df['ID']=[f'TRAIN_{i:06d}' for i in range(100)]
        df['hardness']=rng.normal(size=100)
        first=a.select(df,{'TRAIN_000007'})
        df['hardness']='never read'
        self.assertEqual(first,a.select(df,{'TRAIN_000007'}));self.assertEqual(len(first),25)
        self.assertEqual(sum(r['subset']=='check' for r in first),10)
        self.assertNotIn('TRAIN_000007',[r['ID'] for r in first])
        self.assertTrue(all('hardness' not in r for r in first))

    def test_rotation_alignment_and_tolerance(self):
        m=np.zeros((256,256),bool);m[35:100,52]=True
        np.testing.assert_array_equal(a.align(a.perturb(m,'rot90','fixture'),'rot90'),m)
        self.assertEqual(a.tolerant_f1(m,np.roll(m,1,axis=1)),1.)
        self.assertEqual(a.tolerant_f1(m,np.roll(m,8,axis=1)),0.)

    def test_all_offset_sampling_resolves_grid_phase_alias(self):
        for offset in (0,16):
            gray=np.full((256,256),180,np.uint8)
            gray[offset::32,:]=40;gray[:,offset::32]=40
            _,maps=a.maps_and_metrics(gray)
            self.assertEqual(all_offset_intercepts(maps['line'])['spacing'],32.)
        # The original fixed transects remain frozen, including their known limitation.
        gray=np.full((256,256),180,np.uint8);gray[::32,:]=40;gray[:,::32]=40
        measured,_=a.maps_and_metrics(gray)
        self.assertIsNone(measured['line_spacing'])


if __name__=='__main__':unittest.main()
