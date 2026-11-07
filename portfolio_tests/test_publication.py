import json
from pathlib import Path
import tempfile
import unittest
from tools.blend_predictions import blend, read_predictions, save_new
from tools.verify_evidence import metric_check, verify


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root=Path(__file__).resolve().parents[1]/'.task-tmp'
        cls.temp_root.mkdir(exist_ok=True)

    def test_published_evidence_consistency(self):
        self.assertEqual(verify()['status'],'passed')

    def test_wrong_aggregate_rmse_rejected(self):
        with self.assertRaises(ValueError):
            metric_check({'n':2,'sse':8.,'rmse':3.})
        metric_check({'n':2,'sse':8.,'rmse':2.})

    def test_alignment_and_finite_values(self):
        a=(['A','B'],[10.,20.])
        with self.assertRaises(ValueError):blend(a,(['B','A'],[20.,10.]),a)
        with self.assertRaises(ValueError):blend(a,a,(['A','B'],[float('nan'),20.]))

    def test_fixed_composition_and_round_trip_preserves_old_output(self):
        ids=['SYNTH_A','SYNTH_B']
        result=blend((ids,[10.,20.]),(ids,[30.,40.]),(ids,[50.,60.]))
        self.assertEqual(result,(ids,[25.,35.]))
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            p=Path(d)/'example.csv';save_new(p,result)
            self.assertEqual(read_predictions(p),result)
            with self.assertRaises(FileExistsError):save_new(p,(ids,[1.,2.]))
            self.assertEqual(read_predictions(p),result)

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            p=Path(d)/'bad.csv';p.write_text('ID,hardness\nA,1\nA,2\n')
            with self.assertRaises(ValueError):read_predictions(p)


if __name__=='__main__':unittest.main()
