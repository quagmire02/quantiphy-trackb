"""Numerical geometry and complete submission contracts; no weights required."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from endpoint_geometry import (agree, consensus, crop_box, decode_points, estimate,
                               frame_indices, native_points, points_agree)
from run import HERE, ROOT
from run_endpoint_rescue import aligned_base, final_csv, measure_one


class EndpointContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (HERE / 'outputs').mkdir(exist_ok=True)

    def test_full_crop_coordinate_invariance_and_si_units(self):
        points = {'reference': [[100., 100.], [300., 100.]], 'target': [[100., 200.], [200., 200.]]}
        self.assertEqual(estimate(points, 2., 'cm'), 100.)
        roi = crop_box(points, 1000, 500)
        origin = np.array(roi[:2])
        size = np.array(roi[2:])-origin
        norm = {key: (np.asarray(value)-origin)/size*1000 for key,value in points.items()}
        back = {key: native_points(value, roi) for key,value in norm.items()}
        self.assertAlmostEqual(estimate(back, 2., 'cm'), 100.)
        self.assertAlmostEqual(estimate(back, None, 'cm', m_per_native_px=.01), 100.)

    def test_json_cannot_turn_invented_metric_number_into_geometry(self):
        self.assertIsNone(decode_points('{"answer":42}'))
        self.assertIsNone(decode_points('{"target":[[0,0],[1001,0]],"reference":[[1,1],[2,2]]}'))
        self.assertIsNone(decode_points('{"target":[[0,0],[0,0]],"reference":[[1,1],[2,2]]}'))
        self.assertIsNone(decode_points('{"target":[[0,0],[100,0]]}'))
        self.assertIsNotNone(decode_points('{"target":[[0,0],[100,0]]}', False))

    def test_consensus_does_not_cherry_pick_frames(self):
        self.assertTrue(agree(10., 11.))
        self.assertFalse(agree(10., 30.))
        self.assertIsNone(consensus([10.]))
        self.assertIsNone(consensus([10., 10., 40.]))
        self.assertEqual(consensus([9., 10., 11.]), 10.)

    def test_matching_ratios_do_not_override_wrong_spatial_identity(self):
        a = {'target': np.array([[0.,0.],[100.,0.]])}
        b = {'target': np.array([[500.,0.],[600.,0.]])}
        self.assertFalse(points_agree(a, b))
        self.assertTrue(points_agree(a, {'target': a['target'][::-1]}))

    def test_requested_time_sampling_is_not_uniform_video_guessing(self):
        self.assertEqual(frame_indices(101, 25, [1.,1.]), [24,25,26])
        self.assertEqual(frame_indices(101, 25, [-1.,-1.]), [99,100])
        self.assertEqual(frame_indices(101, 25, [0.,1.]), [])
        self.assertEqual(frame_indices(101, 25), [20,50,80])

    def test_full_template_and_unselected_rows_are_preserved(self):
        df = pd.read_parquet(ROOT / 'data/test/test_dataset.parquet').reset_index(drop=True)
        df['id'] = df.index
        df['category'] = df.inference_type.str[0]+df.video_type.str[1]
        base = aligned_base(df, HERE / 'inputs/v17_sources.csv')
        item = int(df.loc[df.category.eq('S2'), 'id'].iloc[0])
        (HERE / 'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='endpoint_export_', dir=HERE/'outputs') as tmp:
            audit = final_csv(df, base, {item: 12.345}, Path(tmp))
            upload = pd.read_csv(Path(tmp)/'candidate_endpoints.csv')
            self.assertEqual(len(upload), 3289)
            np.testing.assert_array_equal(upload.id, df.id+1)
            self.assertAlmostEqual(upload.parsed_value.iloc[item], 12.345)
            others = df.id != item
            np.testing.assert_allclose(upload.parsed_value[others], base.parsed_value.values[others], rtol=1e-12)
            three = int(df.loc[df.category.str.endswith('3'), 'id'].iloc[0])
            with self.assertRaisesRegex(ValueError, '3D'):
                final_csv(df, base, {three: 1.}, Path(tmp))

    def test_actual_crop_evidence_path_with_simulated_vlm_and_frames(self):
        row = SimpleNamespace(id=99991, video_id='synthetic', video_path='unused', fps=10.,
                              question='What is the width of the blue box in cm?', prior='red box width = 2m')
        prior = {'id': 'synthetic', 'ref_object': 'red box', 'ref_quantity': 'width', 'prior_value': 2., 'prior_unit': 'm'}
        question = {'id': row.id, 'targets': ['blue box'], 'target_quantity': 'width', 'target_unit': 'cm', 'target_time': None}
        frame = np.zeros((200,400,3), dtype=np.uint8)
        # Reference span 80 native px, target span 40 native px -> target 1m.
        initial = {'reference': [[100,250],[300,250]], 'target': [[500,500],[600,500]]}
        native = {key: native_points(value, [0,0,400,200]) for key,value in initial.items()}
        roi = crop_box(native, 400, 200)
        size = np.array(roi[2:])-roi[:2]
        zoomed = {key: ((value-np.array(roi[:2]))/size*1000).tolist() for key,value in native.items()}
        class Model:
            def __init__(self): self.calls=0
            def chat(self, *args, **kwargs):
                self.calls += 1
                return json.dumps(initial if self.calls % 2 else zoomed)
        class Capture:
            def get(self, key): return 11
            def release(self): pass
        with tempfile.TemporaryDirectory(prefix='endpoint_measure_', dir=HERE/'outputs') as tmp:
            args = SimpleNamespace(base_dir=Path(tmp), split='test', longest=1024)
            cache = Path(tmp)/'primary/cache/test/items'/str(row.id)
            cache.mkdir(parents=True)
            (cache/'parse_prior.jsonl').write_text(json.dumps(prior)+'\n')
            (cache/'parse_question.jsonl').write_text(json.dumps(question)+'\n')
            model = Model()
            with patch('cv2.VideoCapture', return_value=Capture()), patch('pipeline.refine.read_selected', return_value={2:frame,5:frame,8:frame}):
                result = measure_one(model, row, args)
            self.assertAlmostEqual(result['value'], 100.)
            self.assertEqual(result['accepted_frames'], 3)
            self.assertEqual(model.calls, 6)


if __name__ == '__main__':
    unittest.main()
