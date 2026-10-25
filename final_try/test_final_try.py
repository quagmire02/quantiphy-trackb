"""CPU regressions for consequential parsing, routing, unit and identity contracts."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from pipeline.ground import parse_boxes
from pipeline.parse import finalize_question
from repairs import camera_distance, direct_value
from run import assemble, positive, read_values


class Contracts(unittest.TestCase):
    def test_question_string_is_one_target_and_prose_json(self):
        q = finalize_question('Example: {"x": 1} Result: {"targets":"red ball","target_quantity":"velocity"}', 'What is its velocity in m/s?')
        self.assertEqual(q['targets'], ['red ball'])
        self.assertEqual(q['target_quantity'], 'speed')
        self.assertTrue(q['question_ok'])

    def test_second_endpoint_is_explicit(self):
        reply = json.dumps({'targets': ['basketball'], 'target_quantity': 'distance', 'target_time': [0, 0]})
        q = finalize_question(reply, "What is the basketball's initial distance from the desk in cm?")
        self.assertEqual(q['targets'], ['basketball', 'desk'])
        self.assertTrue(q['question_ok'])
        q = finalize_question(reply, 'What is the distance in cm?')
        self.assertFalse(q['question_ok'])

    def test_box_formats_and_range(self):
        self.assertEqual(parse_boxes('Here: [{"bbox_2d": [1,2,30,40]}, {"bbox_2d": [50,60,70,80]}'), [[1.,2.,30.,40.], [50.,60.,70.,80.]])
        self.assertEqual(parse_boxes('{"x1":1,"y1":2,"x2":30,"y2":40}'), [[1.,2.,30.,40.]])
        self.assertEqual(parse_boxes('[0, 0, 1200, 500]'), [])

    def test_answer_units_stay_attached_to_number(self):
        self.assertAlmostEqual(direct_value('Final Answer: 1.5 m', 'How long in cm?'), 150)
        self.assertAlmostEqual(direct_value('3 m/s^2', 'Acceleration in cm/s^2?'), 300)
        self.assertAlmostEqual(direct_value('3 cm/s²', 'Acceleration in m/s²?'), .03)
        self.assertTrue(np.isnan(direct_value('3 m/s^2', 'Speed in m/s?')))
        self.assertTrue(np.isnan(direct_value('0.0 m/s', 'Speed in m/s?')))
        self.assertTrue(np.isnan(direct_value('The length is 2 m or 3 m.', 'Length in m?')))

    def test_camera_range_uses_annotation_and_no_extrapolation(self):
        row = SimpleNamespace(question='What is the distance from the camera at 1s in cm?', video_type='S3MC',
                              depth_info='t=0s, distance_ball_camera = 2m\nt=2s, distance_ball_camera = 4m')
        rec = {'targets': ['ball', 'camera'], 'target_time': [1, 1]}
        self.assertEqual(camera_distance(row, rec), 300.)
        rec['target_time'] = [3, 3]
        self.assertIsNone(camera_distance(row, rec))

    def test_routing_aligns_by_id_and_never_takes_invalid(self):
        df = pd.DataFrame({'id': [2, 0, 1]})
        first = pd.Series({0: np.inf, 1: 0, 2: 7.})
        second = pd.Series({0: 3., 1: 4., 2: 99.})
        value, source = assemble(df, [('primary', first), ('rescue', second)])
        self.assertEqual(list(value), [7., 3., 4.])
        self.assertEqual(list(source), ['primary', 'rescue', 'rescue'])
        self.assertTrue(positive(value).all())

    def test_saved_replay_preserves_all_measured_answers(self):
        here = Path(__file__).resolve().parent
        original = pd.read_csv(here / 'inputs/v17_sources.csv').set_index('id')
        output = here / 'outputs/replay_test/candidate_preserved_sources.csv'
        if not output.exists():
            self.skipTest('Run offline --output final_try/outputs/replay_test first')
        current = pd.read_csv(output).set_index('id')
        measured = original.source.str.contains('-measure', regex=False)
        np.testing.assert_array_equal(original.loc[measured, 'parsed_value'], current.loc[measured, 'parsed_value'])
        np.testing.assert_allclose(original.parsed_value, current.parsed_value, rtol=1e-12, atol=0)
        self.assertEqual(len(current), 3289)
        self.assertTrue(positive(current.parsed_value).all())

    def test_orchestrator_adapter_scope_and_selective_rescue(self):
        # Execute the entire orchestrator with tiny deterministic worker outputs.
        # No weights or validation answers enter the mock predictions.
        from pipeline.data import load_split
        from run import main, HERE
        try:
            df = load_split('validation')
        except FileNotFoundError:
            self.skipTest('Validation metadata/videos not downloaded yet')
        calls = []
        (HERE / 'outputs').mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix='contract_', dir=HERE / 'outputs'))
        def fake_worker(cmd, env, **kwargs):
            kind = cmd[cmd.index('--phase') + 1]
            tree = Path(env['QUANTIPHY_RESULTS'])
            cache = tree / 'cache/validation'
            cache.mkdir(parents=True, exist_ok=True)
            selected = json.loads(Path(cmd[cmd.index('--ids') + 1]).read_text()) if '--ids' in cmd else list(df.id)
            calls.append((tree.name, kind, selected, env.get('QUANTIPHY_VLM_ADAPTER')))
            if kind == 'measure':
                # Every third row fails primary measurement; secondary fails too.
                pd.DataFrame({'id': df.id, 'value': np.where(df.index % 3 == 0, np.nan, 2.), 'reason': ''}).to_csv(cache / 'measure_solutions.csv', index=False)
            else:
                with (cache / 'direct.jsonl').open('w') as fh:
                    for row in df[df.id.isin(selected)].itertuples():
                        fh.write(json.dumps({'id': row.id, 'video_id': row.video_id, 'question': row.question,
                                             'reply': '0' if tree.name == 'primary' else '3'}) + '\n')
        try:
            with patch.object(sys, 'argv', ['run.py', '--split', 'validation', '--primary-model', 'mock/32B',
                              '--secondary-model', 'mock/4B', '--secondary-adapter', 'mock_adapter', '--output', str(out)]), patch('run.subprocess.run', side_effect=fake_worker):
                main()
            by = {(name, kind): (selected, adapter) for name, kind, selected, adapter in calls}
            self.assertLess(len(by['secondary', 'measure'][0]), len(df))
            self.assertIsNone(by['secondary', 'measure'][1])
            self.assertEqual(by['secondary_direct', 'direct'][1], 'mock_adapter')
            result = pd.read_csv(out / 'candidate_fresh_sources.csv')
            self.assertTrue(positive(result.parsed_value).all())
            self.assertEqual(set(result.source), {'32B-measure', '4B-direct'})
        finally:
            shutil.rmtree(out)


if __name__ == '__main__':
    unittest.main()
