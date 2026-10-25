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
from run import assemble, positive, read_values, measurement_weak, budget_ids


class Contracts(unittest.TestCase):
    def test_budget_selects_only_fallbacks_with_balanced_categories(self):
        df = pd.DataFrame({'id': [0,1,2,3,4,5,6,7], 'category': ['S2','S2','D2','D2','S3','S3','D3','D3']})
        historical = pd.DataFrame({'source': ['0:32B-measure','vlm','vlm','median','vlm','vlm','vlm','vlm']})
        self.assertEqual(budget_ids(df, historical, 4), [1,2,4,6])
        self.assertEqual(budget_ids(df, historical, 100), [1,2,4,6,3,5,7])

    def test_deadline_run_produces_full_csv_and_preserves_unprocessed_rows(self):
        from run import main, HERE
        from pipeline.data import load_split
        try:
            df = load_split('test')
        except FileNotFoundError:
            self.skipTest('Test metadata/videos not downloaded yet')
        old = pd.read_csv(HERE/'inputs/v17_sources.csv').set_index('id')
        selected = budget_ids(df, old, 4)
        def fake_worker(cmd, env, **kwargs):
            tree = Path(env['QUANTIPHY_RESULTS']) / 'cache/test'
            tree.mkdir(parents=True, exist_ok=True)
            subset = json.loads(Path(cmd[cmd.index('--ids')+1]).read_text())
            if cmd[cmd.index('--phase')+1] == 'measure':
                pd.DataFrame({'id': subset, 'value': [2.]*len(subset), 'reason': ['']*len(subset)}).to_csv(tree/'measure_solutions.csv', index=False)
        with tempfile.TemporaryDirectory(prefix='budget_contract_', dir=HERE/'outputs') as tmp:
            with patch.object(sys, 'argv', ['run.py','--primary-model','mock/32B','--secondary-model','mock/4B',
                                           '--max-new-items','4','--output',tmp]), patch('run.subprocess.run', side_effect=fake_worker):
                main()
            result = pd.read_csv(Path(tmp)/'candidate_guided_sources.csv').set_index('id')
            untouched = ~df.id.isin(selected).values
            np.testing.assert_allclose(result.parsed_value.values[untouched], old.parsed_value.values[untouched], rtol=1e-12)
            self.assertTrue((result.loc[selected,'parsed_value'] == 2.).all())
            upload = pd.read_csv(Path(tmp)/'candidate_guided.csv')
            self.assertEqual(len(upload),3289)
            np.testing.assert_array_equal(upload.id.values,df.id.values+1)

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
            result = pd.read_csv(out / 'candidate_32b_led_sources.csv')
            self.assertTrue(positive(result.parsed_value).all())
            self.assertEqual(set(result.source), {'32B-measure', '4B-direct'})
            guided = pd.read_csv(out / 'candidate_guided_sources.csv')
            self.assertEqual(set(guided.source), {'32B-measure', '32B-reviewed-with-4B-hypothesis'})
            # No saved-answer fallback or pooled test prediction median can enter live output.
            self.assertFalse(guided.source.str.contains('median|v17', regex=True).any())
            hints = json.loads((out / 'primary_review/same_question_hypotheses.json').read_text())
            self.assertTrue(all(x['32B'] is None and x['4B'] == 3 for x in hints.values()))
            self.assertEqual(set(map(int, hints)), set(by['primary_review', 'direct'][0]))
        finally:
            shutil.rmtree(out)

    def test_question_scopes_do_not_share_video_semantics(self):
        from guided import scopes, context, ContextVLM, ground_one
        frame = pd.DataFrame({'id': [1, 2], 'video_id': ['same', 'same'],
                              'question': ['width of red ball?', 'SECRET OTHER QUESTION'],
                              'prior': ['red ball height=2m', 'SECRET OTHER PRIOR'], 'depth_info': ['', '']})
        items = list(scopes(frame, Path('/tmp/question_scopes')))
        self.assertEqual([len(x[0]) for x in items], [1, 1])
        self.assertNotEqual(items[0][1], items[1][1])
        row = next(items[0][0].itertuples())
        before = context(row)
        frame.loc[1, 'prior'] = 'CHANGED OTHER PRIOR'
        self.assertEqual(before, context(next(list(scopes(frame, Path('/tmp/question_scopes')))[0][0].itertuples())))
        self.assertNotIn('SECRET', before)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            ground_one(None, frame, {}, Path('/tmp/not_written'), None)

    def test_joint_grounding_requires_exact_ids_and_valid_boxes(self):
        from guided import decode_objects
        reply = '[{"object_id":"o0","boxes":[[1,2,30,40]]},{"object_id":"unrequested","boxes":[[0,0,100,100]]}]'
        self.assertEqual(decode_objects(reply, {'o0': 'red ball'}), {'o0': [[1.,2.,30.,40.]]})
        self.assertEqual(decode_objects('[{"object_id":"o0","boxes":[[0,0,2000,10]]}]', {'o0': 'ball'}), {'o0': []})

    def test_unresolved_distinct_object_conflict_cannot_share_a_box(self):
        import pipeline.ground as grounding
        objects = {'red ball': {'phrase': 'red ball', 'frame': 0, 'boxes': [[0, 0, 50, 50]]},
                   'blue ball': {'phrase': 'blue ball', 'frame': 0, 'boxes': [[0, 0, 50, 50]]}}
        with patch.object(grounding, 'ground', return_value={'boxes': [[0, 0, 50, 50]]}):
            result = grounding.resolve_conflicts(None, np.zeros((4, 60, 60, 3), dtype=np.uint8), objects)
        self.assertTrue(all(not x['boxes'] and not x['conflict']['resolved'] for x in result.values()))

    def test_helper_triggers_include_suspicious_successes(self):
        frame = pd.DataFrame({'value': [2., 2., 2., 2., np.nan],
                              'min_mask_short_px': [30., 4., 30., 30., np.nan],
                              'max_border_fraction': [0., 0., .8, 0., np.nan],
                              'moge_used': [False, False, False, True, False]})
        self.assertEqual(list(measurement_weak(frame)), [False, True, True, True, True])

    def test_joint_grounding_uses_helper_only_for_same_question_and_verifies(self):
        from guided import ground_one
        class FakeVLM:
            def __init__(self):
                self.messages = []
            def chat(self, messages, **kwargs):
                self.messages.append(messages)
                return '[{"object_id":"o0","boxes":[[100,100,200,200]]},{"object_id":"o1","boxes":[[600,600,800,800]]}]'
        row = pd.DataFrame({'id': [1], 'video_id': ['v'], 'question': ['width of blue ball in cm?'],
                            'prior': ['red ball diameter=2cm'], 'depth_info': [''], 'video_path': ['unused'], 'fps': [24.]})
        from run import HERE
        (HERE / 'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='ground_contract_', dir=HERE / 'outputs') as tmp:
            root = Path(tmp)
            proposal = root / 'helper/items/1/ground/v.json'
            proposal.parent.mkdir(parents=True)
            proposal.write_text(json.dumps({'question_id': 1, 'shape': [100, 100], 'objects': {'red ball': {'phrase': 'red ball', 'frame': 0, 'boxes': [[10, 10, 20, 20]]}}}))
            engine = SimpleNamespace(TRACK_LONGEST=1024, GROUND_SWEEP_FRAMES=2,
                                     read_frames=lambda *args: np.zeros((2, 100, 100, 3), dtype=np.uint8),
                                     key_frames=lambda *args: {'red ball': (0, 'red ball'), 'blue ball': (0, 'blue ball')},
                                     object_keys=lambda rec: ('red ball', ['blue ball']))
            model = FakeVLM()
            ground_one(engine, row, {1: {'ok': True}}, root / 'item', model, root / 'helper')
            self.assertEqual(len(model.messages), 2)
            self.assertIn('Unverified 4B', model.messages[0][0]['content'][1]['text'])
            self.assertIn('Check these candidate boxes', model.messages[1][0]['content'][1]['text'])
            result = json.loads((root / 'item/ground/v.json').read_text())
            self.assertEqual(result['question_id'], 1)
            self.assertEqual(result['objects']['blue ball']['boxes'], [[60., 60., 80., 80.]])
            proposal.write_text(json.dumps({'question_id': 999, 'shape': [100, 100], 'objects': {}}))
            with self.assertRaisesRegex(ValueError, 'another question'):
                ground_one(engine, row, {1: {'ok': True}}, root / 'newitem', model, root / 'helper')

    def test_real_worker_never_passes_two_questions_to_a_measurement_stage(self):
        import worker
        import engine
        from run import HERE
        frame = pd.DataFrame({'id': [11, 12], 'video_id': ['shared_video', 'shared_video'],
                              'question': ['length in m?', 'speed in m/s?'],
                              'prior': ['ball length=1m', 'car speed=2m/s'],
                              'depth_info': ['', ''], 'video_type': ['S2MC', 'V2MC'], 'video_path': ['unused', 'unused']})
        calls = []
        def observe(one, *args, **kwargs):
            self.assertEqual(len(one), 1)
            calls.append(int(one.id.iloc[0]))
        def solve(one, *args):
            observe(one)
            return pd.DataFrame({'id': one.id, 'value': [2.], 'reason': ['']})
        with tempfile.TemporaryDirectory(prefix='worker_contract_', dir=HERE / 'outputs') as tmp:
            with patch.object(sys, 'argv', ['worker.py', '--phase', 'measure', '--split', 'test', '--stages', 'parse,ground,solve']), \
                 patch('pipeline.data.load_split', return_value=frame), patch('pipeline.scoring.RESULTS', Path(tmp)), \
                 patch('pipeline.vlm.QwenVL'), patch.object(engine, 'stage_parse', side_effect=observe), \
                 patch('guided.ground_one', side_effect=lambda eng, one, *a: observe(one)), \
                 patch.object(engine, 'load_parsed', return_value={}), patch.object(engine, 'stage_solve', side_effect=solve), \
                 patch.object(engine, 'load_jsonl', return_value={}), patch.object(engine, 'moge_corrections'), \
                 patch.object(engine, 'MOGE_CORR_PATH'):
                worker.main()
            self.assertEqual(calls.count(11), 3)
            self.assertEqual(calls.count(12), 3)
            result = pd.read_csv(Path(tmp) / 'cache/test/measure_solutions.csv')
            self.assertEqual(list(result.id), [11, 12])
            self.assertTrue((Path(tmp) / 'cache/test/items/11').exists())
            self.assertTrue((Path(tmp) / 'cache/test/items/12').exists())


if __name__ == '__main__':
    unittest.main()
