"""Run after run.py: selective endpoint/crop evidence, then a complete upload CSV."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from endpoint_geometry import (SIZE, agree, consensus, crop_box, decode_points, estimate,
                               frame_indices, native_points, points_agree, request_text, span)
from pipeline.parse import finalize_prior, finalize_question, prior_messages, question_messages
from pipeline.units import question_unit, to_si, unit_factor
from run import HERE, ROOT, INPUTS, measurement_weak, positive


def aligned_base(df, path):
    base = pd.read_csv(path)
    if base.id.duplicated().any() or set(base.id) != set(df.id):
        raise ValueError('Base sources must have every internal zero-based ID exactly once')
    base = base.set_index('id').reindex(df.id)
    if not np.array_equal(base.video_id.str.strip(), df.video_id.str.strip()):
        raise ValueError('Base source videos do not match dataset')
    if not np.array_equal(base.question.str.strip(), df.question.str.strip()):
        raise ValueError('Base source questions do not match dataset')
    if not positive(base.parsed_value).all():
        raise ValueError('Base has unusable predictions')
    return base


def final_csv(df, base, updates, output, split='test'):
    values = base.parsed_value.copy()
    sources = base.source.copy()
    for item, value in updates.items():
        if item not in values.index or not np.isfinite(value) or value <= 0:
            raise ValueError('Invalid endpoint update')
        if not str(df.set_index('id').loc[item, 'category']).endswith('2'):
            raise ValueError('Endpoint experiment must not replace a 3D answer')
        values.loc[item], sources.loc[item] = value, '2D-endpoints-crop-consensus'
    audit = df[['id', 'video_id', 'question', 'category']].copy()
    audit['parsed_value'], audit['source'] = values.values, sources.values
    audit['base_value'] = base.parsed_value.values
    audit['changed'] = ~np.isclose(values.values, base.parsed_value.values)
    audit.to_csv(output / 'candidate_endpoints_sources.csv', index=False)
    if split == 'test':
        path = ROOT / 'data/template/quantiphy_submission_template.csv'
        template = pd.read_csv(path if path.exists() else INPUTS / 'template.csv')
        if len(template) != len(df) or not np.array_equal(template.id, df.id+1) or not np.array_equal(template.video_id.str.strip(), df.video_id.str.strip()):
            raise ValueError('Submission template mismatch')
        template['parsed_value'] = values.values
        template.to_csv(output / 'candidate_endpoints.csv', index=False)
    else:
        from pipeline.scoring import quick_report
        quick_report(df, values.values, 'candidate_endpoints')
    return audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, help='32B model folder/ID; no trained adapter is loaded')
    ap.add_argument('--base-dir', type=Path, default=HERE / 'outputs/guided1')
    ap.add_argument('--base-sources', type=Path, help='Defaults to <base-dir>/candidate_guided_sources.csv')
    ap.add_argument('--output', type=Path, default=HERE / 'outputs/endpoints1')
    ap.add_argument('--split', choices=['test', 'validation'], default='test')
    ap.add_argument('--four-bit', choices=['0', '1'], default='1')
    ap.add_argument('--longest', type=int, default=1024)
    ap.add_argument('--max-items', type=int, default=0, help='Smoke-test cap; unchanged items retain base predictions')
    ap.add_argument('--replay', action='store_true', help='Assemble from endpoint logs; no model or videos needed')
    ap.add_argument('--plan-only', action='store_true', help='Print eligible item counts without loading weights')
    args = ap.parse_args()
    args.output, args.base_dir = args.output.resolve(), args.base_dir.resolve()
    if not args.output.is_relative_to(HERE / 'outputs'):
        ap.error('--output must stay inside final_try/outputs')
    if args.output == args.base_dir:
        ap.error('Use a separate output folder')
    if args.longest < 256 or args.max_items < 0:
        ap.error('Use longest >=256 and max-items >=0')
    from pipeline.data import load_split
    if args.replay and args.split == 'test':
        df = pd.read_parquet(ROOT / 'data/test/test_dataset.parquet').reset_index(drop=True)
        df['id'] = df.index
        df['category'] = df.inference_type.str[0] + df.video_type.str[1]
    else:
        df = load_split(args.split)
    df['video_id'] = df.video_id.str.strip()
    base_path = args.base_sources or args.base_dir / 'candidate_guided_sources.csv'
    if not base_path.exists():
        ap.error(f'Run final_try/run.py first; missing {base_path}')
    base = aligned_base(df, base_path)
    config = dict(model=args.model, base_dir=str(args.base_dir), split=args.split, longest=args.longest,
                  four_bit=args.four_bit, max_items=args.max_items,
                  base_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes() + (HERE/'endpoint_geometry.py').read_bytes()).hexdigest(),
                  dataset_sha256=hashlib.sha256(df[[c for c in ['id','video_id','question','fps','prior','depth_info'] if c in df]].fillna('').to_json().encode()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Endpoint configuration changed; choose another --output')
    manifest.write_text(json.dumps(config, indent=2))
    log = args.output / 'endpoint_evidence.jsonl'
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            item = json.loads(line)
            done[item['id']] = item
    if not args.replay:
        weak = base.source.str.contains('direct|prior-estimate|fallback', case=False, regex=True)
        primary_path = args.base_dir / 'primary/cache' / args.split / 'measure_solutions.csv'
        if primary_path.exists():
            table = pd.read_csv(primary_path).set_index('id').reindex(df.id)
            weak |= measurement_weak(table)
        def length_question(question):
            unit = question_unit(question)
            return bool(unit and unit_factor(unit)[1] == 0)
        eligible = df.category.str.endswith('2').values & weak.values & df.question.map(length_question).values
        if args.plan_only:
            print(f'Eligible 2D weak/fallback questions: {int(eligible.sum())}; no model loaded.')
            print(df.loc[eligible].groupby('category').size().to_string())
            return
        todo = df.loc[eligible & ~df.id.isin(done)]
        if args.max_items:
            remaining = max(0, args.max_items-len(done))
            todo = todo.head(remaining)
        if len(todo):
            os.environ['QUANTIPHY_VLM_4BIT'] = args.four_bit
            os.environ.pop('QUANTIPHY_VLM_ADAPTER', None)
            from pipeline.vlm import QwenVL
            model = QwenVL(args.model)
            with log.open('a', encoding='utf-8') as fh:
                for row in todo.itertuples():
                    try:
                        record = measure_one(model, row, args)
                    except Exception as exc:
                        record = {'id': int(row.id), 'value': None, 'reason': f'{type(exc).__name__}: {exc}'}
                        print(f'Endpoint error {row.id}: {record["reason"]}', flush=True)
                        # A GPU failure must stop rather than silently skipping the entire experiment.
                        if 'out of memory' in str(exc).lower():
                            raise
                    record['video_id'], record['question'] = row.video_id, row.question
                    fh.write(json.dumps(record, allow_nan=False)+'\n')
                    fh.flush()
                    done[row.id] = record
                    print(f'Endpoint {row.id}: {record.get("value")} / {record.get("reason", "accepted")}', flush=True)
    questions = df.set_index('id')
    updates = {}
    for item, rec in done.items():
        if item not in questions.index or rec.get('video_id') != questions.loc[item, 'video_id'] or rec.get('question') != questions.loc[item, 'question']:
            raise ValueError('Endpoint log does not match dataset identity')
        if rec.get('value') is not None:
            updates[item] = float(rec['value'])
    audit = final_csv(df, base, updates, args.output, args.split)
    print(f'Wrote {args.output}/candidate_endpoints.csv; {len(updates)} endpoint updates; {audit.changed.sum()} numerical changes.', flush=True)


def measure_one(model, row, args):
    import cv2
    from PIL import Image
    from pipeline.refine import read_selected
    from pipeline.video import resize_longest
    from pipeline.solver import meters_per_pixel
    from pipeline.parse import SIZE_Q
    cache = args.base_dir / 'primary/cache' / args.split / 'items' / str(int(row.id))
    def cached(name, identity):
        path = cache / name
        if path.exists():
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                if rec['id'] == identity:
                    return rec
        return None
    prior = cached('parse_prior.jsonl', row.video_id)
    question = cached('parse_question.jsonl', int(row.id))
    if not prior:
        prior = finalize_prior(model.chat(prior_messages(row.prior, [row.question]), max_new_tokens=192), row.prior)
    if not question:
        question = finalize_question(model.chat(question_messages(row.question), max_new_tokens=192), row.question)
    rec = {**prior, **question}
    fail = lambda reason, **extra: dict(id=int(row.id), value=None, reason=reason, **extra)
    if rec.get('target_quantity') not in SIZE or not rec.get('target_unit') or unit_factor(rec['target_unit'])[1] != 0:
        return fail('not a supported 2D size/gap question')
    if re_search_ruler(row.prior):
        rec['ref_quantity'] = 'ruler tick spacing'
        rec['ref_object'] = rec.get('ref_object') or 'ruler'
    if not rec.get('ref_object') or rec.get('prior_value') is None or rec['prior_value'] <= 0:
        return fail('reference not usable')
    motion_scale = rec.get('ref_quantity') in ('speed', 'acceleration')
    if not motion_scale and rec.get('ref_quantity') not in set(SIZE_Q) | {'ruler tick spacing'}:
        return fail('unsupported reference quantity')
    cap = cv2.VideoCapture(row.video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    fps = float(row.fps)
    indices = frame_indices(total, fps, rec.get('target_time'))
    if len(indices) < 2:
        return fail('not enough frames at requested time')
    native = read_selected(row.video_path, set(indices))
    if len(native) != len(indices):
        return fail('missing requested frames')
    scale = None
    if motion_scale:
        path = cache / 'tracks' / f'{row.video_id}.npz'
        ground_path = cache / 'ground' / f'{row.video_id}.json'
        if not path.exists() or not ground_path.exists():
            return fail('motion reference needs existing question-local track')
        from engine import norm
        with np.load(path) as tracks:
            ref = norm(rec['ref_object'])
            if ref not in tracks:
                return fail('motion reference track missing')
            features = tracks[ref]
        solution_path = args.base_dir / 'primary/cache' / args.split / 'measure_solutions.csv'
        if solution_path.exists():
            sol = pd.read_csv(solution_path).set_index('id')
            if row.id in sol.index and str(sol.loc[row.id, 'reason']).startswith('guard:'):
                return fail('motion calibration previously rejected by physical guard')
        scale, _ = meters_per_pixel(rec, features, fps, (len(features)-1)/fps)
        shape = json.loads(ground_path.read_text())['shape']
        h0, w0 = native[indices[0]].shape[:2]
        ratios = np.array([shape[1]/w0, shape[0]/h0])
        if abs(ratios[0]/ratios[1]-1) > .01:
            return fail('tracking/native aspect ratios disagree')
        scale *= float(np.mean(ratios))
    prior_si = to_si(rec['prior_value'], rec.get('prior_unit'))
    readings, accepted = [], []
    prompt = request_text(row, rec, motion_scale)
    for fi in indices:
        frame = native[fi]
        h, w = frame.shape[:2]
        full = resize_longest(frame[None], args.longest)[0]
        def ask(image):
            message = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt + f'\nOriginal frame={fi}, timestamp={fi/fps:.6f}s.'}]}]
            return model.chat(message, images=[Image.fromarray(image)], max_new_tokens=192)
        reply = ask(full)
        p = decode_points(reply, not motion_scale)
        reading = {'frame': int(fi), 'time': fi/fps, 'full_reply': reply}
        readings.append(reading)
        if not p:
            continue
        original = {key: native_points(value, [0,0,w,h]) for key, value in p.items()}
        roi = crop_box(original, w, h)
        x0, y0, x1, y1 = roi
        crop = resize_longest(frame[y0:y1, x0:x1][None], args.longest)[0]
        crop_reply = ask(crop)
        checked = decode_points(crop_reply, not motion_scale)
        reading.update(crop=list(roi), crop_reply=crop_reply)
        if not checked:
            continue
        # Small displayed spans are below reliable coordinate precision even after zoom.
        displayed = {key: native_points(value, [0,0,crop.shape[1],crop.shape[0]]) for key, value in checked.items()}
        if any(span(value) < (12 if key == 'reference' else 8) for key, value in displayed.items()):
            reading['rejected'] = 'too few displayed pixels'
            continue
        corrected = {key: native_points(value, roi) for key, value in checked.items()}
        if not points_agree(original, corrected, floor_px=.006*max(w,h)):
            reading['rejected'] = 'full/crop endpoint identities disagree'
            continue
        a = estimate(original, prior_si, rec['target_unit'], scale)
        b = estimate(corrected, prior_si, rec['target_unit'], scale)
        reading.update(full_value=a, crop_value=b, points={key:value.tolist() for key,value in corrected.items()})
        if agree(a, b, 1.2):
            accepted.append(b)
    result = consensus(accepted, factor=1.25, min_frames=2)
    return dict(id=int(row.id), value=result, reason='accepted' if result is not None else 'insufficient full/crop/frame agreement',
                reference_mode='tracked motion' if motion_scale else 'reference endpoints', request=rec,
                readings=readings, accepted_frames=len(accepted))


def re_search_ruler(prior):
    import re
    return bool(re.search(r'\bruler\b.*\b(calibre|caliber)\b', str(prior), re.I))


if __name__ == '__main__':
    main()
