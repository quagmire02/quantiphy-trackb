"""Portable runner. All caches and outputs stay below final_try/outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUTS = HERE / 'inputs'


def positive(values):
    v = pd.to_numeric(values, errors='coerce')
    return np.isfinite(v) & (v > 0)


def read_values(path, ids, column='value'):
    if not path.exists():
        return pd.Series(np.nan, index=ids, dtype=float)
    frame = pd.read_csv(path)
    if frame.id.duplicated().any() or not set(frame.id).issubset(set(ids)):
        raise ValueError(f'Duplicate or out-of-split IDs: {path}')
    return frame.set_index('id')[column].reindex(ids)


def direct_values(path, df):
    from repairs import direct_value
    out = pd.Series(np.nan, index=df.id, dtype=float)
    if not path.exists():
        return out
    questions = df.set_index('id').question
    videos = df.set_index('id').video_id
    for line in path.read_text(encoding='utf-8').splitlines():
        rec = json.loads(line)
        i = rec['id']
        if i not in out.index or str(rec['video_id']).strip() != videos[i].strip():
            raise ValueError(f'Raw reply ID/video mismatch: {i} in {path}')
        if 'question' in rec and rec['question'] != questions[i]:
            raise ValueError(f'Cached question changed: {i}')
        out[i] = direct_value(rec['reply'], questions[i])
    return out


def assemble(df, candidates):
    value = pd.Series(np.nan, index=df.id, dtype=float)
    source = pd.Series('', index=df.id, dtype=object)
    for name, candidate in candidates:
        candidate = candidate.reindex(df.id)
        use = ~positive(value) & positive(candidate)
        value.loc[use], source.loc[use] = candidate.loc[use], name
    return value, source


def measurement_weak(frame):
    """Triggers for an independent helper; these are not calibrated confidence scores."""
    weak = ~positive(frame.value)
    if 'f_source' in frame:
        weak |= frame.f_source.fillna('').str.contains('moge', case=False)
    if 'moge_used' in frame:
        weak |= frame.moge_used.fillna(False).astype(str).str.lower().isin(['true', '1', '1.0'])
    if 'min_mask_short_px' in frame:
        weak |= frame.min_mask_short_px.le(6)
    if 'max_border_fraction' in frame:
        weak |= frame.max_border_fraction.gt(.5)
    return weak


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--primary-model', help='32B Hugging Face model folder or ID')
    ap.add_argument('--secondary-model', help='4B base or merged trained model folder/ID')
    ap.add_argument('--secondary-adapter', help='QLoRA adapter folder (omit for merged weights)')
    ap.add_argument('--secondary-ground-model', help='Optional unmerged 4B base for grounding if secondary-model is merged')
    ap.add_argument('--split', choices=['test', 'validation'], default='test')
    ap.add_argument('--offline', action='store_true', help='Audit/replay supplied test predictions without models')
    ap.add_argument('--preflight', action='store_true', help='Validate configuration and data; do not load models')
    ap.add_argument('--output', type=Path, default=HERE / 'outputs' / 'guided1')
    ap.add_argument('--four-bit', choices=['0', '1'], default='1')
    ap.add_argument('--ground-adapter', action='store_true', help='Experimental: also use secondary adapter for grounding')
    args = ap.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(HERE / 'outputs'):
        ap.error('--output must be inside final_try/outputs')
    if args.offline and args.split != 'test':
        ap.error('Supplied artifacts cover test only')
    if not args.offline and (not args.primary_model or not args.secondary_model):
        ap.error('Set --primary-model and --secondary-model (and --secondary-adapter for unmerged LoRA)')
    # Only load input metadata here; preflight additionally checks video availability.
    if args.preflight or not args.offline:
        from pipeline.data import load_split
        df = load_split(args.split)
    else:
        df = pd.read_parquet(ROOT / 'data/test/test_dataset.parquet').reset_index(drop=True)
        df['id'] = df.index
        df['prior'] = df.get('ground_truth_prior', df.get('prior'))
        df['category'] = df.inference_type.str[0] + df.video_type.str[1]
    ids = pd.Index(df.id, name='id')
    if ids.duplicated().any():
        raise ValueError('Dataset IDs are not unique')
    digest = hashlib.sha256(df[['id', 'video_id', 'question', 'prior', 'depth_info']].fillna('').to_json().encode()).hexdigest()
    code_files = sorted(list(HERE.glob('*.py')) + list((HERE / 'pipeline').glob('*.py')))
    code_digest = hashlib.sha256(b''.join(p.read_bytes() for p in code_files)).hexdigest()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'preflight'}
    config.update(dataset_sha256=digest, code_sha256=code_digest)
    print(f'{args.split}: {len(df)} questions, {df.video_id.nunique()} videos; dataset {digest[:12]}', flush=True)
    if args.preflight:
        for key in ('primary_model', 'secondary_model', 'secondary_adapter', 'secondary_ground_model'):
            path = getattr(args, key)
            if path and (path.startswith(('/', '.', '~')) or ':' in path) and not Path(path).expanduser().exists():
                raise FileNotFoundError(f'{key}: {path}')
        import torch
        from transformers import AutoModelForImageTextToText, Sam2VideoModel
        from moge.model.v2 import MoGeModel
        if args.secondary_adapter:
            import peft
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for inference')
        print('Preflight passed: dataset, videos, CUDA and model libraries available. Weights are not loaded.')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Configuration, dataset or code changed. Choose a new --output under final_try/outputs.')
    manifest.write_text(json.dumps(config, indent=2), encoding='utf-8')

    def cache(name):
        return args.output / name / 'cache' / args.split

    def phase(name, kind, model, adapter=None, selected=None, hints=None, proposals=None):
        tree = args.output / name
        env = os.environ.copy()
        for key in ('QUANTIPHY_VLM_ADAPTER', 'QUANTIPHY_VLM_QUANT_VISION', 'QUANTIPHY_VLM_MAX_GPU_GB'):
            env.pop(key, None)
        env.update(QUANTIPHY_RESULTS=str(tree), QUANTIPHY_VLM=model,
                   QUANTIPHY_VLM_4BIT=args.four_bit, PYTHONUNBUFFERED='1',
                   QUANTIPHY_GUIDED='1' if name.startswith('primary') else '0')
        if adapter:
            env['QUANTIPHY_VLM_ADAPTER'] = adapter
        tree.mkdir(parents=True, exist_ok=True)
        (tree / 'cache').mkdir(exist_ok=True)
        cmd = [sys.executable, str(HERE / 'worker.py'), '--phase', kind, '--split', args.split]
        if selected is not None:
            path = tree / f'{kind}_ids.json'
            path.write_text(json.dumps([int(i) for i in selected]))
            cmd += ['--ids', str(path)]
        if hints is not None:
            path = tree / 'same_question_hypotheses.json'
            path.write_text(json.dumps(hints, allow_nan=False))
            cmd += ['--hints', str(path)]
        if proposals is not None:
            cmd += ['--proposal-cache', str(proposals)]
        print(f'Running {name}/{kind}: {model}; selected={len(selected) if selected is not None else "all"}', flush=True)
        subprocess.run(cmd, env=env, cwd=ROOT, check=True)

    baseline = pd.Series(np.nan, index=ids)
    protected = baseline.copy()
    saved_secondary = baseline.copy()
    if args.split == 'test':
        old = pd.read_csv(INPUTS / 'v17_sources.csv').set_index('id')
        if not old.index.equals(ids) or not np.array_equal(old.video_id.str.strip(), df.video_id.str.strip()):
            raise ValueError('Saved baseline IDs/videos do not align with the dataset')
        if not np.array_equal(old.question.str.strip().str.lower(), df.question.str.strip().str.lower()):
            raise ValueError('Saved baseline questions do not match dataset; refusing to reuse stale predictions')
        baseline = old.parsed_value
        protected = baseline.where(old.source.str.contains('-measure', regex=False))
        if args.offline:
            saved_secondary = read_values(INPUTS / '4b_measure.csv', ids)
    if args.offline:
        primary = read_values(INPUTS / '32b_measure.csv', ids)
        secondary = saved_secondary
        primary_direct = direct_values(INPUTS / '32b_direct.jsonl', df)
        saved = pd.read_csv(INPUTS / 'v16_sources.csv').set_index('id').reindex(ids)
        secondary_direct = saved.parsed_value.where(saved.source.eq('vlm'))
    else:
        phase('primary', 'measure', args.primary_model)
        primary = read_values(cache('primary') / 'measure_solutions.csv', ids)
        primary_table = pd.read_csv(cache('primary') / 'measure_solutions.csv').set_index('id').reindex(ids)
        weak = measurement_weak(primary_table)
        pending = ids[weak]
        phase('secondary', 'measure', args.secondary_ground_model or args.secondary_model,
              args.secondary_adapter if args.ground_adapter else None, selected=pending)
        fresh_secondary = read_values(cache('secondary') / 'measure_solutions.csv', ids)
        secondary = fresh_secondary
        phase('primary_assisted', 'measure', args.primary_model, selected=ids[weak], proposals=cache('secondary'))
        assisted = read_values(cache('primary_assisted') / 'measure_solutions.csv', ids).where(weak)
        pending = ids[~positive(primary) & ~positive(secondary)]
        phase('primary', 'direct', args.primary_model, selected=pending)
        primary_direct = direct_values(cache('primary') / 'direct.jsonl', df)
        # Every failed measurement is a weak case worth a trained direct opinion.
        # Collect paired answers; choosing 4B for every 2D fallback is an explicit ablation.
        pending = ids[~positive(primary) & ~positive(secondary)]
        phase('secondary_direct', 'direct', args.secondary_model, args.secondary_adapter, selected=pending)
        secondary_direct = direct_values(cache('secondary_direct') / 'direct.jsonl', df)
        # The 4B may guide 32B only on the SAME question and only for weak/conflicting cases.
        disagree = positive(primary_direct) & positive(secondary_direct) & ((primary_direct / secondary_direct > 2) | (secondary_direct / primary_direct > 2))
        review_ids = ids[(~positive(primary_direct) & positive(secondary_direct)) | disagree]
        hypotheses = {str(int(i)): {'32B': float(primary_direct[i]) if positive(pd.Series([primary_direct[i]])).iloc[0] else None,
                                   '4B': float(secondary_direct[i])} for i in review_ids}
        phase('primary_review', 'direct', args.primary_model, selected=review_ids, hints=hypotheses)
        reviewed = direct_values(cache('primary_review') / 'direct.jsonl', df)
    metadata = pd.Series(np.nan, index=ids, dtype=float)
    from repairs import camera_distance
    parse_path = INPUTS / '32b_parse_question.jsonl' if args.offline else cache('primary') / 'parse_question.jsonl'
    records = {r['id']: r for r in map(json.loads, parse_path.read_text().splitlines())} if parse_path.exists() else {}
    for row in df.itertuples():
        value = camera_distance(row, records.get(row.id))
        if value is not None:
            metadata[row.id] = value
    candidates = [('metadata-camera-distance', metadata), ('32B-measure', primary), ('4B-measure', secondary),
                  ('32B-direct', primary_direct), ('4B-direct', secondary_direct)]
    two_d = pd.Series(df.category.str.endswith('2').values, index=ids)
    trained_2d = [('metadata-camera-distance', metadata), ('32B-measure', primary), ('4B-measure', secondary),
                  ('trained-4B-direct-2D', secondary_direct.where(two_d)),
                  ('32B-direct', primary_direct), ('4B-direct', secondary_direct)]
    if args.offline:
        candidates += [('v17-fallback', baseline)]
        options_list = [('candidate_fresh', candidates), ('candidate_preserved', [('v17-measure', protected)] + candidates)]
    else:
        guided_candidates = candidates[:1] + [('32B-measure-with-4B-box-help', assisted)] + candidates[1:3] + [('32B-reviewed-with-4B-hypothesis', reviewed)] + candidates[3:]
        options_list = [('candidate_32b_led', candidates), ('candidate_guided', guided_candidates), ('candidate_trained_2d', trained_2d)]
        if args.split == 'test':
            # Explicit historical comparison, separate from independent inference candidates.
            # It inherits v17's provenance; do not describe it as rule-verified.
            options_list.append(('candidate_v17_union', [('historical-v17-measure', protected)] + guided_candidates + [('historical-v17-fallback', baseline)]))
    diagnostic = df[['id', 'video_id', 'question', 'category']].copy()
    for label, series in [('32b_measure', primary), ('4b_measure', secondary),
                          ('32b_direct', primary_direct), ('4b_direct', secondary_direct)]:
        diagnostic[label] = series.values
    diagnostic['measurement_ratio_32b_to_4b'] = primary.values / secondary.values
    diagnostic['metadata_camera_distance'] = metadata.values
    if not args.offline:
        diagnostic['32b_reviewed'] = reviewed.values
        diagnostic['32b_assisted_measure'] = assisted.values
        diagnostic['helper_triggered'] = weak.values
    diagnostic.to_csv(args.output / 'diagnostics.csv', index=False)
    for name, options in options_list:
        values, sources = assemble(df, options)
        if not args.offline:
            # Last resort depends exclusively on this item's prior and clip duration.
            from repairs import prior_scale_fallback
            for row in df.loc[~positive(values).values].itertuples():
                value = prior_scale_fallback(row)
                if value is not None:
                    values.loc[row.id], sources.loc[row.id] = value, 'same-question-prior-estimate'
        if not positive(values).all():
            raise ValueError('Incomplete predictions; no submission written')
        frame = df[['id', 'video_id', 'question', 'category']].copy()
        frame['parsed_value'], frame['source'] = values.values, sources.values
        frame['baseline_value'] = baseline.values
        frame['changed_from_v17'] = ~np.isclose(values, baseline, equal_nan=True)
        frame.to_csv(args.output / f'{name}_sources.csv', index=False)
        if args.split == 'test':
            template_path = ROOT / 'data/template/quantiphy_submission_template.csv'
            template = pd.read_csv(template_path if template_path.exists() else INPUTS / 'template.csv')
            if len(template) != len(df) or not np.array_equal(template.id, df.id + 1) or not np.array_equal(template.video_id.str.strip(), df.video_id.str.strip()):
                raise ValueError('Template IDs/video order do not match dataset')
            template['parsed_value'] = values.values
            template.to_csv(args.output / f'{name}.csv', index=False)
        else:
            from pipeline.scoring import quick_report
            quick_report(df, values.values, name)
        print(name, '\n', pd.crosstab(df.category, sources.values).to_string(), flush=True)
    print(f'Wrote {args.output}; offline replay only.' if args.offline else f'Wrote {args.output}; new candidates use independent question scopes; historical v17 union is separate.')


if __name__ == '__main__':
    main()
