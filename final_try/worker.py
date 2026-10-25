"""One isolated process per GPU phase; model memory is released on process exit."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['measure', 'direct'], required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--ids', type=Path)
    ap.add_argument('--hints', type=Path)
    ap.add_argument('--proposal-cache', type=Path)
    ap.add_argument('--stages', default='parse,ground,track,findmissing,anchors,refix,refine,geom,solve')
    args = ap.parse_args()
    # Imports follow environment configuration set by the parent process.
    import numpy as np
    import pandas as pd
    import torch
    from pipeline.data import load_split
    from pipeline.scoring import RESULTS
    import engine

    full = load_split(args.split)
    if args.ids:
        ids = set(json.loads(args.ids.read_text()))
        df = full[full.id.isin(ids)]
    else:
        df = full
    cache = RESULTS / 'cache' / args.split
    cache.mkdir(parents=True, exist_ok=True)
    if df.empty:
        if args.phase == 'measure':
            pd.DataFrame(columns=['id', 'value', 'reason']).to_csv(cache / 'measure_solutions.csv', index=False)
        return
    if args.phase == 'measure':
        from guided import ContextVLM, ground_one, scopes
        import gc
        import pipeline.track as tracking
        import pipeline.depth as depth
        # Cross-question calibration and semantic caches are deliberately unavailable.
        engine.MOGE_CORR_PATH = cache / 'disabled_cross_question_calibration.json'
        engine.moge_corrections = lambda: {}
        stages = args.stages.split(',')
        if {'parse', 'ground'} & set(stages):
            from pipeline.vlm import QwenVL
            vlm = QwenVL()
            for one, item_cache in scopes(df, cache):
                item_cache.mkdir(parents=True, exist_ok=True)
                row = next(one.itertuples())
                # A single row reaches every inherited groupby/video routine.
                if 'parse' in stages:
                    engine.stage_parse(one, item_cache, ContextVLM(vlm, row))
                if 'ground' in stages:
                    ground_one(engine, one, engine.load_parsed(one, item_cache), item_cache, vlm, args.proposal_cache)
            del vlm
            torch.cuda.empty_cache()

        def track_all(refine=False):
            # Reuse model WEIGHTS across questions, never prompts, masks or tracking state.
            original = tracking.Sam2Tracker
            shared = original()
            tracking.Sam2Tracker = lambda: shared
            try:
                for one, item_cache in scopes(df, cache):
                    if refine:
                        engine.stage_refine(one, engine.load_parsed(one, item_cache), item_cache)
                    else:
                        engine.stage_track(one, engine.load_parsed(one, item_cache), item_cache)
            finally:
                tracking.Sam2Tracker = original
                del shared
                gc.collect()
                torch.cuda.empty_cache()

        for stage in stages:
            if stage in ('parse', 'ground', 'solve'):
                continue
            if stage == 'track':
                track_all()
            elif stage in ('findmissing', 'anchors', 'refix'):
                from pipeline.vlm import QwenVL
                vlm = QwenVL()
                for one, item_cache in scopes(df, cache):
                    row = next(one.itertuples())
                    getattr(engine, 'stage_' + stage)(one, engine.load_parsed(one, item_cache), item_cache, ContextVLM(vlm, row))
                del vlm
                torch.cuda.empty_cache()
                track_all()
            elif stage == 'refine':
                track_all(refine=True)
            elif stage == 'geom':
                if not (df.video_type.str[1] == '3').any():
                    continue
                original = depth.MoGe
                shared = original()
                depth.MoGe = lambda: shared
                try:
                    for one, item_cache in scopes(df[df.video_type.str[1] == '3'], cache):
                        engine.stage_geom(one, item_cache)
                finally:
                    depth.MoGe = original
                    del shared
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                raise ValueError(f'Unknown stage: {stage}')
        if 'solve' in stages:
            from repairs import camera_distance
            solutions, questions = [], []
            for one, item_cache in scopes(df, cache):
                row = next(one.itertuples())
                parsed = engine.load_parsed(one, item_cache)
                rec = engine.load_jsonl(item_cache / 'parse_question.jsonl').get(row.id)
                if rec:
                    questions.append(rec)
                sol = engine.stage_solve(one, parsed, item_cache)
                # Observable weakness signals, computed from this question's own tracks.
                track_path = item_cache / 'tracks' / f'{row.video_id}.npz'
                if track_path.exists():
                    from pipeline.geometry import F
                    short, border = [], []
                    with np.load(track_path) as tracks:
                        for key in tracks.files:
                            f = tracks[key]
                            present = f[:, F['present']] == 1
                            if present.any():
                                f = f[present]
                                short.append(float(np.median(np.minimum(f[:, F['x2']]-f[:, F['x1']], f[:, F['y2']]-f[:, F['y1']]))))
                                border.append(float(np.mean(f[:, F['touches_border']] == 1)))
                    if short:
                        sol['min_mask_short_px'], sol['max_border_fraction'] = min(short), max(border)
                value = camera_distance(row, parsed.get(row.id))
                if value is not None and np.isfinite(value) and value > 0:
                    sol.loc[sol.id == row.id, ['value', 'reason']] = [value, 'metadata camera distance']
                solutions.append(sol)
            sol = pd.concat(solutions, ignore_index=True)
            (cache / 'parse_question.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in questions))
            sol.to_csv(cache / 'measure_solutions.csv', index=False)
            print(f'Measured {sol.value.notna().sum()}/{len(sol)}', flush=True)
        return

    from pipeline.official import SYSTEM_PROMPT, zero_shot_text
    from pipeline.video import read_frames, sample_indices, resize_longest
    from pipeline.vlm import QwenVL
    from repairs import direct_value
    path = cache / 'direct.jsonl'
    records = engine.load_jsonl(path)
    todo = df[~df.id.isin(records)]
    if todo.empty:
        return
    vlm = QwenVL()
    hints = json.loads(args.hints.read_text()) if args.hints else {}
    with path.open('a', encoding='utf-8') as fh:
        for vid, group in todo.groupby('video_id', sort=False):
            frames = read_frames(group.video_path.iloc[0], longest=896)
            idx = sample_indices(len(frames), 16)
            sampled = resize_longest(frames[idx], 448)
            for row in group.itertuples():
                text = zero_shot_text(row.question, row.prior, row.video_type, row.depth_info)
                system, budget = SYSTEM_PROMPT, 128
                if os.environ.get('QUANTIPHY_GUIDED') == '1':
                    system = ('You solve a single quantitative physics question from its video and supplied prior. '
                              'First identify the reference and target, distinguish their identities and times, '
                              'convert units, and use the reference to calibrate the requested quantity. '
                              'Use camera ranges only for their named objects. Do not substitute a size for a traveled path. '
                              'You may write concise calculations. End with exactly Final Answer: <number> <unit>. '
                              'If measurement is uncertain, give your best supported estimate.')
                    budget = 384
                hint = hints.get(str(row.id))
                if hint:
                    text += ('\nTwo unverified estimates produced independently for THIS SAME QUESTION, in its requested unit: '
                             + json.dumps(hint) + '\nThey are hypotheses, not ground truth. Recheck the video, prior scale, '
                             'object identity, time and units. Correct either estimate if needed; do not simply average them.')
                reply = vlm.ask(system, text, sampled, idx, len(frames), row.fps, max_new_tokens=budget)['text']
                value = direct_value(reply, row.question)
                rec = {'id': int(row.id), 'video_id': vid, 'question': row.question, 'reply': reply,
                       'value': value if np.isfinite(value) else None}
                fh.write(json.dumps(rec, allow_nan=False) + '\n')
                fh.flush()
                print(f'Direct {row.id}: {reply[:100]}', flush=True)


if __name__ == '__main__':
    main()
