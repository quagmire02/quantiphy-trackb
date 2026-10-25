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
        # Measure complete clips: clip guards and reference calibration need every question.
        videos = set(full.loc[full.id.isin(ids), 'video_id'])
        df = full[full.video_id.isin(videos)] if args.phase == 'measure' else full[full.id.isin(ids)]
    else:
        df = full
    cache = RESULTS / 'cache' / args.split
    cache.mkdir(parents=True, exist_ok=True)
    if df.empty:
        if args.phase == 'measure':
            pd.DataFrame(columns=['id', 'value', 'reason']).to_csv(cache / 'measure_solutions.csv', index=False)
        return
    if args.phase == 'measure':
        stages = args.stages.split(',')
        if {'parse', 'ground'} & set(stages):
            from pipeline.vlm import QwenVL
            vlm = QwenVL()
            if 'parse' in stages:
                engine.stage_parse(df, cache, vlm)
            if 'ground' in stages:
                engine.stage_ground(df, engine.load_parsed(df, cache), cache, vlm)
            del vlm
            torch.cuda.empty_cache()
        parsed = engine.load_parsed(df, cache)
        for stage in stages:
            if stage in ('parse', 'ground', 'solve'):
                continue
            if stage == 'track':
                engine.stage_track(df, parsed, cache)
            elif stage in ('findmissing', 'anchors', 'refix'):
                from pipeline.vlm import QwenVL
                vlm = QwenVL()
                getattr(engine, 'stage_' + stage)(df, parsed, cache, vlm)
                del vlm
                torch.cuda.empty_cache()
                engine.stage_track(df, parsed, cache)
            elif stage == 'refine':
                engine.stage_refine(df, parsed, cache)
            elif stage == 'geom':
                engine.stage_geom(df, cache)
            elif stage == 'mogecal':
                engine.stage_mogecal(df, parsed, cache)
            else:
                raise ValueError(f'Unknown stage: {stage}')
        if 'solve' in stages:
            if args.split == 'test' and not args.ids and 'geom' in stages:
                engine.stage_mogecal(df, parsed, cache)
            sol = engine.stage_solve(df, parsed, cache)
            from repairs import camera_distance
            for row in df.itertuples():
                value = camera_distance(row, parsed.get(row.id))
                if value is not None and np.isfinite(value) and value > 0:
                    sol.loc[sol.id == row.id, ['value', 'reason']] = [value, 'metadata camera distance']
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
    with path.open('a', encoding='utf-8') as fh:
        for vid, group in todo.groupby('video_id', sort=False):
            frames = read_frames(group.video_path.iloc[0], longest=896)
            idx = sample_indices(len(frames), 16)
            sampled = resize_longest(frames[idx], 448)
            for row in group.itertuples():
                reply = vlm.ask(SYSTEM_PROMPT, zero_shot_text(row.question, row.prior, row.video_type, row.depth_info),
                                sampled, idx, len(frames), row.fps, max_new_tokens=128)['text']
                value = direct_value(reply, row.question)
                rec = {'id': int(row.id), 'video_id': vid, 'question': row.question, 'reply': reply,
                       'value': value if np.isfinite(value) else None}
                fh.write(json.dumps(rec, allow_nan=False) + '\n')
                fh.flush()
                print(f'Direct {row.id}: {reply[:100]}', flush=True)


if __name__ == '__main__':
    main()
