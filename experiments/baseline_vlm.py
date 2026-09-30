"""Experiment 1: local Qwen3-VL zero-shot baseline with the official QuantiPhy prompt.

  --mode video     official "Original" setting (video + prior [+ depth for 3D])
  --mode no_video  official "No video" setting (prior-only; tests reliance on world knowledge)

Resumable: raw replies are appended to results/<split>_raw/<name>.jsonl; rerun to continue.

Example:
  python experiments/baseline_vlm.py --split validation --mode video --max-frames 32 --longest 640
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.official import SYSTEM_PROMPT, parse_number, zero_shot_text  # noqa: E402
from pipeline.scoring import RESULTS, official_score, quick_report, write_predictions  # noqa: E402
from pipeline.video import read_frames, resize_longest, sample_indices  # noqa: E402
from pipeline.vlm import QwenVL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--mode", default="video", choices=["video", "no_video"])
    # 16 frames @ 640px: ~4.5 GiB peak, ~4.6 s/question on RTX 2060 (32 frames spills past 6 GB -> 4x slower)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--longest", type=int, default=640, help="longest frame side in pixels fed to the VLM")
    ap.add_argument("--limit", type=int, default=None, help="only first N rows (smoke test)")
    ap.add_argument("--unsolved-from", default=None,
                    help="solutions CSV (id,value) from measure_2d.py: only run rows it could not solve")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    short = args.model.split("/")[-1].replace("-Instruct", "").replace("-bnb-4bit", "")
    name = args.name or (f"{short}_{args.mode}" + (f"_f{args.max_frames}_px{args.longest}" if args.mode == "video" else ""))
    df = load_split(args.split)
    if args.unsolved_from:
        sol = pd.read_csv(args.unsolved_from)
        df = df[df["id"].isin(set(sol["id"]) - set(sol.loc[sol["value"].notna(), "id"]))
                | ~df["id"].isin(sol["id"])]
    if args.limit:
        df = df.head(args.limit)

    raw_path = RESULTS / f"{args.split}_raw" / f"{name}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
    todo = df[~df["id"].isin(done)]
    print(f"{name}: {len(df)} rows, {len(done)} already done, {len(todo)} to run")

    if len(todo):
        vlm = QwenVL(args.model)
        torch.cuda.reset_peak_memory_stats()
        with raw_path.open("a") as fh:
            # Group by video so each clip is decoded once.
            for video_id, group in tqdm(todo.groupby("video_id", sort=False), desc="videos"):
                frames = idx = None
                if args.mode == "video":
                    # Downscale while decoding: a long 4K clip decoded at full size exceeds WSL's RAM.
                    all_frames = read_frames(group["video_path"].iloc[0], longest=2 * args.longest)
                    idx = sample_indices(len(all_frames), args.max_frames)
                    frames = resize_longest(all_frames[idx], args.longest)
                    total = len(all_frames)
                for row in group.itertuples():
                    text = zero_shot_text(row.question, row.prior, row.video_type, row.depth_info)
                    res = vlm.ask(SYSTEM_PROMPT, text, frames, idx,
                                  total if frames is not None else None, row.fps)
                    rec = {"id": int(row.id), "video_id": video_id, "reply": res["text"],
                           "value": parse_number(res["text"]), "tokens": res["n_input_tokens"],
                           "seconds": res["seconds"]}
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    done[rec["id"]] = rec
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    recs = pd.DataFrame([done[i] for i in df["id"]])
    print(f"mean input tokens {recs['tokens'].mean():.0f}, mean seconds/question {recs['seconds'].mean():.2f}")
    print("sample replies:", recs["reply"].head(8).tolist())
    write_predictions(df, recs["value"].values, name, args.split, extra={"raw_response": recs["reply"]})
    if args.split == "validation" and not args.limit:
        quick_report(df, recs["value"].values, name)
        official_score()


if __name__ == "__main__":
    main()
