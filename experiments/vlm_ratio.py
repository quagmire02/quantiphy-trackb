"""Restructured VLM fallback: ask the VLM for a *comparison* with the reference object, do the
arithmetic in code (answer = ratio x prior), so the answer follows the given value by construction.

  size  vs size prior        "how many times the <ref>'s <q>"         -> ratio * prior
  speed vs size prior        "how many <ref> <q>s per second"         -> ratio * prior / 1 s
  speed vs speed prior       "how many times the <ref>'s speed"       -> ratio * prior
  acceleration vs acc prior  "how many times the <ref>'s acceleration"-> ratio * prior
  size  vs speed prior       "how many times the distance the <ref> moves in one second" -> ratio * prior * 1 s
Anything else keeps the plain VLM answer.

  python experiments/vlm_ratio.py --split validation            # run (resumable)
  python experiments/vlm_ratio.py --split validation --evaluate  # compare with the plain VLM answers
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
from pipeline.official import build_context_prefix  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402
from pipeline.units import from_si, to_si  # noqa: E402
from pipeline.video import read_frames, resize_longest, sample_indices  # noqa: E402

SIZE = {"length", "width", "height", "diameter", "radius", "thickness", "size"}
LENGTHY = SIZE | {"distance", "displacement", "path_length", "path_diameter"}
SYSTEM = ("You compare objects in a short video. Look carefully at the apparent sizes and motions of the "
          "objects in the frames. Reply with ONE positive number only (e.g. 0.4, 2.5, 12), no units, no words.")
NAME = "vlm_ratio_f16_px448"


def plan(rec: dict, question: str):
    """(prompt text, multiplier in SI) or None if this combination is not handled."""
    if not rec or not rec.get("ok"):
        return None
    tq, rq, ref = rec["target_quantity"], rec["ref_quantity"], rec["ref_object"]
    prior_si = abs(to_si(rec["prior_value"], rec["prior_unit"]))
    rqn = {"size": "size"}.get(rq, rq)
    ask = f'Question about the video: "{question.strip()}"\nDo not answer it directly. Instead, '
    if tq in LENGTHY and rq in SIZE:
        return ask + f"tell how many times the {rqn} of the {ref} the requested quantity is.", prior_si
    if tq == "speed" and rq in SIZE:
        return ask + (f"tell how many times the {rqn} of the {ref} the object moves in one second "
                      f"(its speed measured in {ref} {rqn}s per second)."), prior_si
    if tq == "speed" and rq == "speed":
        return ask + f"tell how many times the speed of the {ref} the requested speed is.", prior_si
    if tq == "acceleration" and rq == "acceleration":
        return ask + f"tell how many times the acceleration of the {ref} the requested acceleration is.", prior_si
    if tq in LENGTHY and rq == "speed":
        return ask + (f"tell how many times the distance the {ref} moves in one second the requested "
                      f"quantity is."), prior_si
    return None


def parse_ratio(reply: str) -> float:
    m = re.search(r"\d*\.?\d+(?:[eE][-+]?\d+)?", reply.replace(",", "."))
    v = float(m.group(0)) if m else np.nan
    return v if np.isfinite(v) and v > 0 else np.nan


def run(split, ids=None):
    from pipeline.vlm import QwenVL
    df = m2.load_split(split)
    cache = RESULTS / "cache" / split
    parsed = m2.load_parsed(df, cache)
    if ids is not None:
        df = df[df.id.isin(ids)]
    out = RESULTS / f"{split}_raw" / f"{NAME}.jsonl"
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    todo = df[~df.id.isin(done) & df.apply(lambda r: plan(parsed.get(r.id), r.question) is not None, axis=1)]
    print(f"{NAME}: {len(todo)} items to ask")
    if not len(todo):
        return
    vlm = QwenVL(m2.VLM_ID)
    with out.open("a") as fh:
        for vid, g in tqdm(todo.groupby("video_id", sort=False), desc="ratio"):
            allf = read_frames(g.video_path.iloc[0], longest=896)
            idx = sample_indices(len(allf), 16)
            frames = resize_longest(allf[idx], 448)
            for r in g.itertuples():
                text, mult = plan(parsed[r.id], r.question)
                depth = r.depth_info if str(r.video_type)[1] == "3" else None
                text = build_context_prefix(r.prior, depth) + text
                res = vlm.ask(SYSTEM, text, frames, idx, len(allf), r.fps, max_new_tokens=12)
                ratio = parse_ratio(res["text"])
                value = from_si(ratio * mult, parsed[r.id]["target_unit"]) if np.isfinite(ratio) else np.nan
                fh.write(json.dumps({"id": int(r.id), "reply": res["text"], "ratio": ratio, "value": value}) + "\n")
                fh.flush()


def load(split) -> pd.Series:
    p = RESULTS / f"{split}_raw" / f"{NAME}.jsonl"
    recs = [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    return pd.Series({r["id"]: r["value"] for r in recs}, dtype=float)


def evaluate():
    df = m2.load_split("validation").set_index("id")
    ratio = load("validation")
    vlm = pd.read_csv(RESULTS / "validation_predictions/Qwen3-VL-4B_video_f16_px448.csv")
    vlm = vlm.set_index(vlm.columns[0])["parsed_value"]
    src = pd.read_csv(RESULTS / "validation_predictions/val_measure_v20.csv")
    src = src.set_index(src.columns[0])
    both = ratio.dropna().index
    a = df.answer[both]
    print(f"items with a ratio answer: {len(both)} of {len(df)}")
    print(f"  all of them:   ratio MRA {mra(ratio[both], a).mean():.3f}   plain VLM {mra(vlm[both], a).mean():.3f}")
    fb = [i for i in both if src.at[i, "source"] != "measure"]
    print(f"  fallback ones ({len(fb)}): ratio {mra(ratio[fb], df.answer[fb]).mean():.3f}   plain VLM {mra(vlm[fb], df.answer[fb]).mean():.3f}")
    for c in ["S2", "D2", "S3", "D3"]:
        ii = [i for i in both if df.at[i, "category"] == c]
        if ii:
            print(f"  {c} (n={len(ii)}): ratio {mra(ratio[ii], df.answer[ii]).mean():.3f}   plain VLM {mra(vlm[ii], df.answer[ii]).mean():.3f}")
    full = src["parsed_value"].copy()
    for i in full.index:
        if src.at[i, "source"] != "measure" and i in ratio.index and np.isfinite(ratio[i]):
            full[i] = ratio[i]
    cats = {c: round(mra(full[df.category == c], df.answer[df.category == c]).mean() * 100, 1) for c in ["S2", "D2", "S3", "D3"]}
    print(f"validation with ratio fallback: avg {np.mean(list(cats.values())):.1f} {cats} (v20: 62.6)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--unsolved-from", default=None, help="only items unsolved in this solutions CSV")
    a = ap.parse_args()
    if a.evaluate:
        evaluate()
    else:
        ids = None
        if a.unsolved_from:
            s = pd.read_csv(a.unsolved_from)
            ids = set(s.loc[s.value.isna(), "id"])
        run(a.split, ids)
