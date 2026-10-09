"""Mask verifier: ask the VLM whether each tracked object's box really is that object.

For every tracked object of a clip: frame where the mask is present (nearest the middle), the full
frame with the tracked box drawn in red + a zoomed crop around it. The VLM picks one of:
  A the box tightly holds the whole <object>
  B it is a different object (or nothing)
  C it holds only part of the <object>
  D it holds the <object> plus a lot of other things (background, other objects)
Cached in results/cache/<split>/verify.jsonl.

  python experiments/verify_masks.py --split validation
  python experiments/verify_masks.py --split validation --evaluate measure_v20
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
from pipeline.geometry import F  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402
from pipeline.video import read_frames  # noqa: E402

PROMPT = ("The first image is a video frame with a red box drawn on it; the second image is a close-up of "
          "the same red box. Which statement is true?\n"
          "A) The red box tightly contains the whole {obj}.\n"
          "B) The red box contains a different object, not the {obj}.\n"
          "C) The red box contains only a part of the {obj}.\n"
          "D) The red box contains the {obj} but also a lot of other things (background or other objects).\n"
          "Answer with one letter.")


def views(frame, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    full = frame.copy()
    t = max(2, int(max(frame.shape[:2]) / 300))
    cv2.rectangle(full, (x1, y1), (x2, y2), (255, 0, 0), t)
    w, h = x2 - x1, y2 - y1
    m = int(max(w, h) * 0.4) + 8
    H, W = frame.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
    crop = full[cy1:cy2, cx1:cx2]
    s = 448 / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(16, int(crop.shape[1] * s)), max(16, int(crop.shape[0] * s))))
    return full, crop


def run(split):
    from pipeline.vlm import QwenVL
    df = m2.load_split(split)
    cache = RESULTS / "cache" / split
    out = cache / "verify.jsonl"
    done = {(r["video_id"], r["key"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    vlm = QwenVL(m2.VLM_ID)
    with out.open("a") as fh:
        for vid, items in tqdm(list(df.groupby("video_id")), desc="verify"):
            tp, gp = cache / "tracks" / f"{vid}.npz", cache / "ground" / f"{vid}.json"
            if not tp.exists() or not gp.exists():
                continue
            feats = dict(np.load(tp))
            if not feats:
                continue
            objs = json.loads(gp.read_text())["objects"]
            frames = None
            for key, f in feats.items():
                if (vid, key) in done:
                    continue
                pres = np.flatnonzero(f[:, F["present"]] == 1)
                if not len(pres):
                    continue
                if frames is None:
                    frames = read_frames(items["video_path"].iloc[0], m2.TRACK_LONGEST)
                fi = int(pres[np.argmin(np.abs(pres - len(f) // 2))])
                box = f[fi, [F["x1"], F["y1"], F["x2"], F["y2"]]]
                phrase = objs.get(key.split("#")[0], {}).get("phrase", key)
                full, crop = views(frames[fi], box)
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                                     {"type": "text", "text": PROMPT.format(obj=phrase)}]}]
                reply = vlm.chat(msgs, images=[Image.fromarray(full), Image.fromarray(crop)], max_new_tokens=4)
                m = re.search(r"\b([ABCD])\b", reply.upper())
                rec = {"video_id": vid, "key": key, "phrase": phrase, "frame": fi, "reply": reply,
                       "verdict": m.group(1) if m else "?"}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()


def evaluate(split, sol_name):
    df = m2.load_split(split)
    cache = RESULTS / "cache" / split
    parsed = m2.load_parsed(df, cache)
    ver = pd.DataFrame(map(json.loads, (cache / "verify.jsonl").read_text().splitlines()))
    vmap = {(r.video_id, r.key): r.verdict for r in ver.itertuples()}
    print("verdicts:", ver.verdict.value_counts().to_dict())
    sol = pd.read_csv(cache / f"{sol_name}_solutions.csv").merge(df, on="id")
    sol = sol[sol.value.notna()].copy()
    rows = []
    for r in sol.itertuples():
        ref, tg = m2.object_keys(parsed[r.id])
        tv = [vmap.get((r.video_id, k), "?") for k in tg]
        rows.append({"id": r.id, "ref_v": vmap.get((r.video_id, ref), "?"),
                     "tgt_v": "".join(tv), "mra": float(np.asarray(mra(pd.Series([r.value]), pd.Series([r.answer])))[0]),
                     "q": r.question[:60], "vid": r.video_id})
    e = pd.DataFrame(rows)
    e["tgt_ok"] = e.tgt_v.str.fullmatch("A+")
    e["ref_ok"] = e.ref_v == "A"
    print(f"\nmeasured items {len(e)}: MRA {e.mra.mean():.3f}")
    print(e.groupby("tgt_v").mra.agg(["mean", "count"]).round(2).to_string())
    print(e.groupby("ref_v").mra.agg(["mean", "count"]).round(2).to_string())
    for name, flag in [("target flagged", ~e.tgt_ok), ("reference flagged", ~e.ref_ok), ("either flagged", ~(e.tgt_ok & e.ref_ok))]:
        print(f"{name}: n={flag.sum()}, MRA flagged {e.mra[flag].mean():.3f} vs passed {e.mra[~flag].mean():.3f}")
    print("\nflagged items:"); print(e[~(e.tgt_ok & e.ref_ok)].sort_values("mra").to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--evaluate", default=None, help="solutions name to evaluate verdicts against (validation)")
    a = ap.parse_args()
    evaluate(a.split, a.evaluate) if a.evaluate else run(a.split)
